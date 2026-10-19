# -*- coding: utf-8 -*-
"""
放量③：可切换文本编码器（对照试验用，默认行为不变）
- 环境变量 ENTITY_ENCODER=tfidf（默认，复现试生产全部数字）| specter2（SPECTER2 base，768 维 [CLS]）
- SPECTER2 为外部预训练模型，冻结、不在本语料拟合，无泄漏风险；
  编码结果按文本哈希落盘缓存（data/processed/specter2_cache.pkl），重复训练/评测零重编码。
- 输出维度：tfidf 512 / specter2 768；实体特征 = 文本向量 + 类型 one-hot(6) + 语境 one-hot(4)。
"""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "specter2_base"
CACHE_FILE = BASE / "specter2_cache.pkl"

BACKEND = os.environ.get("ENTITY_ENCODER", "tfidf").strip().lower()
TEXT_DIM = 768 if BACKEND == "specter2" else 512

_model = None
_tok = None


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(c: dict):
    with CACHE_FILE.open("wb") as f:
        pickle.dump(c, f)


def _specter_encode(texts: list[str]) -> np.ndarray:
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModel.from_pretrained(str(MODEL_DIR)).eval()
        print(f"[text_encoder] SPECTER2 已加载（{MODEL_DIR.name}）", flush=True)
    import torch
    cache = _load_cache()
    keys = [hashlib.sha1((t or "").encode("utf-8")).hexdigest() for t in texts]
    todo = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in cache]
    if todo:
        B = 16
        for s in range(0, len(todo), B):
            batch = [t or " " for _, t in todo[s:s + B]]
            enc = _tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = _model(**enc).last_hidden_state[:, 0, :]  # [CLS]
            emb = torch.nn.functional.normalize(out, p=2, dim=1).numpy()
            for (i, _), e in zip(todo[s:s + B], emb):
                cache[keys[i]] = e.astype(np.float32)
            if s % (B * 20) == 0 or s + B >= len(todo):
                _save_cache(cache)  # 周期性落盘，进程中断不丢进度
                print(f"[text_encoder] 编码进度 {len(cache)} 条", flush=True)
    return np.stack([cache[k] for k in keys])


def encode_texts(texts: list[str], tfidf_vectorizer=None) -> np.ndarray:
    """统一入口：specter2 -> [n,768]；tfidf -> [n,512]（须传已拟合 vectorizer）。"""
    if BACKEND == "specter2":
        return _specter_encode(texts)
    assert tfidf_vectorizer is not None, "tfidf 后端须传 vectorizer"
    return tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
