# -*- coding: utf-8 -*-
"""
M4 共享特征层（B4/B5/B6 共用，对应公平性清单第 2 条：同等输入特征）
文本编码：默认 TF-IDF（512 维）；ENTITY_ENCODER=specter2 切换 SPECTER2（放量③对照实验用，
对照结论：任务 B 持平、任务 A 硬证据层受损，正式口径保留 TF-IDF，见初稿 5.6 节）
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 评审响应（评审3第6条）：TFIDF_TRAIN_ONLY=1 时仅用训练注入论文拟合 TF-IDF，
# 排除 10 篇留出测试论文的文本统计（传导式拟合对照实验）。
TFIDF_TRAIN_ONLY = os.environ.get("TFIDF_TRAIN_ONLY", "0") == "1"

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

CREDIT_14 = ["conceptualization", "data-curation", "formal-analysis", "funding-acquisition",
             "investigation", "methodology", "project-administration", "resources",
             "software", "supervision", "validation", "visualization",
             "writing-original-draft", "writing-review-editing"]
ROLE2I = {r: i for i, r in enumerate(CREDIT_14)}
METHOD_ROLES = {"methodology", "software", "formal-analysis", "investigation",
                "data-curation", "validation"}
ENTITY_TYPES = ["algorithm", "model", "statistical", "experiment", "software", "pipeline"]
CTX_ROLES = ["developed", "applied", "baseline", "reviewed"]

_cache = {}


def _load(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def tasks():
    def _l():
        t = json.loads((BASE / "annotation_tasks_formal.json").read_text(encoding="utf-8"))
        t += json.loads((BASE / "annotation_tasks.json").read_text(encoding="utf-8"))
        return {x["task_id"]: x for x in t}
    return _load("tasks", _l)


def split():
    return _load("split", lambda: json.loads((BASE / "gold_split.json").read_text(encoding="utf-8")))


def gold_cells():
    """(task_id, position, entity_id, z, roles, ev)，z∈{0,1}。"""
    def _l():
        out = []
        for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            g = json.loads(l)
            if g["z"] in (0, 1):
                out.append((g["task_id"], g["position"], g["entity_id"], g["z"],
                            tuple(g["roles"]), g["ev"]))
        return out
    return _load("gold", _l)


def tfidf():
    """在实体上下文 + 论文摘要上拟合 TF-IDF（512 维）。
    默认拟合全部 49 篇标注论文（无监督、不用标签）；TFIDF_TRAIN_ONLY=1 时
    仅拟合 39 篇训练注入论文（排除留出测试论文的文本统计）。"""
    def _l():
        if TFIDF_TRAIN_ONLY:
            keep = set(split()["train_inject"])
            src = {k: v for k, v in tasks().items() if k in keep}
        else:
            src = tasks()
        corpus = [e["context"] for t in src.values() for e in t["entities"]]
        corpus += [t["abstract"] for t in src.values()]
        v = TfidfVectorizer(max_features=512, ngram_range=(1, 2), sublinear_tf=True)
        v.fit(corpus)
        return v
    return _load("tfidf_trainonly" if TFIDF_TRAIN_ONLY else "tfidf", _l)


def author_features(task):
    """每篇论文的作者特征矩阵 [n_authors, 14+2]：角色 multi-hot + 归一化 position + 方法角色数。"""
    feats = []
    for a in task["authors"]:
        v = np.zeros(16, dtype=np.float32)
        for r in a["declared_roles"]:
            if r in ROLE2I:
                v[ROLE2I[r]] = 1.0
        v[14] = a["position"] / max(len(task["authors"]), 1)
        v[15] = len(set(a["declared_roles"]) & METHOD_ROLES) / 6.0
        feats.append(v)
    return np.stack(feats), [a["position"] for a in task["authors"]]


def entity_features(task):
    """每篇论文的实体特征矩阵 [text_dim+6+4]。
    文本部分由 text_encoder 统一编码：默认 TF-IDF（512 维，复现试生产），
    ENTITY_ENCODER=specter2 时为 SPECTER2 [CLS]（768 维）。"""
    import text_encoder as TE
    texts = [e["context"] for e in task["entities"]]
    T = TE.encode_texts(texts, tfidf_vectorizer=tfidf())
    feats, eids = [], []
    for t, e in zip(T, task["entities"]):
        oh_t = np.zeros(6, dtype=np.float32)
        if e["type"] in ENTITY_TYPES:
            oh_t[ENTITY_TYPES.index(e["type"])] = 1.0
        oh_c = np.zeros(4, dtype=np.float32)
        if e["role_ctx"] in CTX_ROLES:
            oh_c[CTX_ROLES.index(e["role_ctx"])] = 1.0
        feats.append(np.concatenate([t, oh_t, oh_c]))
        eids.append(e["entity_id"])
    return np.stack(feats), eids


def prior_lookup():
    """(openalex_id, entity_id) -> weight（π_ak 原料）。
    放量①后优先读 priors_union.pkl（E0+E1 合并）；无则回退 priors.pkl（仅 E0）。
    注意：_load 缓存键随文件名区分，避免同会话内新旧混用。"""
    def _l():
        f = BASE / "priors_union.pkl"
        if not f.exists():
            f = BASE / "priors.pkl"
        df = pd.read_pickle(f)
        return {(r.openalex_id, r.entity_id): r.weight for r in df.itertuples()}
    key = "priors_union" if (BASE / "priors_union.pkl").exists() else "priors"
    return _load(key, _l)


def author_oid(task_id, position):
    """(task_id, position) -> openalex_id。"""
    def _l():
        df_au = pd.read_pickle(BASE / "authorships.pkl")
        t2d = {tid: t["doi"] for tid, t in tasks().items()}
        m = {}
        for tid, doi in t2d.items():
            sub = df_au[df_au["doi"] == doi]
            for r in sub.itertuples():
                m[(tid, r.position)] = r.openalex_id if pd.notna(r.openalex_id) else None
        return m
    return _load("oid", _l).get((task_id, position))


def paper_arrays(task_id):
    """返回 (A_feat, positions, E_feat, entity_ids)。"""
    t = tasks()[task_id]
    A, positions = author_features(t)
    E, eids = entity_features(t)
    return A, positions, E, eids


def train_val_test_papers():
    """训练 39 篇内部再切 8 篇做早停验证。"""
    tr = split()["train_inject"]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(tr))
    val = [tr[i] for i in idx[:8]]
    train = [tr[i] for i in idx[8:]]
    return train, val, split()["held_out_test"]
