# -*- coding: utf-8 -*-
"""
M2 证据提取 - 作者历史先验 π_ak 计算（对应初稿式 (1)）
用法：python pipeline/compute_priors.py [--half-life-years 3]
逻辑：
  - 实体别名在作者历史论文的 标题+摘要 中做词边界 n-gram 匹配
  - 时间衰减：weight = exp(-(t* - t_p') / λ_t)，t* = 2024-01-01
  - 过于通用的单词别名（simulation 等）不参与匹配，防止历史先验被噪声主导
产物：data/processed/priors.pkl  （长表：openalex_id, entity_id, weight）
"""
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

T_STAR = pd.Timestamp("2024-01-01")
# 过通用单词条目不参与历史匹配（避免噪声主导先验）
GENERIC_STOP = {"model", "models", "simulation", "simulations", "analysis", "method",
                "methods", "algorithm", "algorithms", "approach", "framework", "network",
                "networks", "regression", "clustering", "classification", "prediction",
                "optimization", "inference", "estimation", "sampling", "test", "testing"}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall((s or "").lower())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-life-years", type=float, default=3.0)
    args = ap.parse_args()
    lam_days = args.half_life_years * 365.0

    df_e = pd.read_pickle(OUT_DIR / "entities.pkl")
    df_h = pd.read_pickle(OUT_DIR / "author_history.pkl")

    # 别名 token 元组 -> entity_id 集合（按 n-gram 长度分组）
    alias_by_len = defaultdict(lambda: defaultdict(set))
    skipped = 0
    for _, r in df_e.iterrows():
        for alias in json.loads(r["aliases"]):
            toks = tuple(tokenize(alias))
            if not toks:
                continue
            if len(toks) == 1 and (toks[0] in GENERIC_STOP or len(toks[0]) < 4):
                skipped += 1
                continue
            alias_by_len[len(toks)][toks].add(r["entity_id"])
    n_alias = sum(len(v) for v in alias_by_len.values())
    print(f"参与匹配的别名 {n_alias} 条（跳过通用词条 {skipped} 条），长度分布: "
          f"{dict((k, len(v)) for k, v in sorted(alias_by_len.items()))}")

    lens = sorted(alias_by_len)
    weights = defaultdict(lambda: defaultdict(float))  # author -> entity -> w
    t0 = time.time()
    n_hits = 0
    for i, row in enumerate(df_h.itertuples(), 1):
        toks = tokenize(f"{row.title or ''} {row.abstract or ''}")
        if not toks:
            continue
        dt = (T_STAR - pd.Timestamp(row.pub_date)).days if row.pub_date else None
        if dt is None or dt < 0:
            continue
        w = math.exp(-dt / lam_days)
        for L in lens:
            if len(toks) < L:
                continue
            table = alias_by_len[L]
            for j in range(len(toks) - L + 1):
                gram = tuple(toks[j:j + L])
                if gram in table:
                    for eid in table[gram]:
                        weights[row.openalex_id][eid] += w
                        n_hits += 1
        if i % 40000 == 0:
            print(f"  进度 {i}/{len(df_h)}，命中 {n_hits}，耗时 {time.time()-t0:.0f}s")

    rows = [{"openalex_id": a, "entity_id": e, "weight": w}
            for a, m in weights.items() for e, w in m.items()]
    df_p = pd.DataFrame(rows)
    df_p.to_pickle(OUT_DIR / "priors.pkl")

    n_authors = df_p["openalex_id"].nunique()
    per_author = df_p.groupby("openalex_id")["entity_id"].count()
    print(f"\n===== π_ak 先验计算结果 =====")
    print(f"命中次数: {n_hits}")
    print(f"有非零先验的作者: {n_authors} / {df_h['openalex_id'].nunique()}")
    print(f"先验记录: {len(df_p)} 条 | 每作者实体数：中位数 {per_author.median():.0f}，"
          f"均值 {per_author.mean():.1f}，最大 {per_author.max()}")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
