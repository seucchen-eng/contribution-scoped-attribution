# -*- coding: utf-8 -*-
"""
放量①：历史先验扩展到评测期实体（E1 新实体）
用法：python pipeline/expand_priors_post.py [--half-life-years 3]
逻辑：与 compute_priors.py 完全同口径（词边界 n-gram + 指数时间衰减，t* = 2024-01-01），
      仅实体表换成 entities_post.pkl（评测期新出现的 2887 个实体）。
      跨刊作者历史（285,663 篇）中对这些"本刊未见"方法的使用记录被计入先验——
      不涉任何金标标签信息，无泄漏。
产物：data/processed/priors_post.pkl   （E1 条目：openalex_id, entity_id, weight）
      data/processed/priors_union.pkl  （priors.pkl + priors_post.pkl 纵向合并）
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

    df_e = pd.read_pickle(OUT_DIR / "entities_post.pkl")
    df_h = pd.read_pickle(OUT_DIR / "author_history.pkl")

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
    print(f"E1 参与匹配的别名 {n_alias} 条（跳过通用词条 {skipped} 条）")

    lens = sorted(alias_by_len)
    weights = defaultdict(lambda: defaultdict(float))
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
    df_p.to_pickle(OUT_DIR / "priors_post.pkl")

    print(f"\n===== E1 先验计算结果 =====")
    print(f"命中次数: {n_hits}")
    print(f"有 E1 先验的作者: {df_p['openalex_id'].nunique()}")
    print(f"覆盖 E1 实体: {df_p['entity_id'].nunique()} / {df_e['entity_id'].nunique()}")
    print(f"先验记录: {len(df_p)} 条 | 耗时 {time.time()-t0:.0f}s")

    # 合并为 union（旧 priors.pkl 不动，便于复现对照）
    df_old = pd.read_pickle(OUT_DIR / "priors.pkl")
    df_u = pd.concat([df_old, df_p], ignore_index=True)
    assert not df_u.duplicated(subset=["openalex_id", "entity_id"]).any(), "E0/E1 实体空间应不相交"
    df_u.to_pickle(OUT_DIR / "priors_union.pkl")
    print(f"priors_union.pkl: {len(df_u)} 条（E0 {len(df_old)} + E1 {len(df_p)}）")


if __name__ == "__main__":
    main()
