# -*- coding: utf-8 -*-
"""
M5 任务B v3 - 配对置换检验（Holm 校正）
对主集（K内全部）逐查询计算各模型指标，比较 B6 vs B6-rb、B6 vs B2、B6 vs B1：
符号翻转置换 10000 次，Holm 校正；另给 per-query 均值差与 95%  bootstrap CI。
输出：data/processed/taskb_perq_v2.npz（逐查询指标，供多 seed 复用）+ 打印检验表
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_taskb as T
import torch
from run_b4_b5 import RoleGNN, MILAttribution

BASE = T.BASE
METRICS = ("NDCG@10", "Recall@10", "MAP", "MRR")
N_PERM = 10000
SEED = 42


def per_query_metrics(queries, score_fn, seed=SEED):
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for q in queries:
        gold = list(dict.fromkeys(q["gold"]))
        gset = set(gold)
        negs = [x for x in dict.fromkeys(q["neg_hard"] + q["neg_easy"]) if x not in gset]
        cands = gold + negs
        keys = rng.random(len(cands))
        scores = np.array([score_fn(c, q) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], keys[i]))
        rel = np.array([1.0 if cands[i] in gset else 0.0 for i in order])
        ng = len(gset)
        k = 10
        dcg = float(np.sum(rel[:k] / np.log2(np.arange(2, k + 2))))
        ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(ng, k) + 2))))
        out["NDCG@10"].append(dcg / ideal)
        out["Recall@10"].append(float(rel[:k].sum()) / ng)
        hits, ap, mrr = 0, 0.0, 0.0
        for rank, r in enumerate(rel, 1):
            if r > 0:
                hits += 1
                ap += hits / rank
                if mrr == 0.0:
                    mrr = 1.0 / rank
        out["MAP"].append(ap / ng)
        out["MRR"].append(mrr)
    return {m: np.array(v) for m, v in out.items()}


def perm_test(d, rng, n=N_PERM):
    """配对符号翻转置换检验，双侧。返回 (观察均值差, p)。"""
    obs = float(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return obs, float(p)


def main():
    rng = np.random.default_rng(SEED)
    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    author_feat, ent_feat, pl, arts, aus = T.build_qt_features()
    prof6 = T.build_b6_role_profiles()
    hist6 = T.role_hist_by_author(aus)

    work = T.F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        import pandas as pd
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    c_mean = float(np.mean(list(c_by_oid.values())))

    scorers = {
        "B1": lambda c, q: 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0,
        "B2": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean),
        "B6-rb": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)),
        "B6role": T.make_b6_profile_scorer(prof6, hist6, pl),
        "B6full": T.make_b6_profile_scorer(prof6, hist6, pl, c_by_oid, c_mean),
    }
    perq = {}
    for name, fn in scorers.items():
        perq[name] = per_query_metrics(inK, fn)
        print(f"{name} 逐查询指标完成", flush=True)
    np.savez(BASE / "taskb_perq_v3.npz",
             **{f"{name}|{m}": v for name, d in perq.items() for m, v in d.items()})

    comps = [("B6full", "B6-rb"), ("B6full", "B2"), ("B6full", "B1"),
             ("B6full", "B6role"), ("B6role", "B6-rb")]
    rows = []
    for a, b in comps:
        for m in METRICS:
            d = perq[a][m] - perq[b][m]
            obs, p = perm_test(d, rng)
            boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append([f"{a} vs {b}", m, obs, lo, hi, p])
    # Holm 校正（同一比较族内 4 指标 + 跨比较共 12 检验）
    order = sorted(range(len(rows)), key=lambda i: rows[i][5])
    adj = [0.0] * len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, rows[i][5] * (len(rows) - rank)))
        adj[i] = running
    print(f"\n{'比较':<12}{'指标':<10}{'均值差':>8}{'95%CI':>18}{'p':>8}{'p_Holm':>8}")
    for i, (cmp_, m, obs, lo, hi, p) in enumerate(rows):
        print(f"{cmp_:<12}{m:<10}{obs:>8.4f}   [{lo:>6.4f},{hi:>6.4f}]{p:>8.4f}{adj[i]:>8.4f}")


if __name__ == "__main__":
    main()
