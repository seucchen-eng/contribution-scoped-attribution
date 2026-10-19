# -*- coding: utf-8 -*-
"""
放量①补充统计：全量集（20,852）与 E1 冷启动子集（13,753）的多 seed + 配对置换检验
用法：python pipeline/run_taskb_full.py
口径与 run_taskb.py / run_taskb_permtest.py 完全一致（方案A多正例、符号翻转 10000 次、Holm 校正）。
主集（K内 7,099）多种子与置换检验沿用既有产物（taskb_perq_s*.npy / taskb_perq_v2.npz），
本脚本只补全量与 E1 两个新集合。
产物：taskb_perq_v3_s{0..4,42}_*.npy、taskb_full_report.txt（基于 queries_v3.jsonl）
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_taskb as T
from run_taskb_permtest import per_query_metrics, perm_test

BASE = T.BASE
METRICS = ("NDCG@10", "Recall@10", "MAP", "MRR")


def contribution_scalars(aus):
    work = T.F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    return c_by_oid, float(np.mean(list(c_by_oid.values())))


def main():
    rng = np.random.default_rng(42)
    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    sets = {
        "full": queries,
        "e1": [q for q in queries if q["entity_id"].startswith("E1")],
    }
    aus = pd.read_pickle(BASE / "authorships.pkl")
    pl = T.F.prior_lookup()
    hist = T.role_hist_by_author(aus)
    c_by_oid, c_mean = contribution_scalars(aus)

    # ---------- 基线逐查询指标（与 seed 无关，seed 42 乱序口径） ----------
    base_scorers = {
        "B1": lambda c, q: 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0,
        "B2": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean),
        "B6-rb": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)),
    }
    perq = {}
    for sname, qs in sets.items():
        for mname, fn in base_scorers.items():
            perq[(sname, mname)] = per_query_metrics(qs, fn)
            print(f"{sname} {mname} 完成", flush=True)

    # ---------- B6 多 seed（主 seed42 + s0..s4） ----------
    seed_tags = [("42", "ckpt_b6.pt", "b6_profiles.pkl")] + \
                [(str(s), f"ckpt_b6_s{s}.pt", f"b6_profiles_s{s}.pkl") for s in range(5)]
    for stag, ckpt, cache in seed_tags:
        prof = T.build_b6_role_profiles(ckpt, cache)
        scorers = {
            "B6role": T.make_b6_profile_scorer(prof, hist, pl),
            "B6full": T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean),
        }
        for sname, qs in sets.items():
            for mname, fn in scorers.items():
                d = per_query_metrics(qs, fn)
                for m, v in d.items():
                    np.save(BASE / f"taskb_perq_v3_s{stag}_{sname}_{mname}_{m.replace('@','_')}.npy", v)
                if stag == "42":
                    perq[(sname, mname)] = d
                print(f"seed {stag} {sname} {mname}: "
                      + " ".join(f"{m}={v.mean():.4f}" for m, v in d.items()), flush=True)

    # ---------- 多 seed 汇总 ----------
    lines = ["===== 全量 / E1 多 seed 汇总（mean±std over 6 runs） ====="]
    for sname in sets:
        for mname in ("B6role", "B6full"):
            for m in ("NDCG@10", "MRR"):
                vals = [np.load(BASE / f"taskb_perq_v3_s{st}_{sname}_{mname}_{m.replace('@','_')}.npy").mean()
                        for st, _, _ in seed_tags]
                lines.append(f"{sname:<5}{mname:<8}{m:<9}{np.mean(vals):.4f}±{np.std(vals):.4f}")

    # ---------- 配对置换检验（seed42 口径，Holm 校正） ----------
    rows = []
    for sname in sets:
        for a, b in [("B6full", "B6-rb"), ("B6full", "B2"), ("B6full", "B1"), ("B6full", "B6role")]:
            for m in METRICS:
                d = perq[(sname, a)][m] - perq[(sname, b)][m]
                obs, p = perm_test(d, rng)
                rows.append([sname, f"{a} vs {b}", m, obs, p])
    order = sorted(range(len(rows)), key=lambda i: rows[i][4])
    adj = [0.0] * len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, rows[i][4] * (len(rows) - rank)))
        adj[i] = running
    lines.append(f"\n{'集合':<6}{'比较':<16}{'指标':<10}{'均值差':>8}{'p':>8}{'p_Holm':>8}")
    for i, (sname, cmp_, m, obs, p) in enumerate(rows):
        lines.append(f"{sname:<6}{cmp_:<16}{m:<10}{obs:>8.4f}{p:>8.4f}{adj[i]:>8.4f}")

    report = "\n".join(lines)
    print(report)
    (BASE / "taskb_full_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
