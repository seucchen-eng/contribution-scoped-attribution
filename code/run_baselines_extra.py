# -*- coding: utf-8 -*-
"""
评审响应补充实验：任务B 两个新基线（2026-10-19）
  Random   随机排序下界：多次重复（默认 20 次）取均值±std。
           走快速路径：随机排序等价于对 gold/neg 相关向量做随机置换，
           与 eval_queries 的指标口径完全一致（NDCG@k 理想序截断、Recall@k、MAP、MRR）。
  B7-role  作者级角色直方图基线：s = w(a,k) × PH(r|a)
  B7-full  作者级角色直方图 + 活跃度：s = w(a,k) × PH(r|a) × c_a
  其中 PH(r|a) = 作者在语料中的声明工作角色直方图（B6 中 P6 的第一级回退），
  即用作者级直方图替代实体级归因层输出，检验归因层对推荐排序的独立贡献。
输出：data/processed/taskb_extra_baselines.json + 控制台表格
用法：python pipeline/run_baselines_extra.py [--reps 20]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_taskb import eval_queries, role_hist_by_author, make_b6_profile_scorer

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
KS_NDCG = (5, 10, 20)
KS_RECALL = (10, 20)


def metrics_from_rel(rel, ng):
    """与 run_taskb.eval_queries 完全一致的指标计算。rel: 已按排序排好的 0/1 相关向量。"""
    m = {}
    for k in KS_NDCG:
        kk = min(k, len(rel))
        dcg = float(np.sum(rel[:kk] / np.log2(np.arange(2, kk + 2))))
        ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(ng, k) + 2))))
        m[f"NDCG@{k}"] = dcg / ideal
    for k in KS_RECALL:
        m[f"Recall@{k}"] = float(rel[:k].sum()) / ng
    hits, ap, mrr = 0, 0.0, 0.0
    for rank, r in enumerate(rel, 1):
        if r > 0:
            hits += 1
            ap += hits / rank
            if mrr == 0.0:
                mrr = 1.0 / rank
    m["MAP"] = ap / ng
    m["MRR"] = mrr
    return m


def eval_random(queries, n_reps, seed0=1000):
    """随机排序基线的快速评测：等价于 eval_queries 中 score_fn 返回独立均匀随机数
    （分数几乎必然互不相同，确定性乱序破并列不生效，故直接随机置换相关向量）。"""
    shapes = []
    for q in queries:
        gset = set(q["gold"])
        ng = len(gset)
        nc = ng + len(set(x for x in q["neg_hard"] + q["neg_easy"] if x not in gset))
        shapes.append((ng, nc))
    per_rep = defaultdict(list)
    for rep in range(n_reps):
        rng = np.random.default_rng(seed0 + rep)
        agg = defaultdict(list)
        for ng, nc in shapes:
            rel = np.zeros(nc)
            rel[:ng] = 1.0
            rng.shuffle(rel)
            for m, v in metrics_from_rel(rel, ng).items():
                agg[m].append(v)
        for m, v in agg.items():
            per_rep[m].append(float(np.mean(v)))
        print(f"    Random rep {rep + 1}/{n_reps}: NDCG@10={per_rep['NDCG@10'][-1]:.4f}", flush=True)
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}
            for m, v in per_rep.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    print("加载查询与语料数据 ...", flush=True)
    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    aus = pd.read_pickle(BASE / "authorships.pkl")
    pl = F.prior_lookup()
    hist = role_hist_by_author(aus)
    print(f"查询 {len(queries)} 条 | 先验对 {len(pl)} | 角色直方图作者 {len(hist)}", flush=True)

    # B2/B7-full 的贡献标量 c_a（与 run_taskb.stage_eval 同口径）
    work = F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        nw = len({x.get("role_canonical", "") for x in roles} & work)
        vals[a.openalex_id].append(nw / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    c_mean = float(np.mean(list(c_by_oid.values()))) if c_by_oid else 0.3

    s_b7role = make_b6_profile_scorer({}, hist, pl)                     # w × PH
    s_b7full = make_b6_profile_scorer({}, hist, pl, c_by_oid, c_mean)   # w × PH × c_a

    subsets = {
        "main": [q for q in queries if not q["entity_id"].startswith("E1")],
        "e1": [q for q in queries if q["entity_id"].startswith("E1")],
        "full": queries,
    }

    out = {}
    for sname, qs in subsets.items():
        print(f"\n##### 子集 {sname}（{len(qs)} 条查询）#####", flush=True)
        out[f"{sname}|Random"] = eval_random(qs, args.reps)
        out[f"{sname}|B7-role(w×PH)"] = eval_queries(qs, s_b7role)
        print(f"    B7-role 完成 NDCG@10={out[f'{sname}|B7-role(w×PH)']['NDCG@10']:.4f}", flush=True)
        out[f"{sname}|B7-full(w×PH×c_a)"] = eval_queries(qs, s_b7full)
        print(f"    B7-full 完成 NDCG@10={out[f'{sname}|B7-full(w×PH×c_a)']['NDCG@10']:.4f}", flush=True)

    (BASE / "taskb_extra_baselines.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = f"\n{'子集|模型':<26}{'NDCG@5':>9}{'NDCG@10':>9}{'NDCG@20':>9}{'R@10':>8}{'R@20':>8}{'MAP':>8}{'MRR':>8}"
    print(hdr)
    for sname in subsets:
        for label in ("Random", "B7-role(w×PH)", "B7-full(w×PH×c_a)"):
            r = out[f"{sname}|{label}"]
            if label == "Random":
                print(f"{sname + '|' + label:<26}"
                      f"{r['NDCG@5']['mean']:>9.4f}{r['NDCG@10']['mean']:>9.4f}{r['NDCG@20']['mean']:>9.4f}"
                      f"{r['Recall@10']['mean']:>8.4f}{r['Recall@20']['mean']:>8.4f}"
                      f"{r['MAP']['mean']:>8.4f}{r['MRR']['mean']:>8.4f}  (NDCG@10 std={r['NDCG@10']['std']:.4f})")
            else:
                print(f"{sname + '|' + label:<26}"
                      f"{r['NDCG@5']:>9.4f}{r['NDCG@10']:>9.4f}{r['NDCG@20']:>9.4f}"
                      f"{r['Recall@10']:>8.4f}{r['Recall@20']:>8.4f}"
                      f"{r['MAP']:>8.4f}{r['MRR']:>8.4f}")
    print(f"\n结果已写入 {BASE / 'taskb_extra_baselines.json'}")


if __name__ == "__main__":
    main()
