# -*- coding: utf-8 -*-
"""
评审响应补充实验②：独立核验的细粒度推荐子集（2026-10-19）
动机：评审指出任务B主协议的金标/负例由弱先验构造、非独立核验。
本脚本用 10 篇保留测试论文的 648 格人工金标（含角色真值）构造 (实体, 角色) 查询：
  gold  = z=1 且 roles 命中该角色的作者（openalex_id）
  neg   = 同实体 z=0 作者 ∪ 同实体 z=1 但 roles 不含该角色的作者（均为人工核验）
  z=2（不确定）作者不进入候选。
模型：Random（200 次重复）、B1、B2、B6-rb、B6role、B6full、B7role、B7full。
指标与 run_taskb.eval_queries 完全一致；另报负例≥5 的稳健子集。
输出：data/processed/taskb_verified_results.json + data/processed/queries_verified.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_taskb import eval_queries, role_hist_by_author, make_b6_profile_scorer
from run_baselines_extra import eval_random

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_verified_queries():
    sp = json.loads((BASE / "gold_split.json").read_text(encoding="utf-8"))
    hold = set(sp["held_out_test"])
    rows = [json.loads(l) for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["task_id"] in hold]
    aus = pd.read_pickle(BASE / "authorships.pkl")
    amap = {}
    for a in aus.itertuples():
        tid = a.doi.replace("10.1371/", "")
        amap[(tid, int(a.position))] = a.openalex_id if pd.notna(a.openalex_id) else None

    role_gold = defaultdict(set)   # (entity, role) -> {oid}
    ent_neg = defaultdict(set)     # entity -> {oid}（z=0 核验负例）
    for r in rows:
        oid = amap.get((r["task_id"], r["position"]))
        if not oid:
            continue
        e = r["entity_id"]
        if r["z"] == 1:
            for role in r["roles"]:
                role_gold[(e, role)].add(oid)
        elif r["z"] == 0:
            ent_neg[e].add(oid)

    queries = []
    for (e, role), gold in sorted(role_gold.items()):
        other_role_pos = set()
        for (ee, rr), gg in role_gold.items():
            if ee == e and rr != role:
                other_role_pos |= gg
        negs = (ent_neg.get(e, set()) | other_role_pos) - gold
        if not gold or not negs:
            continue
        queries.append({
            "qid": f"V{e}_{role}", "entity_id": e, "role": role, "ctx": "verified",
            "gold": sorted(gold), "neg_hard": sorted(negs), "neg_easy": [],
        })
    return queries


def main():
    queries = build_verified_queries()
    with (BASE / "queries_verified.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n5 = [q for q in queries if len(q["neg_hard"]) >= 5]
    print(f"核验查询 {len(queries)} 条（其中负例≥5: {len(n5)} 条）", flush=True)

    aus = pd.read_pickle(BASE / "authorships.pkl")
    pl = F.prior_lookup()
    hist = role_hist_by_author(aus)
    work = F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    c_mean = float(np.mean(list(c_by_oid.values()))) if c_by_oid else 0.3

    import run_taskb as T
    prof6 = T.build_b6_role_profiles()  # 加载缓存的 seed42 B6 角色画像

    scorers = {
        "B1": lambda c, q: 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0,
        "B2": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean),
        "B6-rb": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)),
        "B6role": make_b6_profile_scorer(prof6, hist, pl),
        "B6full": make_b6_profile_scorer(prof6, hist, pl, c_by_oid, c_mean),
        "B7role": make_b6_profile_scorer({}, hist, pl),
        "B7full": make_b6_profile_scorer({}, hist, pl, c_by_oid, c_mean),
    }

    out = {}
    for tag, qs in (("all", queries), ("neg>=5", n5)):
        print(f"\n##### 核验子集 {tag}（{len(qs)} 条查询）#####", flush=True)
        out[f"{tag}|Random"] = eval_random(qs, n_reps=200)
        for name, fn in scorers.items():
            out[f"{tag}|{name}"] = eval_queries(qs, fn)

    (BASE / "taskb_verified_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = f"\n{'子集|模型':<22}{'NDCG@5':>9}{'NDCG@10':>9}{'R@10':>8}{'MAP':>8}{'MRR':>8}"
    print(hdr)
    for tag in ("all", "neg>=5"):
        for name in ["Random", "B1", "B2", "B6-rb", "B6role", "B6full", "B7role", "B7full"]:
            r = out[f"{tag}|{name}"]
            if name == "Random":
                print(f"{tag + '|' + name:<22}"
                      f"{r['NDCG@5']['mean']:>9.4f}{r['NDCG@10']['mean']:>9.4f}"
                      f"{r['Recall@10']['mean']:>8.4f}{r['MAP']['mean']:>8.4f}{r['MRR']['mean']:>8.4f}"
                      f"  (NDCG@10 std={r['NDCG@10']['std']:.4f})")
            else:
                print(f"{tag + '|' + name:<22}"
                      f"{r['NDCG@5']:>9.4f}{r['NDCG@10']:>9.4f}"
                      f"{r['Recall@10']:>8.4f}{r['MAP']:>8.4f}{r['MRR']:>8.4f}")


if __name__ == "__main__":
    main()
