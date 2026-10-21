# -*- coding: utf-8 -*-
"""
评审响应（第4c条）：核验查询子集的配对差值 + 按来源论文聚类的 bootstrap 区间。
对 115 / 40 两个子集，报 B6full 与 B7full、B1、Random 的 NDCG@10 配对差值及
95% 区间（query 级 iid bootstrap + 来源论文簇 bootstrap）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_taskb import role_hist_by_author, make_b6_profile_scorer
import run_taskb as T

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---- 重建核验查询（与 run_taskb_verified.py 同口径），并记录来源论文 ----
sp = json.loads((BASE / "gold_split.json").read_text(encoding="utf-8"))
hold = set(sp["held_out_test"])
rows = [json.loads(l) for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines()]
rows = [r for r in rows if r["task_id"] in hold]
aus = pd.read_pickle(BASE / "authorships.pkl")
amap = {}
for a in aus.itertuples():
    tid = a.doi.replace("10.1371/", "")
    amap[(tid, int(a.position))] = a.openalex_id if pd.notna(a.openalex_id) else None

role_gold = defaultdict(set)
ent_neg = defaultdict(set)
qsrc = defaultdict(set)          # (entity, role) -> {task_id}
for r in rows:
    oid = amap.get((r["task_id"], r["position"]))
    if not oid:
        continue
    e = r["entity_id"]
    if r["z"] == 1:
        for role in r["roles"]:
            role_gold[(e, role)].add(oid)
            qsrc[(e, role)].add(r["task_id"])
    elif r["z"] == 0:
        ent_neg[e].add(oid)

queries = []
for (e, role), gold in sorted(role_gold.items()):
    other = set()
    for (ee, rr), gg in role_gold.items():
        if ee == e and rr != role:
            other |= gg
    negs = (ent_neg.get(e, set()) | other) - gold
    if not gold or not negs:
        continue
    queries.append({"qid": f"V{e}_{role}", "entity_id": e, "role": role,
                    "gold": sorted(gold), "neg_hard": sorted(negs), "neg_easy": [],
                    "src": sorted(qsrc[(e, role)])})

hard = [q for q in queries if len(q["neg_hard"]) >= 5]
print(f"核验查询 {len(queries)}，困难子集 {len(hard)}；"
      f"跨论文查询 {sum(1 for q in queries if len(q['src'])>1)} 条")

# ---- 打分器（与 run_taskb_verified.py 相同）----
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
prof6 = T.build_b6_role_profiles()

scorers = {
    "B1": lambda c, q: 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0,
    "B2": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean),
    "B6full": make_b6_profile_scorer(prof6, hist, pl, c_by_oid, c_mean),
    "B7full": make_b6_profile_scorer({}, hist, pl, c_by_oid, c_mean),
}

# ---- 逐查询 NDCG@10（与 eval_queries 同口径，固定种子破并列）----
def per_query_ndcg(qs, score_fn, k=10, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for q in qs:
        gold = list(dict.fromkeys(q["gold"]))
        gset = set(gold)
        negs = [x for x in dict.fromkeys(q["neg_hard"] + q["neg_easy"]) if x not in gset]
        cands = gold + negs
        keys = rng.random(len(cands))
        scores = np.array([score_fn(c, q) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], keys[i]))
        rel = np.array([1.0 if cands[i] in gset else 0.0 for i in order])
        ng = len(gset)
        kk = min(k, len(rel))
        dcg = float(np.sum(rel[:kk] / np.log2(np.arange(2, kk + 2))))
        ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(ng, k) + 2))))
        out.append(dcg / ideal)
    return np.array(out)

def boot_ci(diff, clusters=None, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(diff), size=(n, len(diff)))
        bs = diff[idx].mean(axis=1)
    else:
        ucl = sorted(set(clusters))
        c2i = {c: [i for i, x in enumerate(clusters) if x == c] for c in ucl}
        draws = rng.integers(0, len(ucl), size=(n, len(ucl)))
        bs = np.empty(n)
        for b in range(n):
            ii = [i for c in draws[b] for i in c2i[ucl[c]]]
            bs[b] = diff[ii].mean()
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

res = {}
for tag, qs in (("all115", queries), ("hard40", hard)):
    pv = {name: per_query_ndcg(qs, fn) for name, fn in scorers.items()}
    # Random 参照：每次重抽并列键的期望水平，用 200 次随机序的逐查询均值
    rnd = np.mean([per_query_ndcg(qs, lambda c, q: 0.0, seed=s) for s in range(200)], axis=0)
    pv["Random"] = rnd
    clusters = [q["src"][0] for q in qs]   # 绝大多数查询单一来源论文
    res[tag] = {}
    for name in ["B1", "B2", "B6full", "B7full", "Random"]:
        res[tag][name] = float(pv[name].mean())
    for name in ["B7full", "B1", "Random"]:
        d = pv["B6full"] - pv[name]
        ci_q = boot_ci(d)
        ci_c = boot_ci(d, clusters=clusters)
        res[tag][f"B6-{name}"] = {"diff": float(d.mean()), "ci_query": ci_q, "ci_paper_cluster": ci_c}
        print(f"[{tag}] B6 vs {name:<7} diff={d.mean():+.4f}  "
              f"CI(query)=[{ci_q[0]:+.4f},{ci_q[1]:+.4f}]  "
              f"CI(paper)=[{ci_c[0]:+.4f},{ci_c[1]:+.4f}]")
    d = pv["B7full"] - pv["B1"]
    ci_q = boot_ci(d); ci_c = boot_ci(d, clusters=clusters)
    res[tag]["B7-B1"] = {"diff": float(d.mean()), "ci_query": ci_q, "ci_paper_cluster": ci_c}
    print(f"[{tag}] B7 vs B1     diff={d.mean():+.4f}  "
          f"CI(query)=[{ci_q[0]:+.4f},{ci_q[1]:+.4f}]  CI(paper)=[{ci_c[0]:+.4f},{ci_c[1]:+.4f}]")
    print(f"[{tag}] means: " + "  ".join(f"{n}={v:.4f}" for n, v in res[tag].items()
                                          if isinstance(v, float)))

(BASE / "taskb_verified_paired.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
print("已存 taskb_verified_paired.json")
