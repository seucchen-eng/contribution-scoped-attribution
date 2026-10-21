# -*- coding: utf-8 -*-
"""
评审响应（评审3第6条）：TF-IDF 仅用训练文本拟合的对照评价。
环境：TFIDF_TRAIN_ONLY=1（features.py 内切换拟合语料）。
对 3 个种子（42/0/1）：构建 B6 画像 -> 任务 B 主集 NDCG@10（B6full 打分器）。
任务 A 数字取自 scores_b6_tro_seed*.json 的评测输出（训练日志）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_taskb as T
from run_taskb_permtest import per_query_metrics

BASE = T.BASE
SEEDS = [42, 0, 1]

queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
inK = [q for q in queries if not q["entity_id"].startswith("E1")]

aus = pd.read_pickle(BASE / "authorships.pkl")
hist = T.role_hist_by_author(aus)
pl = T.F.prior_lookup()
work = T.F.METHOD_ROLES
vals = defaultdict(list)
for a in aus.itertuples():
    if pd.isna(a.openalex_id):
        continue
    roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
    vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
c_mean = float(np.mean(list(c_by_oid.values())))

import features as F
assert F.TFIDF_TRAIN_ONLY, "需要 TFIDF_TRAIN_ONLY=1 运行"

res = {}
for s in SEEDS:
    ckpt = f"ckpt_b6_tro_s{s}.pt"
    cache = f"b6_profiles_tro_s{s}.pkl"
    prof = T.build_b6_role_profiles(ckpt, cache)
    fn = T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean)
    perq = per_query_metrics(inK, fn)
    res[s] = {m: float(v.mean()) for m, v in perq.items()}
    print(f"tro seed {s}: NDCG@10={res[s]['NDCG@10']:.4f} MAP={res[s]['MAP']:.4f} MRR={res[s]['MRR']:.4f}",
          flush=True)

nd = [res[s]["NDCG@10"] for s in SEEDS]
print(f"\n训练集TF-IDF B6full 主集 NDCG@10 = {np.mean(nd):.4f} ± {np.std(nd):.4f}（3 种子）")
print("对照（传导式拟合）：0.0399 ± 0.0001（6 种子）")
(BASE / "taskb_tfidf_trainonly.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("已存 taskb_tfidf_trainonly.json")
