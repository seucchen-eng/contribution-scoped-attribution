# -*- coding: utf-8 -*-
"""
放量③试点：SPECTER2 vs TF-IDF 对照（任务A 603格保留测试集 + 任务B K内主集）
用法（先预热编码缓存后）：
  ENTITY_ENCODER=specter2 python pipeline/pilot_specter.py
产物：taska/taskb 指标打印 + data/processed/b6_profiles_sp.pkl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import run_taskb as T
from evaluate import load_gold, eval_task_a
from run_b4_b5 import MILAttribution, score_all as score_all_b45
from run_b6 import AttributionLayer, score_all as score_all_b6

BASE = T.BASE
E_DIM = 778  # SPECTER2 768 + one-hot 10（试点脚本专用，勿在 tfidf 下运行）

_, _, test_p = F.train_val_test_papers()

print("========== 任务 A（603 格保留测试集） ==========")
m5 = MILAttribution(16, E_DIM)
m5.load_state_dict(torch.load(BASE / "ckpt_b5_sp.pt")); m5.eval()
s5 = score_all_b45(m5, "b5", test_p)
m6 = AttributionLayer(16, E_DIM)
m6.load_state_dict(torch.load(BASE / "ckpt_b6_sp.pt")); m6.eval()
s6 = score_all_b6(m6, test_p)
for subset in ["all", "hard", "hardev"]:
    gold = load_gold("held_out_test", subset=subset)
    a5, a6 = eval_task_a(s5, gold), eval_task_a(s6, gold)
    print(f"[{subset:>6}] B5-sp AUPRC={a5['AUPRC']:.3f} F1={a5['F1@0.5']:.3f} ECE={a5['ECE']:.3f} | "
          f"B6-sp AUPRC={a6['AUPRC']:.3f} F1={a6['F1@0.5']:.3f} ECE={a6['ECE']:.3f}")
print("（TF-IDF 对照：B5 全层 0.980/硬证据 0.890；B6 全层 1.000/硬证据 1.000，ECE 0.015）")

print("\n========== 任务 B（K内主集 7,099 查询，B6 画像族） ==========")
prof = T.build_b6_role_profiles("ckpt_b6_sp.pt", "b6_profiles_sp.pkl")
aus = pd.read_pickle(BASE / "authorships.pkl")
hist = T.role_hist_by_author(aus)
pl = F.prior_lookup()
work = F.METHOD_ROLES
vals = defaultdict(list)
for a in aus.itertuples():
    if pd.isna(a.openalex_id):
        continue
    roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
    vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
c_mean = float(np.mean(list(c_by_oid.values())))

queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
inK = [q for q in queries if not q["entity_id"].startswith("E1")]
s_role = T.make_b6_profile_scorer(prof, hist, pl)
s_full = T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean)
for name, fn in [("B6-role-sp", s_role), ("B6-full-sp", s_full)]:
    r = T.eval_queries(inK, fn)
    print(f"{name}: NDCG@10={r['NDCG@10']:.4f} Recall@10={r['Recall@10']:.4f} "
          f"MAP={r['MAP']:.4f} MRR={r['MRR']:.4f}")
print("（TF-IDF 对照：B6-role 0.0340，B6-full 0.0400；最强基线 B2 0.037）")
