# -*- coding: utf-8 -*-
"""
M4 基线 B1/B2 实现与评测（打通链路用）
  B1 共著共享：所有作者同一分数（论文知识均等传播）
  B2 贡献标量加权：w_a = 1/position × (1 + 方法类声明角色数)（GraphRec 式贡献指数的可复现近似）
运行：python pipeline/run_b1_b2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_gold, eval_task_a, eval_task_b, permutation_test, BASE

METHOD_ROLES = {"methodology", "software", "formal-analysis", "investigation",
                "data-curation", "validation"}

gold = load_gold("held_out_test")
df_au = pd.read_pickle(BASE / "authorships.pkl")
tasks = json.loads((BASE / "annotation_tasks_formal.json").read_text(encoding="utf-8"))
tasks += json.loads((BASE / "annotation_tasks.json").read_text(encoding="utf-8"))
t2doi = {t["task_id"]: t["doi"] for t in tasks}

# 预计算每位作者的 B2 权重
w_cache = {}
def b2_weight(tid, pos):
    if (tid, pos) not in w_cache:
        doi = t2doi[tid]
        sub = df_au[(df_au["doi"] == doi) & (df_au["position"] == pos)]
        if sub.empty:
            w_cache[(tid, pos)] = 0.0
        else:
            roles = {x.get("role_canonical") for x in json.loads(sub.iloc[0]["roles"])}
            n_meth = len(roles & METHOD_ROLES)
            w_cache[(tid, pos)] = (1.0 / pos) * (1 + n_meth)
    return w_cache[(tid, pos)]

# 生成两种分数（B1 加随机微抖动破并列，5 种子平均在评测侧体现为期望值；
# 这里固定 seed 保证可复现）
rng = np.random.default_rng(42)
s_b1, s_b2 = {}, {}
for tid, pos, eid, z in gold:
    s_b1[(tid, pos, eid)] = 0.5 + rng.uniform(-0.01, 0.01)
    w = b2_weight(tid, pos)
    s_b2[(tid, pos, eid)] = 1 / (1 + np.exp(-(w - 0.5) * 4))  # 标量映射到 [0,1]

print("===== 任务 A：归属质量（保留测试集）=====")
for name, s in [("B1 共著共享", s_b1), ("B2 贡献标量加权", s_b2)]:
    m = eval_task_a(s, gold)
    print(f"{name}: AUPRC={m['AUPRC']:.3f} P@0.5={m['P@0.5']:.3f} "
          f"R@0.5={m['R@0.5']:.3f} F1={m['F1@0.5']:.3f} ECE={m['ECE']:.3f}")

print("\n===== 任务 B：任务驱动推荐（保留测试集）=====")
for name, s in [("B1 共著共享", s_b1), ("B2 贡献标量加权", s_b2)]:
    m = eval_task_b(s, gold)
    print(f"{name}: NDCG@3={m['NDCG@3']:.3f} NDCG@5={m['NDCG@5']:.3f} "
          f"Recall@3={m['Recall@3']:.3f} Recall@5={m['Recall@5']:.3f} MRR={m['MRR']:.3f}")

p = permutation_test(s_b2, s_b1, gold, metric="NDCG@5")
print(f"\nB2 vs B1 置换检验（NDCG@5，10000 次）: p = {p:.4f}")
