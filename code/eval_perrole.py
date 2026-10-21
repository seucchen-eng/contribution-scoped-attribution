# -*- coding: utf-8 -*-
"""
评审响应（第5条）：分角色归因评价。
模型输出逐角色概率（sigmoid，6 个工作类角色），金标 roles 列表给出
每个 z=1 单元的角色真值。构造逐角色四元组评价：
  正例 = (cell, r), r ∈ roles（z=1 单元）
  负例 = (cell, r'), r' ∉ roles（z=1 单元）+ (cell, r) 全部 6 角色（z=0 单元）
  z=2 剔除（与主评价一致）。
另报：正例单元的角色识别准确率（argmax 命中 / top-2 命中）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_b6 import AttributionLayer, prior_logodds, ROLE_SET
from evaluate import load_gold, _auprc

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

# 加载主模型（seed 42）
A0, _, E0, _ = F.paper_arrays(F.train_val_test_papers()[0][0])
model = AttributionLayer(A0.shape[1], E0.shape[1])
model.load_state_dict(torch.load(BASE / "ckpt_b6.pt", map_location="cpu"))
model.eval()

_, _, test_p = F.train_val_test_papers()

# 逐角色概率表 {(tid,pos,eid): np.array[6]}
probs = {}
with torch.no_grad():
    for tid in test_p:
        A, positions, E, eids = F.paper_arrays(tid)
        plo = prior_logodds(tid, positions, eids)
        p_role = torch.sigmoid(model(torch.tensor(A), torch.tensor(E), plo)).numpy()
        for i, pos in enumerate(positions):
            for j, eid in enumerate(eids):
                probs[(tid, pos, eid)] = p_role[i, j]

# 金标（含 roles）：held_out_test，剔除 z=2
rows = []
sp = json.loads((BASE / "gold_split.json").read_text(encoding="utf-8"))
hold = set(sp["held_out_test"])
for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    g = json.loads(l)
    if g["task_id"] in hold and g["z"] in (0, 1):
        rows.append(g)

R2I = {r: i for i, r in enumerate(ROLE_SET)}
print("ROLE_SET:", ROLE_SET)
print("cells (z∈{0,1}):", len(rows), " pos:", sum(r["z"] for r in rows))

# 逐角色四元组评价
y_all, s_all = [], []
per_role = defaultdict(lambda: ([], []))
hit1, hit2, n_pos = 0, 0, 0
for g in rows:
    key = (g["task_id"], g["position"], g["entity_id"])
    if key not in probs:
        continue
    p = probs[key]
    pos_roles = {r for r in g["roles"] if r in R2I} if g["z"] == 1 else set()
    for ri, r in enumerate(ROLE_SET):
        y = 1 if r in pos_roles else 0
        y_all.append(y); s_all.append(float(p[ri]))
        per_role[r][0].append(y); per_role[r][1].append(float(p[ri]))
    if g["z"] == 1 and pos_roles:
        n_pos += 1
        order = np.argsort(-p)
        if ROLE_SET[order[0]] in pos_roles:
            hit1 += 1
        if any(ROLE_SET[i] in pos_roles for i in order[:2]):
            hit2 += 1

print(f"\n=== 逐角色四元组（author×method×role）===")
print(f"tuples: {len(y_all)}  pos: {sum(y_all)}  micro-AUPRC: {_auprc(y_all, s_all):.4f}")
print(f"{'role':<18}{'pos':>6}{'AUPRC':>8}")
for r in ROLE_SET:
    yy, ss = per_role[r]
    a = _auprc(yy, ss) if sum(yy) else float('nan')
    print(f"{r:<18}{sum(yy):>6}{a:>8.4f}")
print(f"\n正例单元角色识别: argmax 命中 {hit1}/{n_pos} = {hit1/n_pos:.3f}; "
      f"top-2 命中 {hit2}/{n_pos} = {hit2/n_pos:.3f}")

# 角色聚合口径（与表2一致， noisy-or）复核
gold = load_gold("held_out_test")
zam = {k: 1 - float(np.prod(1 - v)) for k, v in probs.items()}
y = [g[3] for g in gold]
s = [zam.get((g[0], g[1], g[2]), 0.0) for g in gold]
print(f"\n聚合参与口径复核: n={len(y)} pos={sum(y)} AUPRC={_auprc(y, s):.6f}")

out = {
    "role_set": ROLE_SET,
    "tuple_level": {"n": len(y_all), "n_pos": int(sum(y_all)), "micro_AUPRC": _auprc(y_all, s_all)},
    "per_role": {r: {"n_pos": int(sum(per_role[r][0])),
                     "AUPRC": (_auprc(per_role[r][0], per_role[r][1]) if sum(per_role[r][0]) else None)}
                 for r in ROLE_SET},
    "positive_cell_role_accuracy": {"n_pos_cells": n_pos, "argmax_hit": hit1, "top2_hit": hit2},
    "aggregated_check": {"n": len(y), "n_pos": int(sum(y)), "AUPRC": _auprc(y, s)},
}
(BASE / "taska_perrole_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
print("\n已存 taska_perrole_results.json")
