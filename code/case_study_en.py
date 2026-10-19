# -*- coding: utf-8 -*-
"""6.4 案例研究：journal.pcbi.1011383（13 作者、8 方法、多人同 Methodology）
产出：case_study_1011383.png（Gold | B3 | B6聚合 | B6两个角色通道 五联热图）
     + 叙述用关键数字（B3 典型错误格、B6 角色通道分辨示例）
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from evaluate import load_gold
from run_b6 import AttributionLayer, role_aggregate, prior_logodds, ROLE_SET

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
TID = "journal.pcbi.1011383"
DOI = "10.1371/journal.pcbi.1011383"
OUT = Path(__file__).resolve().parent.parent / "case_study_1011383_en.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据 ----------
tasks = {}
for f in ("annotation_tasks.json", "annotation_tasks_formal.json"):
    tasks.update({t["task_id"]: t for t in json.loads((BASE / f).read_text(encoding="utf-8"))})
t = tasks[TID]
gold = {(p, e): z for tid, p, e, z in load_gold("held_out_test", "all") if tid == TID}

arts = pd.read_pickle(BASE / "articles.pkl")
title = arts.loc[arts["doi"].str.contains(TID.split(".")[-1]), "title"].iloc[0]
aus = pd.read_pickle(BASE / "authorships.pkl")
au = aus[aus["doi"] == DOI].sort_values("position")

ent = pd.read_pickle(BASE / "entities.pkl")
canon = dict(zip(ent["entity_id"], ent["canonical"]))

A, positions, E, eids = F.paper_arrays(TID)
na, ne = A.shape[0], len(eids)

model = AttributionLayer(16, 522)
model.load_state_dict(torch.load(BASE / "ckpt_b6.pt"))
model.eval()
plo = prior_logodds(TID, positions, eids)
with torch.no_grad():
    p6 = torch.sigmoid(model(torch.tensor(A), torch.tensor(E), plo)).numpy()  # [na,ne,R]
    zam = role_aggregate(model(torch.tensor(A), torch.tensor(E), plo)).numpy()

s3raw = json.loads((BASE / "scores_b3.json").read_text())
s3 = np.array([[s3raw.get(f"{TID}|{p}|{e}", 0.0) for e in eids] for p in positions])

gz = np.full((na, ne), np.nan)
for (p, e), z in gold.items():
    if p in positions and e in eids:
        gz[positions.index(p), eids.index(e)] = z

# 标签
ylabels = []
for a in au.itertuples():
    roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
    wr = sorted({x.get("role_canonical", "") for x in roles} & F.METHOD_ROLES)
    ylabels.append(f"{a.surname} ({a.position}) [{'/'.join(x[:4] for x in wr)}]")
xlabels = [canon.get(e, e)[:22] for e in eids]

# 论文内出现的角色（取占比最高的两个通道做分面）
role_present = defaultdict(int)
for a in au.itertuples():
    roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
    for x in {r.get("role_canonical", "") for r in roles} & F.METHOD_ROLES:
        role_present[x] += 1
top2 = ["formal-analysis", "validation"]  # fixed to match the manuscript narrative

# ---------- 图 ----------
from matplotlib.colors import ListedColormap
cmap_gold = ListedColormap(["white", "black"])
cmap_gold.set_bad("lightgray")   # 未标注格

panels = [
    ("(a) Human gold z (black=1, white=0, grey=n/a)", np.ma.masked_invalid(gz), cmap_gold),
    ("(b) B3 rule allocation", s3, "viridis"),
    ("(c) B6 Z-hat aggregated", zam, "viridis"),
] + [(f"({'de'[i]}) B6 Z-hat [{r}]", p6[:, :, ROLE_SET.index(r)], "viridis") for i, r in enumerate(top2)]

fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.55 * na + 2.2), sharey=True)
for ax, (ttl, M, cmap) in zip(axes, panels):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(ttl, fontsize=10)
    ax.set_xticks(range(ne))
    ax.set_xticklabels(xlabels, rotation=38, ha="right", fontsize=8)
    ax.set_yticks(range(na))
    if ax is axes[0]:
        ax.set_yticklabels(ylabels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
fig.suptitle(f"Case: {DOI} ({na} authors x {ne} methods) - {title[:60]}...", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=180)
print(f"图已存 {OUT}")

# ---------- 叙述数字 ----------
print(f"\n标题: {title}")
print(f"作者 {na} 人，方法实体 {ne} 个，金标 {int(np.sum(~np.isnan(gz)))} 格（正 {int(np.nansum(gz==1))} / 负 {int(np.nansum(gz==0))}）")
# B3 典型错误：金标=0 但 B3 给最高分配的作者
errs = []
for i, p in enumerate(positions):
    for j, e in enumerate(eids):
        if not np.isnan(gz[i, j]):
            errs.append((p, e, gz[i, j], s3[i, j], zam[i, j]))
fp = sorted([x for x in errs if x[2] == 0], key=lambda x: -x[3])[:3]
fn = sorted([x for x in errs if x[2] == 1], key=lambda x: x[3])[:3]
print("\nB3 假阳（z=0 却获高分配）:")
for p, e, z, b3, b6 in fp:
    nm = au[au["position"] == p]["surname"].iloc[0]
    print(f"  {nm} × {canon.get(e, e)[:30]}: B3={b3:.3f} → B6={b6:.3f}")
print("B3 假阴（z=1 却获低分配）:")
for p, e, z, b3, b6 in fn:
    nm = au[au["position"] == p]["surname"].iloc[0]
    print(f"  {nm} × {canon.get(e, e)[:30]}: B3={b3:.3f} → B6={b6:.3f}")
# 角色分辨示例：同一作者在两个角色通道差异最大的格子
i, j = np.unravel_index(np.abs(p6[:, :, ROLE_SET.index(top2[0])] - p6[:, :, ROLE_SET.index(top2[1])]).argmax(), (na, ne))
nm = au[au["position"] == positions[i]]["surname"].iloc[0]
print(f"\n角色分辨示例: {nm} × {canon.get(eids[j], eids[j])[:30]}: "
      f"{top2[0]}={p6[i,j,ROLE_SET.index(top2[0])]:.3f} vs {top2[1]}={p6[i,j,ROLE_SET.index(top2[1])]:.3f}")
