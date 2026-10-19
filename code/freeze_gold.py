# -*- coding: utf-8 -*-
"""
M3 收官：分歧裁决 + 金标冻结 + 训练/测试划分
裁决规则（用户 2026-10-13 确认）：
  - 是 vs 否（对立）→ 不确定
  - 其余分歧取两人中更保守的判定；保守排序为 不确定(2) > 否(0) > 是(1)（PU 立场：不轻易断言）
  → 实施效果：一切 z 分歧统一裁为 2；双方一致时取一致值
角色合并：双方均标 1 时取角色交集（为空则取并集）；证据等级取较高者
产出：
  data/processed/gold_standard.jsonl   金标（每格一行）
  data/processed/gold_split.json       8:2 训练注入/保留测试划分（按论文分层）
  data/processed/gold_report.txt       冻结报告
"""
import json
import random
from collections import Counter
from pathlib import Path

base = Path(r"D:\WritingPapers\WritingPapers\data\processed")
EV_RANK = {"direct": 3, "unique_holder": 2, "declared_match": 1, "weak": 0, None: -1}


def load(*names):
    recs = {}
    for name in names:
        p = base / name
        if not p.exists():
            continue
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            for it in r["items"]:
                recs[(r["task_id"], it["position"], it["entity_id"])] = (
                    it["z"], tuple(sorted(it["roles"])), it.get("ev"))
    return recs


A = load("annotations_A.jsonl", "annotations_A_formal.jsonl")
B = load("annotations_B.jsonl", "annotations_B_formal.jsonl")
assert set(A) == set(B), f"覆盖不一致：{len(set(A)^set(B))}"

# 第三裁决（48 条对立分歧 → 硬标签，2026-10-13 裁决员 C）
ADJ = {}
ap = base / "adjudications.jsonl"
if ap.exists():
    for l in ap.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        ADJ[(r["task_id"], r["position"], r["entity_id"])] = (r["adjudicated_z"], r["adjudicated_roles"])

gold, n_adj = [], 0
for k in sorted(set(A) | set(B)):
    za, ra, ea = A[k]
    zb, rb, eb = B[k]
    tid, pos, eid = k
    if za == zb:
        z = za
        if z == 1:
            inter = sorted(set(ra) & set(rb))
            roles = inter if inter else sorted(set(ra) | set(rb))
            ev = ea if EV_RANK[ea] >= EV_RANK[eb] else eb
        else:
            roles, ev = [], None
    elif k in ADJ:  # 对立分歧：第三裁决硬标签
        z, roles = ADJ[k]
        ev, n_adj = ("adjudicated" if z == 1 else None), n_adj + (0 if z != 2 else 1)
    else:
        z, roles, ev, n_adj = 2, [], None, n_adj + 1  # 其余分歧裁为不确定
    gold.append({"task_id": tid, "position": pos, "entity_id": eid,
                 "z": z, "roles": roles, "ev": ev,
                 "agree": za == zb})

with open(base / "gold_standard.jsonl", "w", encoding="utf-8") as f:
    for g in gold:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

# 深度裁决（第三轮，针对保留测试集的 z=2 格子，2026-10-13 方案 A）
# 只覆盖当前仍为 z=2 的格子；z=0 时 ev="adjudicated"，角色置空
DEEP = {}
dp = base / "adjudications_deep.jsonl"
if dp.exists():
    for l in dp.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        DEEP[(r["task_id"], r["position"], r["entity_id"])] = (r["z"], r.get("ev"))

n_deep = 0
for g in gold:
    k = (g["task_id"], g["position"], g["entity_id"])
    if g["z"] == 2 and k in DEEP:
        dz, dev = DEEP[k]
        g["z"] = dz
        g["ev"] = dev
        if dz != 1:
            g["roles"] = []
        g["agree"] = False
        n_deep += 1

with open(base / "gold_standard.jsonl", "w", encoding="utf-8") as f:
    for g in gold:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

# 按论文 8:2 分层划分（训练注入 / 保留测试）
by_paper = {}
for g in gold:
    by_paper.setdefault(g["task_id"], []).append(g)
papers = sorted(by_paper, key=lambda t: -sum(1 for g in by_paper[t] if g["z"] == 1))
random.seed(42)
test_papers = set(random.sample([t for t in papers], max(2, round(len(papers) * 0.2))))
split = {"train_inject": sorted(t for t in papers if t not in test_papers),
         "held_out_test": sorted(test_papers)}
(base / "gold_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")

zs = Counter(g["z"] for g in gold)
evs = Counter(g["ev"] for g in gold if g["z"] == 1)
test_cells = [g for g in gold if g["task_id"] in test_papers]
report = f"""===== 金标冻结报告（M3 收官）=====
论文数: {len(by_paper)}（试点 5 + 正式批 44）
判定单元总数: {len(gold)}
z 分布: 是 {zs[1]} / 否 {zs[0]} / 不确定 {zs[2]}
双人一致: {sum(1 for g in gold if g['agree'])} | 第三裁决硬标签: {len(ADJ)} | 裁为不确定: {n_adj} | 深度裁决覆盖: {n_deep}
z=1 证据等级: direct {evs.get('direct',0)} / unique_holder {evs.get('unique_holder',0)} / declared_match {evs.get('declared_match',0)} / weak {evs.get('weak',0)}
  → 硬证据样本（direct+unique_holder）: {evs.get('direct',0)+evs.get('unique_holder',0)} 条

划分（8:2，按论文分层）:
  训练注入: {len(split['train_inject'])} 篇
  保留测试: {len(split['held_out_test'])} 篇（{len(test_cells)} 格，其中 z=1: {sum(1 for g in test_cells if g['z']==1)}）
"""
print(report)
(base / "gold_report.txt").write_text(report, encoding="utf-8")
