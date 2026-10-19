# -*- coding: utf-8 -*-
"""
M4 模型 B4 / B5 实现与评测（PyTorch CPU）
  B4 角色特征 GNN：CRediT 角色作为作者节点特征 + 一跳传播 + MLP 头，无显式归属层
  B5 通用 MIL：与 B6 同输入的弱监督归属， noisy-or 包损失，但无贡献约束设计
    （未声明当负例、无历史先验正则、无温度退火——与 B6 的差异即消融项）
数据：训练注入 39 篇（金标锚点），C_apr 粗监督用全部 49 篇任务论文；测试 10 篇
运行：python pipeline/run_b45.py [--model b4|b5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_gold, eval_task_a, eval_task_b, BASE
from features import Feats, build, ROLE_IDX

torch.manual_seed(42)
np.random.seed(42)

D_E, D_H = 128, 64


def load_cells(split):
    """返回 [(tid, pos, eid, z, roles)]，roles 为 z=1 时的角色列表。"""
    keep = set(json.loads((BASE / "gold_split.json").read_text(encoding="utf-8"))[split])
    out = []
    for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        g = json.loads(l)
        if g["task_id"] in keep and g["z"] in (0, 1):
            out.append((g["task_id"], g["position"], g["entity_id"], g["z"], g["roles"]))
    return out


class B4(nn.Module):
    """角色特征 GNN：作者节点特征 = 角色 one-hot；一跳：作者聚合其论文实体均值；MLP 打分。"""

    def __init__(self):
        super().__init__()
        self.role_proj = nn.Linear(14, D_H)
        self.ent_proj = nn.Linear(D_E, D_H)
        self.head = nn.Sequential(nn.Linear(D_H * 3, D_H), nn.ReLU(), nn.Linear(D_H, 1))

    def forward(self, role_oh, ent_mean, ent):
        a = torch.tanh(self.role_proj(role_oh) + self.ent_proj(ent_mean))
        e = self.ent_proj(ent)
        return self.head(torch.cat([a, e, a * e], dim=-1)).squeeze(-1)


class B5(nn.Module):
    """通用 MIL 归属：逐格打分 sigmoid(w·[e; role; prior] + b)，noisy-or 包损失。"""

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D_E + 14 + 1, 1)

    def forward(self, ent, role_oh, prior):
        x = torch.cat([ent, role_oh, prior.unsqueeze(-1)], dim=-1)
        return self.w(x).squeeze(-1)


def train_and_eval(model_name, epochs=300, lr=5e-3):
    build()
    F = Feats()
    train = load_cells("train_inject")
    test = load_cells("held_out_test")

    def vec(tid, pos, eid):
        ent = torch.tensor(F.e(eid))
        roh = torch.tensor(F.role_oh.get((tid, pos), np.zeros(14, np.float32)))
        key = F.auth_key.get((tid, pos), "")
        pr = torch.tensor(np.tanh(F.prior_scalar(key, eid) / 10.0), dtype=torch.float32)
        return ent, roh, pr

    model = B4() if model_name == "b4" else B5()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    # 预计算训练格特征
    tr = [(vec(t, p, e), z, (t, p, e)) for t, p, e, z, _ in train]
    # C_apr 包：(tid, pos, role) -> 论文内实体列表（粗监督）
    bags = {}
    tasks = []
    for f in ("annotation_tasks.json", "annotation_tasks_formal.json"):
        tasks += json.loads((BASE / f).read_text(encoding="utf-8"))
    for t in tasks:
        ents = [e["entity_id"] for e in t["entities"]]
        for a in t["authors"]:
            for r in a["declared_roles"]:
                if r in ("methodology", "software", "formal-analysis"):
                    bags.setdefault((t["task_id"], a["position"]), ents)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        # 锚点 BCE（训练注入金标）
        loss = 0
        for (ent, roh, pr), z, _k in tr:
            if model_name == "b4":
                s = model(roh.unsqueeze(0), ent.unsqueeze(0), ent.unsqueeze(0)).squeeze()
            else:
                s = model(ent.unsqueeze(0), roh.unsqueeze(0), pr.unsqueeze(0)).squeeze()
            loss = loss + bce(s.view(1), torch.tensor([float(z)]))
        loss = loss / len(tr)
        # MIL 包损失（B5 特性：未声明当负例的对称处理版，负包也加压）
        bag_items = list(bags.items())
        bag_loss, nb = 0, 0
        for (tid, pos), ents in bag_items:
            scores = []
            for eid in ents:
                ent, roh, pr = vec(tid, pos, eid)
                if model_name == "b4":
                    s = model(roh.unsqueeze(0), ent.unsqueeze(0), ent.unsqueeze(0)).squeeze()
                else:
                    s = model(ent.unsqueeze(0), roh.unsqueeze(0), pr.unsqueeze(0)).squeeze()
                scores.append(torch.sigmoid(s))
            if scores:
                p_bag = 1 - torch.prod(1 - torch.stack(scores))
                bag_loss = bag_loss - torch.log(p_bag + 1e-9)
                nb += 1
        loss = loss + 0.5 * bag_loss / max(nb, 1)
        loss.backward()
        opt.step()

    # 评测
    model.eval()
    scores = {}
    with torch.no_grad():
        for tid, pos, eid, z, _ in test:
            ent, roh, pr = vec(tid, pos, eid)
            s = model(roh.unsqueeze(0), ent.unsqueeze(0), ent.unsqueeze(0)) if model_name == "b4" \
                else model(ent.unsqueeze(0), roh.unsqueeze(0), pr.unsqueeze(0))
            scores[(tid, pos, eid)] = float(torch.sigmoid(s.squeeze()))
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["b4", "b5"])
    args = ap.parse_args()
    name = {"b4": "B4 角色特征 GNN", "b5": "B5 通用 MIL"}[args.model]

    scores = train_and_eval(args.model)
    for subset in ["all", "hard", "hardev"]:
        gold = load_gold("held_out_test", subset=subset)
        s = {k: v for k, v in scores.items() if k in {(g[0], g[1], g[2]) for g in gold}}
        ma, mb = eval_task_a(s, gold), eval_task_b(s, gold)
        print(f"[{name}][{subset}] AUPRC={ma['AUPRC']:.3f} F1={ma['F1@0.5']:.3f} "
              f"ECE={ma['ECE']:.3f} | NDCG@3={mb['NDCG@3']:.3f} MRR={mb['MRR']:.3f}")
    out = BASE / f"scores_{args.model}.json"
    out.write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in scores.items()}),
                   encoding="utf-8")
    print(f"分数已存 {out}")
