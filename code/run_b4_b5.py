# -*- coding: utf-8 -*-
"""
M4 基线 B4（角色特征 GNN）与 B5（通用 MIL 弱监督归属）
运行：python pipeline/run_b4_b5.py [--model b4|b5] [--epochs 300]
设计口径（初稿 5.3）：
  B4：CRediT 角色仅作为作者节点特征，作者—实体二部图消息传递，直接用训练集金标监督；
      无显式归属层 —— 检验“角色信息隐式使用”是否足够
  B5：通用多实例弱监督，仅以 C_apr 方法类声明为包标签（正包+负包），实例即方法实体，
      无锚点/先验/PU 约束 —— 检验“通用弱监督 vs 贡献证据约束”
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from evaluate import eval_task_a, eval_task_b, load_gold

DEV = "cpu"
D = 64  # 内部维度


# ---------------- B4：角色特征 GNN ----------------
class RoleGNN(nn.Module):
    def __init__(self, a_dim, e_dim):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, D)
        self.e_proj = nn.Linear(e_dim, D)
        self.msg_e2a = nn.Linear(D, D)
        self.msg_a2e = nn.Linear(D, D)
        self.out = nn.Sequential(nn.Linear(D * 3, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, A, E):
        """A: [na, a_dim], E: [ne, e_dim] -> [na, ne] 逐格 logit"""
        ha = torch.relu(self.a_proj(A))   # [na, D]
        he = torch.relu(self.e_proj(E))   # [ne, D]
        # 两轮二部消息传递
        m_a = torch.relu(self.msg_e2a(he)).mean(dim=0, keepdim=True)   # [1, D]
        ha = ha + m_a
        m_e = torch.relu(self.msg_a2e(ha)).mean(dim=0, keepdim=True)
        he = he + m_e
        na, ne = ha.size(0), he.size(0)
        ha_e = ha.unsqueeze(1).expand(na, ne, D)
        he_e = he.unsqueeze(0).expand(na, ne, D)
        x = torch.cat([ha_e, he_e, ha_e * he_e], dim=-1)
        return self.out(x).squeeze(-1)   # [na, ne]


# ---------------- B5：通用 MIL ----------------
class MILAttribution(nn.Module):
    def __init__(self, a_dim, e_dim):
        super().__init__()
        self.inst = nn.Sequential(nn.Linear(a_dim + e_dim, D), nn.ReLU(), nn.Linear(D, 1))

    def instance_logits(self, A, E):
        na, ne = A.size(0), E.size(0)
        x = torch.cat([A.unsqueeze(1).expand(na, ne, A.size(1)),
                       E.unsqueeze(0).expand(na, ne, E.size(1))], dim=-1)
        return self.inst(x).squeeze(-1)   # [na, ne]

    def bag_logit(self, A, E):
        """noisy-or 聚合实例概率 -> 包（作者）是否参与任一方法"""
        p = torch.sigmoid(self.instance_logits(A, E))   # [na, ne]
        bag = 1 - torch.prod(1 - p, dim=1)              # [na]
        return torch.logit(bag.clamp(1e-6, 1 - 1e-6))


def cells_to_supervision(task_ids):
    """训练监督：金标格子（z∈{0,1}）按论文分组。"""
    by_paper = {}
    for tid, pos, eid, z, roles, ev in F.gold_cells():
        if tid in task_ids:
            by_paper.setdefault(tid, []).append((pos, eid, z))
    return by_paper


def train_b4(epochs, lr, seed):
    torch.manual_seed(seed)
    train_p, val_p, _ = F.train_val_test_papers()
    sup = cells_to_supervision(set(train_p) | set(val_p))
    A0, _, E0, _ = F.paper_arrays(train_p[0])
    model = RoleGNN(A0.shape[1], E0.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for tid in train_p:
            if tid not in sup:
                continue
            A, positions, E, eids = F.paper_arrays(tid)
            logits = model(torch.tensor(A), torch.tensor(E))
            loss, n = 0.0, 0
            for pos, eid, z in sup[tid]:
                i, j = positions.index(pos), eids.index(eid)
                loss = loss + lossf(logits[i, j].view(1), torch.tensor([float(z)]))
                n += 1
            if n == 0:
                continue
            opt.zero_grad()
            (loss / n).backward()
            opt.step()
            tot += float(loss / n)
        # 早停
        if ep % 10 == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for tid in val_p:
                    if tid not in sup:
                        continue
                    A, positions, E, eids = F.paper_arrays(tid)
                    logits = model(torch.tensor(A), torch.tensor(E))
                    for pos, eid, z in sup[tid]:
                        i, j = positions.index(pos), eids.index(eid)
                        vl += float(lossf(logits[i, j].view(1), torch.tensor([float(z)])))
                        vn += 1
            if vn and vl / vn < best_val:
                best_val, best_state = vl / vn, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               Path(__file__).resolve().parent.parent / "data" / "processed" / "ckpt_b4.pt")
    return model


def train_b5(epochs, lr, seed):
    torch.manual_seed(seed)
    train_p, val_p, _ = F.train_val_test_papers()
    A0, _, E0, _ = F.paper_arrays(train_p[0])
    model = MILAttribution(A0.shape[1], E0.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for tid in train_p:
            t = F.tasks()[tid]
            A, positions, E, eids = F.paper_arrays(tid)
            A_t, E_t = torch.tensor(A), torch.tensor(E)
            # 包标签：作者在该论文是否声明了任一方法类角色（通用 MIL 惯例：正负包都用）
            y = [1.0 if set(a["declared_roles"]) & F.METHOD_ROLES else 0.0 for a in t["authors"]]
            bag = model.bag_logit(A_t, E_t)
            loss = lossf(bag, torch.tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(model.state_dict(),
               Path(__file__).resolve().parent.parent / "data" / "processed" / "ckpt_b5.pt")
    return model


@torch.no_grad()
def score_all(model, kind, task_ids):
    s = {}
    for tid in task_ids:
        A, positions, E, eids = F.paper_arrays(tid)
        logits = (model(torch.tensor(A), torch.tensor(E)) if kind == "b4"
                  else model.instance_logits(torch.tensor(A), torch.tensor(E)))
        p = torch.sigmoid(logits).numpy()
        for i, pos in enumerate(positions):
            for j, eid in enumerate(eids):
                s[(tid, pos, eid)] = float(p[i, j])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["b4", "b5"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = train_b4(args.epochs, args.lr, args.seed) if args.model == "b4" \
        else train_b5(args.epochs, args.lr, args.seed)
    _, _, test_p = F.train_val_test_papers()
    s = score_all(model, args.model, test_p)

    name = {"b4": "B4 角色特征 GNN", "b5": "B5 通用 MIL"}[args.model]
    print(f"===== {name}（seed={args.seed}）=====")
    for subset in ["all", "hard", "hardev"]:
        gold = load_gold("held_out_test", subset=subset)
        ma, mb = eval_task_a(s, gold), eval_task_b(s, gold)
        print(f"[{subset:>6}] AUPRC={ma['AUPRC']:.3f} F1={ma['F1@0.5']:.3f} ECE={ma['ECE']:.3f} | "
              f"NDCG@3={mb['NDCG@3']:.3f} NDCG@5={mb['NDCG@5']:.3f} MRR={mb['MRR']:.3f}")

    # 不确定格子校准
    gold_unc = load_gold("held_out_test", keep_uncertain=True)
    unc = [g for g in gold_unc if g[3] == 2]
    print(f"不确定格子平均预测概率: {np.mean([s[(g[0], g[1], g[2])] for g in unc if (g[0], g[1], g[2]) in s]):.3f}")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / f"scores_{args.model}_seed{args.seed}.json"
    out.write_text(json.dumps({"|".join(map(str, k)): v for k, v in s.items()}), encoding="utf-8")
    print(f"分数已存 {out.name}")


if __name__ == "__main__":
    main()
