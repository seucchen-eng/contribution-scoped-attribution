# -*- coding: utf-8 -*-
"""
M4 B6：本文方法（贡献证据约束的归属学习层，初稿 4.3 节）
运行：python pipeline/run_b6.py [--epochs 300] [--lambda1 1.0] [--lambda2 0.1] [--seed 42]

试生产简化声明（与初稿式 (2)-(6) 的对应与差异，投稿前需补齐）：
  - 式 (2)：u_a 由作者特征共享投影得到（非自由嵌入表），与 B4/B5 保持同等输入特征；
  - 式 (2) 历史证据向量 h_hist 暂置零（TF-IDF 阶段无历史上下文编码），
    仅保留 w_r 的实体侧项与 b_r；SPECTER2 替换后补齐；
  - 式 (3)：π_ak 以 log-odds 注入，缺失时取 0.01（低先验兜底）；
  - 式 (5)：LLM 锚点集 D_llm 暂无置信度产物，本阶段只用人工锚点（训练集金标）BCE；
  - 式 (11)(12) 联合推荐损失 L_rec 在任务 B 查询集构建后再接入（见实施计划 M4 后段）。
评测口径与 B4/B5 完全一致：同一特征、同一划分、同一评价脚本、同一三层子集。
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
D = 64
ROLE_SET = sorted(F.METHOD_ROLES)   # 试生产裁决：R 限于工作类角色（见下）
ROLE2RI = {r: i for i, r in enumerate(ROLE_SET)}
N_ROLES = len(ROLE_SET)
PI_DEFAULT = 0.01
# 设计裁决（2026-10-13，试生产记录）：式 (2)-(6) 中 R 取 6 个工作类角色
# {methodology, software, formal-analysis, investigation, data-curation, validation}，
# 不含 funding-acquisition / writing 等非工作类角色。原因：对非工作类角色施加
# noisy-or 包约束会把“资助/写作”泄漏成方法参与证据（首训出现资深 PI 被以
# 1.000 置信度误判执行核心方法的假阳）；初稿 4.3 节 R=14 的表述在正式稿中
# 需相应修订为“工作类角色子集”。


class AttributionLayer(nn.Module):
    """式 (2)(3)：多证据打分的逐角色参与概率。"""

    def __init__(self, a_dim, e_dim):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, D)
        self.e_proj = nn.Linear(e_dim, D)
        self.W = nn.Parameter(torch.randn(N_ROLES, D, D) * 0.05)   # 角色专属相容矩阵
        self.w = nn.Parameter(torch.randn(N_ROLES, D) * 0.05)      # 角色专属实体侧权重（h_hist=0）
        self.b = nn.Parameter(torch.zeros(N_ROLES))                # 角色偏置

    def forward(self, A, E, prior_lo):
        """A [na,a_dim]，E [ne,e_dim]，prior_lo [na,ne] -> 逐角色 logit [na,ne,R]"""
        ua = torch.relu(self.a_proj(A))
        ue = torch.relu(self.e_proj(E))
        compat = torch.einsum("ad,rde,ne->anr", ua, self.W, ue)     # 作者—方法—角色相容性
        hist = torch.einsum("rd,ne->nr", self.w, ue)                # [ne,R]
        s = compat + hist.unsqueeze(0) + self.b
        return s + prior_lo.unsqueeze(-1)


def role_aggregate(role_logits):
    """逐角色概率经 noisy-or 聚合成作者—方法参与概率 Ẑ_apk。"""
    p = torch.sigmoid(role_logits)
    return 1 - torch.prod(1 - p, dim=-1)   # [na, ne]


def prior_logodds(tid, positions, eids):
    """式 (3) 的 log-odds 先验。试生产裁决：priors.pkl 存的是式 (1) 的未归一化
    衰减计数（分子），直接用钳位当概率会把 w>1 饱和成 1；先用 π=1-exp(-w)
    把计数压成 (0,1) 的“有历史”概率，正式归一化（含 δ 平滑）在 M5 与
    SPECTER2 替换一并敲定并回填初稿。"""
    pl = F.prior_lookup()
    lo = np.full((len(positions), len(eids)), np.log(PI_DEFAULT / (1 - PI_DEFAULT)), dtype=np.float32)
    for i, pos in enumerate(positions):
        oid = F.author_oid(tid, pos)
        if oid is None:
            continue
        for j, eid in enumerate(eids):
            w = pl.get((oid, eid))
            if w is not None:
                pi = 1.0 - np.exp(-max(float(w), 0.0))
                pi = min(max(pi, 1e-4), 0.99)
                lo[i, j] = np.log(pi / (1 - pi))
    return torch.tensor(lo)


def cells_to_supervision(task_ids):
    by_paper = {}
    for tid, pos, eid, z, roles, ev in F.gold_cells():
        if tid in task_ids:
            by_paper.setdefault(tid, []).append((pos, eid, z))
    return by_paper


def train_b6(epochs, lr, seed, lambda1, lambda2):
    torch.manual_seed(seed)
    train_p, val_p, _ = F.train_val_test_papers()
    sup = cells_to_supervision(set(train_p) | set(val_p))
    A0, _, E0, _ = F.paper_arrays(train_p[0])
    model = AttributionLayer(A0.shape[1], E0.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    best_val, best_state = 1e9, None

    for ep in range(epochs):
        model.train()
        for tid in train_p:
            t = F.tasks()[tid]
            A, positions, E, eids = F.paper_arrays(tid)
            A_t, E_t = torch.tensor(A), torch.tensor(E)
            plo = prior_logodds(tid, positions, eids)
            role_logits = model(A_t, E_t, plo)          # [na,ne,R]
            p_role = torch.sigmoid(role_logits)
            z_am = role_aggregate(role_logits)          # [na,ne]

            # 式 (4) L_coarse：C_apr=1 的 (a,r) 对全部方法 noisy-or（PU：C=0 不加损失）
            l_coarse, nc = 0.0, 0
            for i, a in enumerate(t["authors"]):
                for r in a["declared_roles"]:
                    ri = ROLE2RI.get(r)
                    if ri is None:
                        continue
                    bag = 1 - torch.prod(1 - p_role[i, :, ri])
                    l_coarse = l_coarse - torch.log(bag.clamp(1e-6))
                    nc += 1
            if nc:
                l_coarse = l_coarse / nc

            # 式 (5) L_anchor：人工锚点（训练金标）BCE
            l_anchor, na_ = 0.0, 0
            for pos, eid, z in sup.get(tid, []):
                i, j = positions.index(pos), eids.index(eid)
                l_anchor = l_anchor + bce(z_am[i, j].clamp(1e-6, 1 - 1e-6).view(1),
                                          torch.tensor([float(z)]))
                na_ += 1
            if na_:
                l_anchor = l_anchor / na_

            # 式 (6) L_hist：低先验对的 KL(Bern(π)‖Bern(Ẑ))，权重 (1-π)
            pi = torch.sigmoid(plo)                     # [na,ne]
            q = z_am.clamp(1e-6, 1 - 1e-6)
            kl = pi * torch.log(pi.clamp(1e-6) / q) + \
                 (1 - pi) * torch.log((1 - pi).clamp(1e-6) / (1 - q))
            l_hist = ((1 - pi) * kl).mean()

            loss = l_coarse + lambda1 * l_anchor + lambda2 * l_hist
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 早停（与 B4 同口径：验证集金标 BCE）
        if ep % 10 == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for tid in val_p:
                    if tid not in sup:
                        continue
                    A, positions, E, eids = F.paper_arrays(tid)
                    plo = prior_logodds(tid, positions, eids)
                    z_am = role_aggregate(model(torch.tensor(A), torch.tensor(E), plo))
                    for pos, eid, z in sup[tid]:
                        i, j = positions.index(pos), eids.index(eid)
                        vl += float(bce(z_am[i, j].clamp(1e-6, 1 - 1e-6).view(1),
                                        torch.tensor([float(z)])))
                        vn += 1
            if vn and vl / vn < best_val:
                best_val = vl / vn
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               Path(__file__).resolve().parent.parent / "data" / "processed" / "ckpt_b6.pt")
    return model


@torch.no_grad()
def score_all(model, task_ids):
    s = {}
    for tid in task_ids:
        A, positions, E, eids = F.paper_arrays(tid)
        plo = prior_logodds(tid, positions, eids)
        z_am = role_aggregate(model(torch.tensor(A), torch.tensor(E), plo)).numpy()
        for i, pos in enumerate(positions):
            for j, eid in enumerate(eids):
                s[(tid, pos, eid)] = float(z_am[i, j])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda1", type=float, default=1.0, help="锚点损失权重")
    ap.add_argument("--lambda2", type=float, default=0.1, help="先验正则权重")
    args = ap.parse_args()

    model = train_b6(args.epochs, args.lr, args.seed, args.lambda1, args.lambda2)
    _, _, test_p = F.train_val_test_papers()
    s = score_all(model, test_p)

    print(f"===== B6 本文方法（seed={args.seed}, λ1={args.lambda1}, λ2={args.lambda2}）=====")
    for subset in ["all", "hard", "hardev"]:
        gold = load_gold("held_out_test", subset=subset)
        ma, mb = eval_task_a(s, gold), eval_task_b(s, gold)
        print(f"[{subset:>6}] AUPRC={ma['AUPRC']:.3f} F1={ma['F1@0.5']:.3f} ECE={ma['ECE']:.3f} | "
              f"NDCG@3={mb['NDCG@3']:.3f} NDCG@5={mb['NDCG@5']:.3f} MRR={mb['MRR']:.3f}")

    gold_unc = load_gold("held_out_test", keep_uncertain=True)
    unc = [g for g in gold_unc if g[3] == 2]
    print(f"不确定格子平均预测概率: {np.mean([s[(g[0], g[1], g[2])] for g in unc if (g[0], g[1], g[2]) in s]):.3f}")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / f"scores_b6_seed{args.seed}.json"
    out.write_text(json.dumps({"|".join(map(str, k)): v for k, v in s.items()}), encoding="utf-8")
    print(f"分数已存 {out.name}")


if __name__ == "__main__":
    main()
