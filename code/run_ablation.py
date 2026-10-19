# -*- coding: utf-8 -*-
"""
M5 表4消融：B6 变体训练与双任务评测（试生产可运行的 4 个变体）
变体：
  no_anchor     去 L_anchor（λ1=0）
  no_coarse     去 L_coarse（贡献声明监督，基损失权重置 0）
  no_prior      去 π 与 L_hist（训练/打分/画像全程用默认先验；任务B侧仅消融归属层，
                推荐协议的共享基信号 w 不变——表4说明列须写明）
  role_collapse 角色维度合并为 1（任务A需重训；任务B由构造恒等于 w×c_a=B2，免跑）
  z_norm        违反约束一（多对多）：逐角色参与概率在作者维 softmax 归一化
                （Σ_a p=1，信用守恒/单人独占），z_am 取逐角色最大值
  neg_declared  违反约束二（PU）：缺失声明按负例加 BCE（对未声明的工作类角色
                施加 -log(1-bag)），而非只监督已声明对
用法：
  python pipeline/run_ablation.py --variant no_anchor --phase train   # -> ckpt_b6_abl_*.pt
  python pipeline/run_ablation.py --variant no_anchor --phase taska   # 任务A三层子集
  python pipeline/run_ablation.py --variant no_anchor --phase taskb   # 画像+B6full NDCG 等
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import run_b6 as B6mod
from evaluate import load_gold, eval_task_a
from run_b6 import AttributionLayer, role_aggregate, cells_to_supervision, PI_DEFAULT

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
LO_DEF = float(np.log(PI_DEFAULT / (1 - PI_DEFAULT)))


def plo_default(positions, eids):
    return torch.tensor(np.full((len(positions), len(eids)), LO_DEF, dtype=np.float32))


def get_plo(tid, positions, eids, use_prior):
    if use_prior:
        return B6mod.prior_logodds(tid, positions, eids)
    return plo_default(positions, eids)


def aggregate_variant(role_logits, alloc):
    """返回 (逐角色参与概率 p_use, 聚合参与概率 z_am)。
    alloc=True 为 z_norm 变体：作者维 softmax 归一化（违反约束一）。"""
    if alloc:
        p_use = torch.softmax(role_logits, dim=0)      # [na,ne,R]，Σ_a = 1 per (k,r)
        z_am = p_use.max(dim=-1).values                # [na,ne]
    else:
        p_use = torch.sigmoid(role_logits)
        z_am = role_aggregate(role_logits)
    return p_use, z_am


def train_variant(variant, epochs=300, lr=1e-3, seed=42):
    w_coarse = 0.0 if variant == "no_coarse" else 1.0
    lambda1 = 0.0 if variant == "no_anchor" else 1.0
    lambda2 = 0.0 if variant == "no_prior" else 0.1
    use_prior = variant != "no_prior"
    alloc = variant == "z_norm"
    neg_decl = variant == "neg_declared"
    collapse = variant == "role_collapse"

    if collapse:  # 单角色：所有工作类角色映射到通道 0
        B6mod.N_ROLES = 1
        role2ri = {r: 0 for r in F.METHOD_ROLES}
    else:
        role2ri = B6mod.ROLE2RI

    torch.manual_seed(seed)
    train_p, val_p, _ = F.train_val_test_papers()
    sup = cells_to_supervision(set(train_p) | set(val_p))
    A0, _, E0, _ = F.paper_arrays(train_p[0])
    model = AttributionLayer(A0.shape[1], E0.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    best_val, best_state = 1e9, None

    for ep in range(epochs):
        model.train()
        for tid in train_p:
            t = F.tasks()[tid]
            A, positions, E, eids = F.paper_arrays(tid)
            A_t, E_t = torch.tensor(A), torch.tensor(E)
            plo = get_plo(tid, positions, eids, use_prior)
            role_logits = model(A_t, E_t, plo)
            p_use, z_am = aggregate_variant(role_logits, alloc)

            l_coarse, nc = 0.0, 0
            for i, a in enumerate(t["authors"]):
                declared = set(a["declared_roles"])
                for r in declared:
                    ri = role2ri.get(r)
                    if ri is None:
                        continue
                    bag = 1 - torch.prod(1 - p_use[i, :, ri])
                    l_coarse = l_coarse - torch.log(bag.clamp(1e-6))
                    nc += 1
                if neg_decl:  # 违反约束二：缺失声明按负例
                    for r in F.METHOD_ROLES - declared:
                        ri = role2ri.get(r)
                        if ri is None:
                            continue
                        bag = 1 - torch.prod(1 - p_use[i, :, ri])
                        l_coarse = l_coarse - torch.log((1 - bag).clamp(1e-6))
                        nc += 1
            if nc:
                l_coarse = l_coarse / nc

            l_anchor, na_ = 0.0, 0
            for pos, eid, z in sup.get(tid, []):
                i, j = positions.index(pos), eids.index(eid)
                l_anchor = l_anchor + bce(z_am[i, j].clamp(1e-6, 1 - 1e-6).view(1),
                                          torch.tensor([float(z)]))
                na_ += 1
            if na_:
                l_anchor = l_anchor / na_

            pi = torch.sigmoid(plo)
            q = z_am.clamp(1e-6, 1 - 1e-6)
            kl = pi * torch.log(pi.clamp(1e-6) / q) + \
                 (1 - pi) * torch.log((1 - pi).clamp(1e-6) / (1 - q))
            l_hist = ((1 - pi) * kl).mean()

            loss = w_coarse * l_coarse + lambda1 * l_anchor + lambda2 * l_hist
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 10 == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for tid in val_p:
                    if tid not in sup:
                        continue
                    A, positions, E, eids = F.paper_arrays(tid)
                    plo = get_plo(tid, positions, eids, use_prior)
                    _, z_am = aggregate_variant(model(torch.tensor(A), torch.tensor(E), plo), alloc)
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
    torch.save(model.state_dict(), BASE / f"ckpt_b6_abl_{variant}.pt")
    print(f"{variant}: ckpt_b6_abl_{variant}.pt 已存")


@torch.no_grad()
def taska(variant):
    use_prior = variant != "no_prior"
    alloc = variant == "z_norm"
    if variant == "role_collapse":
        B6mod.N_ROLES = 1
    model = AttributionLayer(16, 522)
    model.load_state_dict(torch.load(BASE / f"ckpt_b6_abl_{variant}.pt"))
    model.eval()
    _, _, test_p = F.train_val_test_papers()
    s = {}
    for tid in test_p:
        A, positions, E, eids = F.paper_arrays(tid)
        plo = get_plo(tid, positions, eids, use_prior)
        _, z_am = aggregate_variant(model(torch.tensor(A), torch.tensor(E), plo), alloc)
        z_am = z_am.numpy()
        for i, pos in enumerate(positions):
            for j, eid in enumerate(eids):
                s[(tid, pos, eid)] = float(z_am[i, j])
    print(f"===== 消融 {variant}：任务A =====")
    for subset in ["all", "hard", "hardev"]:
        gold = load_gold("held_out_test", subset=subset)
        m = eval_task_a(s, gold)
        print(f"[{subset:>6}] AUPRC={m['AUPRC']:.3f} P={m['P@0.5']:.3f} R={m['R@0.5']:.3f} "
              f"F1={m['F1@0.5']:.3f} ECE={m['ECE']:.3f}")


@torch.no_grad()
def taskb(variant):
    """消融模型画像 + B6full 打分（共享 w 与 c_a，仅 P6 来自消融画像）。"""
    import run_taskb as T
    use_prior = variant != "no_prior"
    alloc = variant == "z_norm"
    cache = BASE / f"b6_profiles_abl_{variant}.pkl"
    if cache.exists():
        import pickle
        prof = pickle.load(open(cache, "rb"))
    else:
        arts = pd.read_pickle(BASE / "articles.pkl")
        aus0 = pd.read_pickle(BASE / "authorships.pkl")
        aus_by_doi = {d: g.sort_values("position") for d, g in aus0.groupby("doi")}
        pl = F.prior_lookup()
        model = AttributionLayer(16, 522)
        model.load_state_dict(torch.load(BASE / f"ckpt_b6_abl_{variant}.pt"))
        model.eval()
        acc = defaultdict(lambda: np.ones(6))
        for row in arts.itertuples():
            g = aus_by_doi.get(row.doi)
            if g is None or row.doi not in T.corpus_mentions():
                continue
            A, positions, E, eids, oids = T.corpus_paper_arrays(row.doi, row.methods_text, g)
            if not eids:
                continue
            if use_prior:
                plo = T.prior_lo_from_oids(oids, eids, pl)
            else:
                plo = torch.tensor(np.full((len(oids), len(eids)), LO_DEF, dtype=np.float32))
            p_use, _ = aggregate_variant(model(torch.tensor(A), torch.tensor(E), plo), alloc)
            p6 = p_use.numpy()
            for i, oid in enumerate(oids):
                if oid is None:
                    continue
                for j, eid in enumerate(eids):
                    acc[(oid, eid)] = acc[(oid, eid)] * (1 - p6[i, j, :])
        prof = {k: 1 - v for k, v in acc.items()}
        import pickle
        pickle.dump(prof, open(cache, "wb"))
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
    queries = [json.loads(l) for l in (BASE / "queries_v2.jsonl").read_text(encoding="utf-8").splitlines()]
    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    fn = T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean)
    r = T.eval_queries(inK, fn)
    print(f"===== 消融 {variant}：任务B（K内主集 {len(inK)}）=====")
    print(" ".join(f"{m}={v:.4f}" for m, v in r.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["no_anchor", "no_coarse", "no_prior", "role_collapse",
                             "z_norm", "neg_declared"])
    ap.add_argument("--phase", required=True, choices=["train", "taska", "taskb"])
    args = ap.parse_args()
    if args.phase == "train":
        train_variant(args.variant)
    elif args.phase == "taska":
        taska(args.variant)
    else:
        taskb(args.variant)


if __name__ == "__main__":
    main()
