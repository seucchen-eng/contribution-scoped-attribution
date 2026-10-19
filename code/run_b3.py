# -*- coding: utf-8 -*-
"""
M4 基线 B3：HeteAlloc+LLM（初稿 5.3：HeteAlloc 路径相似度分配，方法实体替换为 4.2.1 的 LLM 实体）
运行：python pipeline/run_b3.py [--eps 1e-3]
机制（公平化改造，与 B6 同等输入）：
  - 作者历史强度 w_ak：与 B6 的 π 同源（priors.pkl 的衰减计数）；
  - 分配规则：score(a,p,k) = (w_ak + ε) / Σ_{a'∈A_p} (w_a'k + ε)，
    即论文内按路径相似度（作者→历史论文→实体）把方法知识"分配"给作者——
    无监督、固定规则、论文内归一化（与 B6 的"不归一化、多解共存"形成机制对照，
    对应初稿 4.3.2 设计说明与 4.3.5 第 2 条）；
  - 全体作者对该实体均无历史时（w 全 0），退化为均分 1/|A_p|。
  无训练过程，直接打分；与 B4/B5/B6 同一评测脚本、同一三层子集。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from evaluate import eval_task_a, eval_task_b, load_gold


def score_b3(eps):
    pl = F.prior_lookup()
    _, _, test_p = F.train_val_test_papers()
    s = {}
    for tid in test_p:
        A, positions, E, eids = F.paper_arrays(tid)
        W = np.zeros((len(positions), len(eids)), dtype=np.float64)
        for i, pos in enumerate(positions):
            oid = F.author_oid(tid, pos)
            if oid is None:
                continue
            for j, eid in enumerate(eids):
                w = pl.get((oid, eid))
                if w is not None:
                    W[i, j] = float(w)
        share = (W + eps) / (W + eps).sum(axis=0, keepdims=True)
        for i, pos in enumerate(positions):
            for j, eid in enumerate(eids):
                s[(tid, pos, eid)] = float(share[i, j])
    return s, test_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()

    s, test_p = score_b3(args.eps)

    print(f"===== B3 HeteAlloc+LLM（确定性规则，eps={args.eps}）=====")
    for subset in ["all", "hard", "hardev"]:
        gold = load_gold("held_out_test", subset=subset)
        ma, mb = eval_task_a(s, gold), eval_task_b(s, gold)
        print(f"[{subset:>6}] AUPRC={ma['AUPRC']:.3f} F1={ma['F1@0.5']:.3f} ECE={ma['ECE']:.3f} | "
              f"NDCG@3={mb['NDCG@3']:.3f} NDCG@5={mb['NDCG@5']:.3f} MRR={mb['MRR']:.3f}")

    gold_unc = load_gold("held_out_test", keep_uncertain=True)
    unc = [g for g in gold_unc if g[3] == 2]
    print(f"不确定格子平均预测概率: {np.mean([s[(g[0], g[1], g[2])] for g in unc if (g[0], g[1], g[2]) in s]):.3f}")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / "scores_b3.json"
    out.write_text(json.dumps({"|".join(map(str, k)): v for k, v in s.items()}), encoding="utf-8")
    print(f"分数已存 {out.name}")


if __name__ == "__main__":
    main()
