# -*- coding: utf-8 -*-
"""7.2 锚点噪声敏感性：对训练注入锚点（金标训练部分）按 ρ 对称翻转标签，重训 B6 看任务A衰减。
用法：python pipeline/run_noise.py --rho 0.1 --phase train|taska
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as A

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"


def make_noisy(rho, seed=42):
    orig = A.cells_to_supervision
    def noisy(task_ids):
        sup = orig(task_ids)
        rng = np.random.default_rng(seed)
        return {tid: [(p, e, 1 - z if rng.random() < rho else z) for p, e, z in cells]
                for tid, cells in sup.items()}
    return noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--phase", required=True, choices=["train", "taska"])
    args = ap.parse_args()
    variant = f"noise{int(args.rho * 100)}"
    A.cells_to_supervision = make_noisy(args.rho)   # train_variant 在模块全局里查名
    if args.phase == "train":
        A.train_variant(variant)
    else:
        A.taska(variant)


if __name__ == "__main__":
    main()
