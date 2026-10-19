# -*- coding: utf-8 -*-
"""
M5 任务B - 多 seed 稳健性（B6 消融阶梯）
用法：
  python pipeline/run_taskb_seed.py --seed 0 --phase train   # 重训 B6(seed) -> ckpt_b6_s{N}.pt
  python pipeline/run_taskb_seed.py --seed 0 --phase eval    # 画像(缓存 b6_profiles_s{N}.pkl) + 逐查询指标 -> taskb_perq_s{N}.npz
主结果（seed=42）不动：ckpt_b6.pt / b6_profiles.pkl 保持原样。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_taskb as T
from run_b6 import train_b6
from run_taskb_permtest import per_query_metrics

BASE = T.BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phase", required=True, choices=["train", "eval"])
    args = ap.parse_args()
    s = args.seed

    if args.phase == "train":
        train_b6(300, 1e-3, s, 1.0, 0.1)          # 存到固定 ckpt_b6.pt
        (BASE / "ckpt_b6.pt").rename(BASE / f"ckpt_b6_s{s}.pt")
        print(f"seed {s}: ckpt_b6_s{s}.pt 已存")
        return

    prof = T.build_b6_role_profiles(f"ckpt_b6_s{s}.pt", f"b6_profiles_s{s}.pkl")
    aus = pd.read_pickle(BASE / "authorships.pkl")
    hist = T.role_hist_by_author(aus)
    pl = T.F.prior_lookup()

    work = T.F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    c_mean = float(np.mean(list(c_by_oid.values())))

    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    scorers = {
        "B6role": T.make_b6_profile_scorer(prof, hist, pl),
        "B6full": T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean),
    }
    out = {}
    for name, fn in scorers.items():
        perq = per_query_metrics(inK, fn)
        out[name] = {m: float(v.mean()) for m, v in perq.items()}
        for m, v in perq.items():
            np.save(BASE / f"taskb_perq_v3_s{s}_{name}_{m.replace('@','_')}.npy", v)
        print(f"seed {s} {name}: " + " ".join(f"{m}={v:.4f}" for m, v in out[name].items()), flush=True)


if __name__ == "__main__":
    main()
