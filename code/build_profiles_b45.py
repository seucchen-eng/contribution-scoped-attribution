# -*- coding: utf-8 -*-
"""
评审响应（评审3第5条）：B4/B5 的语料级参与画像构建。
与 build_b6_role_profiles 同口径：全语料前向 -> (oid,eid) 参与概率 noisy-or 聚合。
B4/B5 输出为角色盲标量，画像 Z_M(oid,eid) ∈ [0,1]。
用法: python pipeline/build_profiles_b45.py [--models b4 b5] [--seeds 42 0 1 2 3 4]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import text_encoder as TE
import run_taskb as T
from run_b4_b5 import RoleGNN, MILAttribution

BASE = T.BASE


def build_profile(model_kind, ckpt_name, cache_name):
    cache = BASE / cache_name
    if cache.exists():
        with cache.open("rb") as f:
            prof = pickle.load(f)
        print(f"{cache_name} 已加载缓存：{len(prof)} 对")
        return prof
    arts = pd.read_pickle(BASE / "articles.pkl")
    aus = pd.read_pickle(BASE / "authorships.pkl")
    aus_by_doi = {d: g.sort_values("position") for d, g in aus.groupby("doi")}
    a_dim, e_dim = 16, TE.TEXT_DIM + 10
    model = RoleGNN(a_dim, e_dim) if model_kind == "b4" else MILAttribution(a_dim, e_dim)
    model.load_state_dict(torch.load(BASE / ckpt_name, map_location="cpu"))
    model.eval()
    acc = defaultdict(float)          # 累乘 1-p
    one_minus = defaultdict(lambda: 1.0)
    n_papers = 0
    with torch.no_grad():
        for row in arts.itertuples():
            g = aus_by_doi.get(row.doi)
            if g is None or row.doi not in T.corpus_mentions():
                continue
            A, positions, E, eids, oids = T.corpus_paper_arrays(row.doi, row.methods_text, g)
            if not eids:
                continue
            n_papers += 1
            logits = (model(torch.tensor(A), torch.tensor(E)) if model_kind == "b4"
                      else model.instance_logits(torch.tensor(A), torch.tensor(E)))
            p = torch.sigmoid(logits).numpy()
            for i, oid in enumerate(oids):
                if oid is None:
                    continue
                for j, eid in enumerate(eids):
                    one_minus[(oid, eid)] *= (1 - float(p[i, j]))
    prof = {k: 1 - v for k, v in one_minus.items()}
    with cache.open("wb") as f:
        pickle.dump(prof, f)
    print(f"{cache_name} 构建完成：{n_papers} 篇 | {len(prof)} 对")
    return prof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["b4", "b5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3, 4])
    args = ap.parse_args()
    for mk in args.models:
        for s in args.seeds:
            ckpt = f"ckpt_{mk}.pt" if s == 42 else f"ckpt_{mk}_s{s}.pt"
            cache = f"{mk}_profiles_main.pkl" if s == 42 else f"{mk}_profiles_s{s}.pkl"
            build_profile(mk, ckpt, cache)
