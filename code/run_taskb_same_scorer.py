# -*- coding: utf-8 -*-
"""
评审响应（评审3第5条）：同分函数消融 —— 固定历史信号 w、活跃度 c_a 与回退机制，
仅替换归因输出。B4/B5 的角色盲参与画像 Z_M 经同一混合式角色分布注入式 (9)：
    p6_M(r|a,k) = Z_M(a,k)·hist_a(r) + (1−Z_M(a,k))·uniform
（覆盖对；未覆盖对与 B6 相同回退：声明角色直方图 → 均匀）。
对照：B2、B7、B6-full（同一评价代码路径复算，应与表3一致）。
6 种子均值±std；主集配对置换检验 B6 vs B5same / B4same。
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_taskb as T
from run_taskb_permtest import per_query_metrics, perm_test

BASE = T.BASE
ROLE_SET = T.ROLE_SET if hasattr(T, "ROLE_SET") else None
SEEDS = [42, 0, 1, 2, 3, 4]


def load(name):
    with (BASE / name).open("rb") as f:
        return pickle.load(f)


def make_same_scorer(profM, hist, pl, c_by_oid, c_mean):
    """B4/B5 角色盲画像的同分函数注入（结构对齐 make_b6_profile_scorer）。"""
    unif = np.ones(len(ROLE_SET)) / len(ROLE_SET)
    cache = {}

    def p6(oid, eid):
        key = (oid, eid)
        if key in cache:
            return cache[key]
        h = hist.get(oid)
        hn = h / h.sum() if (h is not None and h.sum() > 0) else None
        z = profM.get(key)
        if z is not None:
            base = hn if hn is not None else unif
            p = z * base + (1 - z) * unif
        else:
            p = hn if hn is not None else unif
        cache[key] = p
        return p

    def fn(c, q):
        w = float(pl.get((c, q["entity_id"]), 0.0))
        if w <= 0:
            return 0.0
        ri = ROLE_SET.index(q["role"]) if q["role"] in ROLE_SET else None
        p = p6(c, q["entity_id"])
        return w * (float(p[ri]) if ri is not None else float(p.max())) * c_by_oid.get(c, c_mean)
    return fn


def main():
    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    cold = [q for q in queries if q["entity_id"].startswith("E1")]
    print(f"主集 {len(inK)} | 冷启动 {len(cold)}")

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

    # 参照：B2 / B7 / B6(seed42) 复算
    ref = {}
    ref["B2"] = per_query_metrics(inK, lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean))
    ref["B7"] = per_query_metrics(inK, T.make_b6_profile_scorer({}, hist, pl, c_by_oid, c_mean))
    prof6 = T.build_b6_role_profiles()
    ref["B6full_s42"] = per_query_metrics(inK, T.make_b6_profile_scorer(prof6, hist, pl, c_by_oid, c_mean))
    for k, v in ref.items():
        print(f"[参照] {k:<12} NDCG@10={v['NDCG@10'].mean():.4f} MAP={v['MAP'].mean():.4f} MRR={v['MRR'].mean():.4f}")

    # 同分函数：B4/B5 各种子
    results = {}
    perq_store = {}
    for mk in ["b4", "b5"]:
        tag = f"{mk.upper()}same"
        per_seed = defaultdict(list)
        for s in SEEDS:
            name = f"{mk}_profiles_main.pkl" if s == 42 else f"{mk}_profiles_s{s}.pkl"
            profM = load(name)
            fn = make_same_scorer(profM, hist, pl, c_by_oid, c_mean)
            perq = per_query_metrics(inK, fn)
            for m, v in perq.items():
                per_seed[m].append(float(v.mean()))
            if s == 42:
                perq_store[tag] = perq
            perq_cold = per_query_metrics(cold, fn)
            per_seed["cold_NDCG@10"].append(float(perq_cold["NDCG@10"].mean()))
        results[tag] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_seed.items()}
        nd = results[tag]["NDCG@10"]
        cd = results[tag]["cold_NDCG@10"]
        print(f"[{tag}] 主集 NDCG@10={nd[0]:.4f}±{nd[1]:.4f} | 冷启动 NDCG@10={cd[0]:.4f}±{cd[1]:.4f}")

    # B6 6 种子参照（已有画像）
    per_seed6 = defaultdict(list)
    for s in SEEDS:
        pkl = "b6_profiles.pkl" if s == 42 else f"b6_profiles_s{s}.pkl"
        prof = T.build_b6_role_profiles(
            "ckpt_b6.pt" if s == 42 else f"ckpt_b6_s{s}.pt", pkl)
        fn = T.make_b6_profile_scorer(prof, hist, pl, c_by_oid, c_mean)
        perq = per_query_metrics(inK, fn)
        if s == 42:
            perq_store["B6full"] = perq
        per_seed6["NDCG@10"].append(float(perq["NDCG@10"].mean()))
    print(f"[B6full] 主集 NDCG@10={np.mean(per_seed6['NDCG@10']):.4f}±{np.std(per_seed6['NDCG@10']):.4f}")
    results["B6full"] = {"NDCG@10": (float(np.mean(per_seed6["NDCG@10"])), float(np.std(per_seed6["NDCG@10"])))}

    # 配对置换检验（seed42 逐查询）
    rng = np.random.default_rng(0)
    for tag in ["B4same", "B5same"]:
        d = perq_store["B6full"]["NDCG@10"] - perq_store[tag]["NDCG@10"]
        obs, p = perm_test(d, rng)
        print(f"B6 vs {tag}: ΔNDCG@10={obs:+.4f}, 置换检验 p={p:.4f}")

    (BASE / "taskb_same_scorer.json").write_text(json.dumps(
        {"ref": {k: {m: float(v.mean()) for m, v in x.items()} for k, x in ref.items()},
         "same_scorer": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已存 taskb_same_scorer.json")


if __name__ == "__main__":
    main()
