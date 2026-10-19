# -*- coding: utf-8 -*-
"""
评审响应补充实验③：聚类 bootstrap 显著性（2026-10-19）
动机：评审指出任务B逐查询置换检验假设查询独立；同一评测论文/同一实体产生多条查询，
     有效样本量被高估。本脚本按簇（doi=评测论文；entity_id=实体）重采样做 bootstrap：
  对每次重采样，整簇抽取其全部查询，汇总均值差；B=10000 次得 95% CI 与双侧 p。
比较：B6full vs B2 / B6-rb / B7full / B1（指标 NDCG@10、MRR），集合 main/e1/full。
主集逐查询数组复用 taskb_perq_v3.npz；e1/full 的基线与 B7 逐查询数组在本脚本补齐并缓存。
Holm 校正在全部检验族上进行。
输出：data/processed/taskb_cluster_boot.json + data/processed/taskb_cluster_boot.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_taskb import role_hist_by_author, make_b6_profile_scorer
from run_taskb_permtest import per_query_metrics

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
METRICS = ("NDCG@10", "MRR")
B = 10000
SEED = 42


def cluster_boot(diff, clusters, rng, B=B):
    """整簇重采样 bootstrap。返回 (观察差, lo, hi, p_双侧)。"""
    idx = defaultdict(list)
    for i, c in enumerate(clusters):
        idx[c].append(i)
    cl = list(idx.values())
    sizes = np.array([len(x) for x in cl], dtype=np.float64)
    dsums = np.array([diff[x].sum() for x in cl], dtype=np.float64)
    S = rng.integers(0, len(cl), size=(B, len(cl)))
    stats = dsums[S].sum(axis=1) / sizes[S].sum(axis=1)
    obs = float(diff.mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2.0 * min(float((stats <= 0).mean()), float((stats >= 0).mean()))
    return obs, float(lo), float(hi), min(p, 1.0), len(cl)


def contribution_scalars(aus):
    work = F.METHOD_ROLES
    vals = defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        vals[a.openalex_id].append(len({x.get("role_canonical", "") for x in roles} & work) / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    return c_by_oid, float(np.mean(list(c_by_oid.values())))


def main():
    rng = np.random.default_rng(SEED)
    print("加载查询 ...", flush=True)
    queries = [json.loads(l) for l in (BASE / "queries_v3.jsonl").read_text(encoding="utf-8").splitlines()]
    sets = {
        "main": [q for q in queries if not q["entity_id"].startswith("E1")],
        "e1": [q for q in queries if q["entity_id"].startswith("E1")],
        "full": queries,
    }
    aus = pd.read_pickle(BASE / "authorships.pkl")
    pl = F.prior_lookup()
    hist = role_hist_by_author(aus)
    c_by_oid, c_mean = contribution_scalars(aus)

    base_scorers = {
        "B1": lambda c, q: 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0,
        "B2": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean),
        "B6-rb": lambda c, q: float(pl.get((c, q["entity_id"]), 0.0)),
        "B7full": make_b6_profile_scorer({}, hist, pl, c_by_oid, c_mean),
    }

    perq = {}
    z = np.load(BASE / "taskb_perq_v3.npz")
    for name in ("B1", "B2", "B6-rb", "B6role", "B6full"):
        perq[("main", name)] = {m: z[f"{name}|{m}"] for m in ("NDCG@10", "MRR")}
    perq[("main", "B7full")] = per_query_metrics(sets["main"], base_scorers["B7full"])
    print("main B7full 逐查询完成", flush=True)

    for sname in ("e1", "full"):
        for mname in ("B6role", "B6full"):
            perq[(sname, mname)] = {
                m: np.load(BASE / f"taskb_perq_v3_s42_{sname}_{mname}_{m.replace('@', '_')}.npy")
                for m in METRICS}
        for mname in ("B1", "B2", "B6-rb", "B7full"):
            perq[(sname, mname)] = per_query_metrics(sets[sname], base_scorers[mname])
            print(f"{sname} {mname} 逐查询完成", flush=True)

    rows = []
    for sname, qs in sets.items():
        clusters_doi = [q["doi"] for q in qs]
        clusters_ent = [q["entity_id"] for q in qs]
        for b in ("B1", "B2", "B6-rb", "B7full"):
            for m in METRICS:
                d = perq[(sname, "B6full")][m] - perq[(sname, b)][m]
                for ctag, clusters in (("doi", clusters_doi), ("entity", clusters_ent)):
                    obs, lo, hi, p, nc = cluster_boot(np.asarray(d), clusters, rng)
                    rows.append([sname, f"B6full vs {b}", m, ctag, obs, lo, hi, p, nc])
                    print(f"{sname:<5} B6full vs {b:<7} {m:<8} [{ctag:<6}] "
                          f"diff={obs:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.4f} (簇数={nc})", flush=True)

    order = sorted(range(len(rows)), key=lambda i: rows[i][7])
    adj = [0.0] * len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, rows[i][7] * (len(rows) - rank)))
        adj[i] = running

    lines = [f"{'集合':<6}{'比较':<18}{'指标':<9}{'簇':<8}{'均值差':>9}{'95%CI':>20}{'p':>9}{'p_Holm':>9}{'簇数':>7}"]
    for i, (sname, cmp_, m, ctag, obs, lo, hi, p, nc) in enumerate(rows):
        lines.append(f"{sname:<6}{cmp_:<18}{m:<9}{ctag:<8}{obs:>+9.4f}   [{lo:+.4f},{hi:+.4f}]{p:>9.4f}{adj[i]:>9.4f}{nc:>7}")
    report = "\n".join(lines)
    print("\n" + report)
    (BASE / "taskb_cluster_boot.txt").write_text(report, encoding="utf-8")
    out = [{"set": r[0], "cmp": r[1], "metric": r[2], "cluster": r[3], "diff": r[4],
            "ci_lo": r[5], "ci_hi": r[6], "p": r[7], "p_holm": adj[i], "n_clusters": r[8]}
           for i, r in enumerate(rows)]
    (BASE / "taskb_cluster_boot.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
