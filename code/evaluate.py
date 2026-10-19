# -*- coding: utf-8 -*-
"""
M4 统一评测脚手架（所有模型共用，对应初稿 5.4 公平性清单第 1 条）
用法：
  from evaluate import load_gold, eval_task_a, eval_task_b, permutation_test
  scores = {(task_id, position, entity_id): 0.83, ...}   # 模型输出的 P(Z=1)
  ma = eval_task_a(scores, gold_test)
  mb = eval_task_b(scores, gold_test)
指标：
  任务 A（归属）: AUPRC / P@0.5 / R@0.5 / F1@0.5 / ECE
  任务 B（推荐）: 以每篇论文的每个方法实体为一次查询，正例 = 该实体 z=1 的作者；
                 候选 = 该论文全体作者；NDCG@5 / Recall@5 / MRR（论文内候选 5–17 人）
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"


HARD_PAPERS = None  # 难例论文集合（formal 批前 25 篇为难例筛选）


def _hard_set():
    global HARD_PAPERS
    if HARD_PAPERS is None:
        sel = json.loads((BASE / "formal_batch_dois.json").read_text(encoding="utf-8"))
        hard_dois = set(sel[:25])
        tasks = []
        for f in ("annotation_tasks.json", "annotation_tasks_formal.json"):
            if (BASE / f).exists():
                tasks += json.loads((BASE / f).read_text(encoding="utf-8"))
        HARD_PAPERS = {t["task_id"] for t in tasks if t["doi"] in hard_dois}
    return HARD_PAPERS


def load_gold(split: str = "held_out_test", subset: str = "all", keep_uncertain: bool = False):
    """返回 [(task_id, position, entity_id, z)]。
    subset: all=全部 | hard=难例论文 | hardev=硬证据正例相关格子
    z==2 默认剔除（keep_uncertain=True 时保留供校准分析）。"""
    sp = json.loads((BASE / "gold_split.json").read_text(encoding="utf-8"))
    keep = set(sp[split])
    hard = _hard_set()
    out = []
    for l in (BASE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        g = json.loads(l)
        if g["task_id"] not in keep:
            continue
        if g["z"] == 2 and not keep_uncertain:
            continue
        if subset == "hard" and g["task_id"] not in hard:
            continue
        if subset == "hardev" and not (g["z"] == 1 and g["ev"] in ("direct", "unique_holder", "adjudicated")) and g["z"] != 0:
            continue
        out.append((g["task_id"], g["position"], g["entity_id"], g["z"]))
    return out


def _auprc(y, s):
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(tp[-1], 1)
    prec = np.concatenate([[1.0], prec])
    rec = np.concatenate([[0.0], rec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def _ece(y, s, n_bins=10):
    y, s = np.asarray(y), np.asarray(s)
    e, tot = 0.0, len(y)
    for lo, hi in [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]:
        m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if m.sum():
            e += m.sum() / tot * abs(y[m].mean() - s[m].mean())
    return float(e)


def eval_task_a(scores: dict, gold: list) -> dict:
    y = [g[3] for g in gold]
    s = [scores.get((g[0], g[1], g[2]), 0.0) for g in gold]
    pred = [1 if v >= 0.5 else 0 for v in s]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"n": len(y), "AUPRC": _auprc(y, s), "P@0.5": p, "R@0.5": r,
            "F1@0.5": 2 * p * r / max(p + r, 1e-9), "ECE": _ece(y, s)}


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def eval_task_b(scores: dict, gold: list, ks=(3, 5)) -> dict:
    """每 (task_id, entity_id) 为一次查询。"""
    queries = defaultdict(dict)   # q -> position -> z
    for tid, pos, eid, z in gold:
        queries[(tid, eid)][pos] = z
    ndcg = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    mrr = []
    for (tid, eid), pos2z in queries.items():
        n_pos = sum(pos2z.values())
        if n_pos == 0:
            continue
        ranked = sorted(pos2z, key=lambda p: -scores.get((tid, p, eid), 0.0))
        gains = [pos2z[p] for p in ranked]
        ideal = sorted(pos2z.values(), reverse=True)
        for k in ks:
            kk = min(k, len(ranked))
            ndcg[k].append(_dcg(gains[:kk]) / max(_dcg(ideal[:kk]), 1e-9))
            rec[k].append(sum(gains[:kk]) / n_pos)
        rr = next((1.0 / (i + 1) for i, g in enumerate(gains) if g == 1), 0.0)
        mrr.append(rr)
    out = {"n_queries": len(mrr), "MRR": float(np.mean(mrr)) if mrr else 0.0}
    for k in ks:
        out[f"NDCG@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
        out[f"Recall@{k}"] = float(np.mean(rec[k])) if rec[k] else 0.0
    return out


def permutation_test(scores_a: dict, scores_b: dict, gold: list,
                     metric="NDCG@5", n_perm=10000, seed=42) -> float:
    """配对置换检验：模型 A 与 B 在任务 B 各查询上的指标差异显著性（双侧 p 值）。"""
    rng = random.Random(seed)
    queries = defaultdict(dict)
    for tid, pos, eid, z in gold:
        queries[(tid, eid)][pos] = z
    queries = {q: d for q, d in queries.items() if sum(d.values()) > 0}

    def per_query(scores):
        vals = {}
        for (tid, eid), pos2z in queries.items():
            ranked = sorted(pos2z, key=lambda p: -scores.get((tid, p, eid), 0.0))
            gains = [pos2z[p] for p in ranked]
            ideal = sorted(pos2z.values(), reverse=True)
            k = int(metric.split("@")[1]) if "@" in metric else 3
            kk = min(k, len(ranked))
            vals[(tid, eid)] = _dcg(gains[:kk]) / max(_dcg(ideal[:kk]), 1e-9) if "NDCG" in metric \
                else sum(gains[:kk]) / max(sum(ideal), 1)
        return vals

    va, vb = per_query(scores_a), per_query(scores_b)
    qs = list(queries)
    diff_obs = abs(np.mean([va[q] for q in qs]) - np.mean([vb[q] for q in qs]))
    cnt = 0
    for _ in range(n_perm):
        d = 0.0
        for q in qs:
            x, y = (va[q], vb[q]) if rng.random() < 0.5 else (vb[q], va[q])
            d += x - y
        if abs(d / len(qs)) >= diff_obs:
            cnt += 1
    return cnt / n_perm
