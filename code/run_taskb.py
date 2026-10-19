# -*- coding: utf-8 -*-
"""
M5 任务B - 任务驱动角色匹配推荐评测（初稿 5.5 任务 B，答 RQ2）
用法：
  python pipeline/run_taskb.py --stage train    # 重训 B4/B5/B6 并存 checkpoint（同 seed=42 口径）
  python pipeline/run_taskb.py --stage eval     # 语料画像 + 查询排序评测
流程（eval）：
  1. 在 500 篇训练语料上跑 B4/B5/B6 归属推断，按作者聚合（noisy-or 跨论文）成
     作者×方法（×角色）知识画像；B1/B2/B3 由历史先验直接构建画像（机制对照）；
  2. 对 queries.jsonl 的每条查询（所需方法 k, 所需角色 r）排序候选作者，
     指标 NDCG@{5,10,20}、Recall@{10,20}、MRR；
  3. 分层报告：K内全部（主集）/ K内 developed / K内 applied / 全量（参考），主集按角色分列。
画像口径：
  B1 共著共享:   score = 1{w_ak>0}
  B2 贡献标量:   score = w_ak × c_a（c_a=作者在语料中平均工作角色数/6，缺省取全库均值）
  B3 HeteAlloc:  score = w_ak / Σ_{候选} w_a'k（候选内分配份额，机制对照：归一化）
  B4 角色GNN:    score = p(a,k)（a=作者语料聚合特征，k=实体特征；角色盲，体现"无显式归属层"）
  B5 通用MIL:    同上（instance MLP 直接打分）
  B6-rb 角色盲:  score = w(a,k)（消融，隔离角色分辨力的贡献）
  B6-role 角色版: score = w(a,k) × P6(r|a,k)（消融，隔离作者活跃度因子的贡献）
  B6 本文:       score = w(a,k) × P6(r|a,k) × c_a（v3 完整版：历史强度 × 角色匹配 × 活跃度）
打分方式（2026-10-13 修正 + v3 裁决）：
  v1/v2 的查询时直接前向打分忠实还原初稿 4.5.2 的匹配函数，但纯推荐场景下论文内
  证据缺席，NN 模型输出压在零附近、先验信号被噪声通道淹没（v2 结果存 taskb_results_v2.json）。
  v3 让全部模型共享同一角色盲历史强度 w，只允许 B6 用归属层做角色分配——
  B6 与 B6-rb 的唯一差异即角色分辨力，是 RQ2 的最小公平消融。
作者特征 = 语料内各论文 16 维作者特征的均值（无记录 → 零向量，诚实冷启动）；
实体特征 = 语料全部提及上下文（cap 3 段）TF-IDF + type/ctx 众数 one-hot，
未覆盖实体退回 canonical 名 TF-IDF + 零 one-hot。
"""
import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from run_b4_b5 import RoleGNN, MILAttribution, train_b4, train_b5
from run_b6 import AttributionLayer, train_b6, ROLE_SET, PI_DEFAULT

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
CTX_WIN = 300


# ---------------- 语料论文特征（对齐 features.paper_arrays 的结构） ----------------
def corpus_mentions():
    def _l():
        m2e = pd.read_pickle(BASE / "mention2entity.pkl")
        recs = {}
        for l in (BASE / "method_mentions.jsonl").read_text(encoding="utf-8").splitlines():
            m = json.loads(l)
            if str(m.get("verified")) != "True":
                continue
            key = f"{m['doi']}|{m['span'].lower()}"
            eid = m2e.get(key)
            if eid is None:
                continue
            recs.setdefault(m["doi"], []).append(
                {"entity_id": eid, "span": m["span"], "offset": int(m["offset"]),
                 "type": m["type"], "role_ctx": m["context_role"]})
        return recs
    return F._load("corpus_mentions", _l)


def corpus_paper_arrays(doi, methods_text, authors_df):
    """返回 (A, positions, E, eids, oids)。结构与 features.paper_arrays 对齐。"""
    task = {"authors": []}
    oids = []
    for a in authors_df.itertuples():
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        declared = [x.get("role_canonical", "") for x in roles]
        task["authors"].append({"position": int(a.position), "declared_roles": declared})
        oids.append(a.openalex_id if pd.notna(a.openalex_id) else None)
    A, positions = F.author_features(task)

    E, eids = [], []
    import text_encoder as TE
    by_eid = defaultdict(lambda: {"ctx": [], "type": None, "role_ctx": None})
    for m in corpus_mentions().get(doi, []):
        s = max(0, m["offset"] - CTX_WIN)
        e = min(len(methods_text), m["offset"] + len(m["span"]) + CTX_WIN)
        by_eid[m["entity_id"]]["ctx"].append(methods_text[s:e])
        by_eid[m["entity_id"]]["type"] = m["type"]
        by_eid[m["entity_id"]]["role_ctx"] = m["role_ctx"]
    eid_list = list(by_eid.keys())
    T = TE.encode_texts([" ".join(by_eid[e]["ctx"]) for e in eid_list],
                        tfidf_vectorizer=F.tfidf())
    for t, eid in zip(T, eid_list):
        d = by_eid[eid]
        oh_t = np.zeros(6, dtype=np.float32)
        if d["type"] in F.ENTITY_TYPES:
            oh_t[F.ENTITY_TYPES.index(d["type"])] = 1.0
        oh_c = np.zeros(4, dtype=np.float32)
        if d["role_ctx"] in F.CTX_ROLES:
            oh_c[F.CTX_ROLES.index(d["role_ctx"])] = 1.0
        E.append(np.concatenate([t, oh_t, oh_c]))
        eids.append(eid)
    if not E:
        import text_encoder as TE
        return A, positions, np.zeros((0, TE.TEXT_DIM + 10), dtype=np.float32), [], oids
    return A, positions, np.stack(E), eids, oids


def prior_lo_from_oids(oids, eids, prior_lookup):
    lo = np.full((len(oids), len(eids)), np.log(PI_DEFAULT / (1 - PI_DEFAULT)), dtype=np.float32)
    for i, oid in enumerate(oids):
        if oid is None:
            continue
        for j, eid in enumerate(eids):
            w = prior_lookup.get((oid, eid))
            if w is not None:
                pi = 1.0 - np.exp(-max(float(w), 0.0))
                pi = min(max(pi, 1e-4), 0.99)
                lo[i, j] = np.log(pi / (1 - pi))
    return torch.tensor(lo)


# ---------------- 训练 ----------------
def stage_train(model_name=None):
    if model_name in (None, "b4"):
        train_b4(300, 1e-3, 42)
        print("ckpt_b4.pt 已存")
    if model_name in (None, "b5"):
        train_b5(300, 1e-3, 42)
        print("ckpt_b5.pt 已存")
    if model_name in (None, "b6"):
        train_b6(300, 1e-3, 42, 1.0, 0.1)
        print("ckpt_b6.pt 已存")


# ---------------- 查询时打分所需的作者/实体特征 ----------------
def build_qt_features():
    """作者聚合特征 + 实体特征构造函数。返回 (author_feat, ent_feat, pl, arts, aus)。"""
    arts = pd.read_pickle(BASE / "articles.pkl")
    aus = pd.read_pickle(BASE / "authorships.pkl")
    aus_by_doi = {d: g.sort_values("position") for d, g in aus.groupby("doi")}
    cm = corpus_mentions()
    v = F.tfidf()
    pl = F.prior_lookup()

    fsum, fcnt = defaultdict(lambda: np.zeros(16, dtype=np.float64)), defaultdict(int)
    ectx = defaultdict(list)
    etype, ectxr = defaultdict(Counter), defaultdict(Counter)
    n_papers = 0
    for row in arts.itertuples():
        g = aus_by_doi.get(row.doi)
        if g is None:
            continue
        n_papers += 1
        task = {"authors": []}
        oids = []
        for a in g.itertuples():
            roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
            declared = [x.get("role_canonical", "") for x in roles]
            task["authors"].append({"position": int(a.position), "declared_roles": declared})
            oids.append(a.openalex_id if pd.notna(a.openalex_id) else None)
        A, _ = F.author_features(task)
        for i, oid in enumerate(oids):
            if oid is not None:
                fsum[oid] += A[i]
                fcnt[oid] += 1
        text = row.methods_text or ""
        for m in cm.get(row.doi, []):
            s = max(0, m["offset"] - CTX_WIN)
            e = min(len(text), m["offset"] + len(m["span"]) + CTX_WIN)
            ectx[m["entity_id"]].append(text[s:e])
            etype[m["entity_id"]][m["type"]] += 1
            ectxr[m["entity_id"]][m["role_ctx"]] += 1
    author_feat = {o: (fsum[o] / fcnt[o]).astype(np.float32) for o in fsum}

    # canonical 名表（K 内 + E1 新实体），供语料未覆盖实体回退
    canon = {}
    for f in ("entities.pkl", "entities_post.pkl"):
        df = pd.read_pickle(BASE / f)
        for r in df.itertuples():
            canon[r.entity_id] = r.canonical

    _ecache = {}
    def ent_feat(eid):
        import text_encoder as TE
        if eid in _ecache:
            return _ecache[eid]
        if eid in ectx:
            t = TE.encode_texts([" ".join(ectx[eid][:3])], tfidf_vectorizer=v)[0]
            oh_t = np.zeros(6, np.float32)
            oh_c = np.zeros(4, np.float32)
            ty = etype[eid].most_common(1)[0][0]
            cr = ectxr[eid].most_common(1)[0][0]
            if ty in F.ENTITY_TYPES:
                oh_t[F.ENTITY_TYPES.index(ty)] = 1.0
            if cr in F.CTX_ROLES:
                oh_c[F.CTX_ROLES.index(cr)] = 1.0
        else:  # 语料未覆盖：canonical 名编码 + 零 one-hot
            t = TE.encode_texts([canon.get(eid, "")], tfidf_vectorizer=v)[0]
            oh_t = np.zeros(6, np.float32)
            oh_c = np.zeros(4, np.float32)
        out = np.concatenate([t, oh_t, oh_c])
        _ecache[eid] = out
        return out

    n_covered = sum(1 for q_e in canon if q_e in ectx)
    print(f"查询时特征构建完成：{n_papers} 篇语料论文 | 作者特征 {len(author_feat)} 人 | "
          f"语料覆盖实体 {len(ectx)} 个（其余回退 canonical 名）")
    return author_feat, ent_feat, pl, arts, aus


def prior_lo_1d(oids, eid, prior_lookup):
    """[na,1] 的先验 log-odds，π=1-exp(-w) 压缩（与 B6 训练口径一致）。"""
    lo = np.full((len(oids), 1), np.log(PI_DEFAULT / (1 - PI_DEFAULT)), dtype=np.float32)
    for i, oid in enumerate(oids):
        if oid is None:
            continue
        w = prior_lookup.get((oid, eid))
        if w is not None:
            pi = 1.0 - np.exp(-max(float(w), 0.0))
            pi = min(max(pi, 1e-4), 0.99)
            lo[i, 0] = np.log(pi / (1 - pi))
    return torch.tensor(lo)


def make_nn_scorer(kind, model, author_feat, ent_feat, pl):
    """按查询批量前向打分：第一次遇到某 qid 时对其全部候选一次前向，之后查缓存。"""
    cache = {}
    zero = np.zeros(16, np.float32)

    def fn(c, q):
        if q["qid"] not in cache:
            gset = set(q["gold"])
            cands = list(dict.fromkeys(q["gold"])) + \
                [x for x in dict.fromkeys(q["neg_hard"] + q["neg_easy"]) if x not in gset]
            A = torch.tensor(np.stack([author_feat.get(x, zero) for x in cands]))
            E = torch.tensor(ent_feat(q["entity_id"])[None, :])
            with torch.no_grad():
                if kind == "b4":
                    s = torch.sigmoid(model(A, E))[:, 0].numpy()
                elif kind == "b5":
                    s = torch.sigmoid(model.instance_logits(A, E))[:, 0].numpy()
                else:  # b6：取查询角色通道，含先验注入
                    plo = prior_lo_1d(cands, q["entity_id"], pl)
                    out = torch.sigmoid(model(A, E, plo))[:, 0, :].numpy()
                    ri = ROLE_SET.index(q["role"]) if q["role"] in ROLE_SET else None
                    s = out[:, ri] if ri is not None else out.max(axis=1)
            cache[q["qid"]] = {x: float(s[i]) for i, x in enumerate(cands)}
        return cache[q["qid"]].get(c, 0.0)
    return fn


# ---------------- v3：B6 角色分辨画像（w × P6(r|a,k)） ----------------
@torch.no_grad()
def build_b6_role_profiles(ckpt_name="ckpt_b6.pt", cache_name="b6_profiles.pkl"):
    """语料上 B6 逐角色归属概率的 noisy-or 聚合：Z6(oid,eid) -> [R]。
    结果缓存（构建需全语料前向，约数分钟）；多 seed 时传不同 ckpt/cache 名。"""
    cache = BASE / cache_name
    if cache.exists():
        with cache.open("rb") as f:
            prof = pickle.load(f)
        print(f"B6 角色画像已加载缓存：{len(prof)} 个 (作者,实体) 对")
        return prof
    arts = pd.read_pickle(BASE / "articles.pkl")
    aus = pd.read_pickle(BASE / "authorships.pkl")
    aus_by_doi = {d: g.sort_values("position") for d, g in aus.groupby("doi")}
    pl = F.prior_lookup()
    import text_encoder as TE
    m6 = AttributionLayer(16, TE.TEXT_DIM + 10)
    m6.load_state_dict(torch.load(BASE / ckpt_name))
    m6.eval()
    acc = defaultdict(lambda: np.ones(len(ROLE_SET)))   # 逐角色 1-p 连乘
    n_papers = 0
    for row in arts.itertuples():
        g = aus_by_doi.get(row.doi)
        if g is None or row.doi not in corpus_mentions():
            continue
        A, positions, E, eids, oids = corpus_paper_arrays(row.doi, row.methods_text, g)
        if not eids:
            continue
        n_papers += 1
        plo = prior_lo_from_oids(oids, eids, pl)
        p6 = torch.sigmoid(m6(torch.tensor(A), torch.tensor(E), plo)).numpy()
        for i, oid in enumerate(oids):
            if oid is None:
                continue
            for j, eid in enumerate(eids):
                acc[(oid, eid)] = acc[(oid, eid)] * (1 - p6[i, j, :])
    prof = {k: 1 - v for k, v in acc.items()}
    with cache.open("wb") as f:
        pickle.dump(prof, f)
    print(f"B6 角色画像构建完成：{n_papers} 篇语料论文 | {len(prof)} 个 (作者,实体) 对，已缓存")
    return prof


def role_hist_by_author(aus):
    """作者在语料中的声明工作角色直方图（P6 的第一级回退）。"""
    work = F.METHOD_ROLES
    h = defaultdict(lambda: np.zeros(len(ROLE_SET)))
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        for x in {r.get("role_canonical", "") for r in roles} & work:
            if x in ROLE_SET:
                h[a.openalex_id][ROLE_SET.index(x)] += 1.0
    return h


def make_b6_profile_scorer(prof, hist, pl, c_by_oid=None, c_mean=0.3):
    """v3 打分：s(a,k,r) = w(a,k) × P6(r|a,k)（× c_a 若给，即 B6-full 完整版）。
    P6：语料归属画像逐角色归一化 -> 作者角色直方图 -> 均匀分布。
    试生产记录：曾测试查询级冷启动回退（全零先验查询上以 ε·P6·c_a 破并列），
    实测为反信号（金标作者历史规模显著小于负例，中位 13 vs 43 篇），已撤销，
    保持 w=0 → 0 的原始定义。见 7.4 节局限分析。"""
    unif = np.ones(len(ROLE_SET)) / len(ROLE_SET)
    p_cache = {}

    def p6(oid, eid):
        if (oid, eid) in p_cache:
            return p_cache[(oid, eid)]
        z = prof.get((oid, eid))
        if z is not None and z.sum() > 1e-9:
            p = z / z.sum()
        else:
            h = hist.get(oid)
            p = h / h.sum() if (h is not None and h.sum() > 0) else unif
        p_cache[(oid, eid)] = p
        return p

    def fn(c, q):
        w = float(pl.get((c, q["entity_id"]), 0.0))
        if w <= 0:
            return 0.0
        ri = ROLE_SET.index(q["role"]) if q["role"] in ROLE_SET else None
        p = p6(c, q["entity_id"])
        s = w * (float(p[ri]) if ri is not None else float(p.max()))
        if c_by_oid is not None:
            s *= c_by_oid.get(c, c_mean)
        return s
    return fn


# ---------------- 查询评测（多正例，方案A口径） ----------------
def eval_queries(queries, score_fn, seed=42):
    """多正例排序指标：NDCG@k（二元相关，理想序截断 min(|G|,k)）、
    Recall@k = |top-k ∩ G|/|G|、MAP、MRR。确定性乱序破并列。"""
    rng = np.random.default_rng(seed)
    M = defaultdict(list)
    for q in queries:
        gold = list(dict.fromkeys(q["gold"]))
        gset = set(gold)
        negs = [x for x in dict.fromkeys(q["neg_hard"] + q["neg_easy"]) if x not in gset]
        cands = gold + negs
        keys = rng.random(len(cands))
        scores = np.array([score_fn(c, q) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], keys[i]))
        rel = np.array([1.0 if cands[i] in gset else 0.0 for i in order])
        ng = len(gset)
        for k in (5, 10, 20):
            kk = min(k, len(rel))
            dcg = float(np.sum(rel[:kk] / np.log2(np.arange(2, kk + 2))))
            ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(ng, k) + 2))))
            M[f"NDCG@{k}"].append(dcg / ideal)
        for k in (10, 20):
            M[f"Recall@{k}"].append(float(rel[:k].sum()) / ng)
        # MAP / MRR
        hits, ap, mrr = 0, 0.0, 0.0
        for rank, r in enumerate(rel, 1):
            if r > 0:
                hits += 1
                ap += hits / rank
                if mrr == 0.0:
                    mrr = 1.0 / rank
        M["MAP"].append(ap / ng)
        M["MRR"].append(mrr)
    return {m: float(np.mean(v)) for m, v in M.items()}


def stage_eval(queries_file="queries_v2.jsonl", subset_filter=None):
    queries = [json.loads(l) for l in (BASE / queries_file).read_text(encoding="utf-8").splitlines()]
    author_feat, ent_feat, pl, arts, aus = build_qt_features()

    import text_encoder as TE
    a_dim, e_dim = 16, TE.TEXT_DIM + 10
    m4 = RoleGNN(a_dim, e_dim); m4.load_state_dict(torch.load(BASE / "ckpt_b4.pt")); m4.eval()
    m5 = MILAttribution(a_dim, e_dim); m5.load_state_dict(torch.load(BASE / "ckpt_b5.pt")); m5.eval()
    prof6 = build_b6_role_profiles()
    hist6 = role_hist_by_author(aus)

    # B2 贡献标量 c_a
    work = F.METHOD_ROLES
    c_by_oid, vals = {}, defaultdict(list)
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        nw = len({x.get("role_canonical", "") for x in roles} & work)
        vals[a.openalex_id].append(nw / 6.0)
    c_by_oid = {o: float(np.mean(v)) for o, v in vals.items()}
    c_mean = float(np.mean(list(c_by_oid.values()))) if c_by_oid else 0.3

    def s_b1(c, q): return 1.0 if pl.get((c, q["entity_id"]), 0) > 0 else 0.0
    def s_b2(c, q): return float(pl.get((c, q["entity_id"]), 0.0)) * c_by_oid.get(c, c_mean)
    def s_b3(c, q):
        gset = set(q["gold"])
        cands = list(dict.fromkeys(q["gold"])) + \
            [x for x in dict.fromkeys(q["neg_hard"] + q["neg_easy"]) if x not in gset]
        tot = sum(float(pl.get((x, q["entity_id"]), 0.0)) for x in cands) + 1e-9 * len(cands)
        return (float(pl.get((c, q["entity_id"]), 0.0)) + 1e-9) / tot
    def s_b4(c, q): return nn_b4(c, q)
    def s_b5(c, q): return nn_b5(c, q)
    def s_b6rb(c, q): return float(pl.get((c, q["entity_id"]), 0.0))   # 角色盲消融
    nn_b4 = make_nn_scorer("b4", m4, author_feat, ent_feat, pl)
    nn_b5 = make_nn_scorer("b5", m5, author_feat, ent_feat, pl)
    s_b6role = make_b6_profile_scorer(prof6, hist6, pl)                      # 消融：w × P6
    s_b6 = make_b6_profile_scorer(prof6, hist6, pl, c_by_oid, c_mean)      # 本文：w × P6 × c_a

    models = [("B1 共著共享", s_b1), ("B2 贡献标量", s_b2), ("B3 HeteAlloc", s_b3),
              ("B4 角色GNN", s_b4), ("B5 通用MIL", s_b5), ("B6-rb 角色盲", s_b6rb),
              ("B6-role 角色版", s_b6role), ("B6 本文", s_b6)]

    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    all_subsets = {
        "K内全部（主集）": inK,
        "K内 developed": [q for q in inK if q["ctx"] == "developed"],
        "K内 applied": [q for q in inK if q["ctx"] == "applied"],
        "E1 方法冷启动": [q for q in queries if q["entity_id"].startswith("E1")],
        "全量（含方法冷启动，参考）": queries,
    }
    subsets = {k: v for k, v in all_subsets.items()
               if subset_filter is None or k in subset_filter}
    lines = []
    out = {}
    for name, qs in subsets.items():
        lines.append(f"\n##### {name}（{len(qs)} 条查询）#####")
        header = f"{'模型':<14}{'NDCG@5':>8}{'NDCG@10':>8}{'NDCG@20':>8}{'R@10':>7}{'R@20':>7}{'MAP':>7}{'MRR':>7}"
        lines.append(header)
        for mname, fn in models:
            r = eval_queries(qs, fn)
            out[f"{name}|{mname}"] = r
            lines.append(f"{mname:<14}{r['NDCG@5']:>8.3f}{r['NDCG@10']:>8.3f}{r['NDCG@20']:>8.3f}"
                         f"{r['Recall@10']:>7.3f}{r['Recall@20']:>7.3f}{r['MAP']:>7.3f}{r['MRR']:>7.3f}")
        # 主集按角色分列（仅 B6 与最强基线）
        if name.startswith("K内全部"):
            lines.append("\n  主集按角色分列（NDCG@10 / MRR）：")
            for role in ROLE_SET:
                qr = [q for q in qs if q["role"] == role]
                if not qr:
                    continue
                r6 = eval_queries(qr, s_b6)
                best_b = max((eval_queries(qr, fn) for _, fn in models[:-1]), key=lambda r: r["NDCG@10"])
                lines.append(f"    {role:<18}B6 {r6['NDCG@10']:.3f}/{r6['MRR']:.3f} | "
                             f"最强基线 {best_b['NDCG@10']:.3f}/{best_b['MRR']:.3f} (n={len(qr)})")
    report = "\n".join(lines)
    print(report)
    tag = "v2" if "v2" in queries_file else "v1"
    if subset_filter is not None:
        tag += "_" + "_".join(sorted(subset_filter))[:20].replace("（", "").replace("）", "")
    (BASE / f"taskb_report_{tag}.txt").write_text(report, encoding="utf-8")
    (BASE / f"taskb_results_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--model", default=None, choices=[None, "b4", "b5", "b6"])
    ap.add_argument("--queries", default="queries_v2.jsonl")
    ap.add_argument("--subsets", default=None, nargs="+",
                    help="只跑指定子集（按子集名精确匹配），不给则全跑")
    args = ap.parse_args()
    if args.stage == "train":
        stage_train(args.model)
    else:
        stage_eval(args.queries, args.subsets)


if __name__ == "__main__":
    main()
