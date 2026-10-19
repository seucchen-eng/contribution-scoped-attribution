# -*- coding: utf-8 -*-
"""
M5 任务B v2 - 多正例时序推荐查询集（2026-10-13 评测设计裁决，方案A）
裁决要点（试生产记录）：
  v1 缺陷：正例="论文内唯一角色持有者"，94.8% 无历史先验、99.6% 不在训练语料，
           而难负例恰是该方法的历史专家——构造性对抗，任何画像模型都不可能赢。
  v2 金标：gold(k,r) = 评测期（250 篇 PLOS CB, t*后）内所有"在提及方法 k 的论文中
           声明承担工作角色 r"的作者（OpenAlex 匹配），多正例、跨论文汇总、ctx 无关；
  难负例：t* 前有 k 历史先验（w>0）但不在 gold(k,r) 的作者（"过时专家"，≤50 人）；
  易负例：先验库中既无 k 历史也不在 gold 的作者（补齐至每查询 100 个负例）；
  同 (doi,eid,r) 去重，ctx 取该实体内多数提及的 context_role（仅作分层标签）。
输出：data/processed/queries_v2.jsonl
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"
WORK = sorted(F.METHOD_ROLES)
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    aus = pd.read_pickle(BASE / "authorships_post.pkl")
    m2e_df = pd.read_pickle(BASE / "mention2entity_post.pkl")
    m2e = {f"{r.doi}|{str(r.span).lower()}": r.entity_id for r in m2e_df.itertuples()}

    # 每篇论文各工作角色的声明持有者
    holders = defaultdict(lambda: defaultdict(list))   # holders[doi][r] = [oid]
    for a in aus.itertuples():
        if pd.isna(a.openalex_id):
            continue
        roles = json.loads(a.roles) if isinstance(a.roles, str) else a.roles
        rc = {x.get("role_canonical", "") for x in roles} & set(WORK)
        for r in rc:
            holders[a.doi][r].append(a.openalex_id)

    # 评测期全部已核验提及：(doi, eid, ctx)，同 (doi,eid) 聚合取多数 ctx
    ent_ctx = defaultdict(Counter)
    for l in (BASE / "method_mentions_post.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(l)
        if str(m.get("verified")) != "True":
            continue
        eid = m2e.get(f"{m['doi']}|{m['span'].lower()}")
        if eid is None:
            continue
        ent_ctx[(m["doi"], eid)][m["context_role"]] += 1

    # gold(k,r)：评测期全部承担者
    gold = defaultdict(set)
    for (doi, eid) in ent_ctx:
        for r, oids in holders.get(doi, {}).items():
            gold[(eid, r)].update(oids)

    # 先验库（负例来源）
    pl = F.prior_lookup()
    by_ent = defaultdict(set)
    for (oid, eid), w in pl.items():
        if w and float(w) > 0:
            by_ent[eid].add(oid)
    universe = np.array(sorted({oid for (oid, _) in pl.keys()}))

    # canonical 名表
    canon = {}
    for f in ("entities.pkl", "entities_post.pkl"):
        df = pd.read_pickle(BASE / f)
        for r in df.itertuples():
            canon[r.entity_id] = r.canonical

    queries = []
    qn = 0
    for (doi, eid), cc in sorted(ent_ctx.items()):
        ctx = cc.most_common(1)[0][0]
        for r in WORK:
            if not holders.get(doi, {}).get(r):
                continue
            g = sorted(gold.get((eid, r), set()))
            if not g:
                continue
            gset = set(g)
            hard_pool = sorted(by_ent.get(eid, set()) - gset)
            if len(hard_pool) > 50:
                hard = list(rng.choice(hard_pool, 50, replace=False))
            else:
                hard = hard_pool
            easy_pool = np.setdiff1d(universe, np.array(sorted(gset | by_ent.get(eid, set()))))
            n_easy = min(100 - len(hard), len(easy_pool))
            easy = list(rng.choice(easy_pool, n_easy, replace=False)) if n_easy > 0 else []
            queries.append({
                "qid": f"Q{qn:05d}", "doi": doi, "entity_id": eid,
                "canonical": canon.get(eid, ""), "ctx": ctx, "role": r,
                "gold": g, "n_gold": len(g),
                "neg_hard": hard, "neg_easy": easy,
            })
            qn += 1

    with (BASE / "queries_v2.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    inK = [q for q in queries if not q["entity_id"].startswith("E1")]
    gs = np.array([q["n_gold"] for q in queries])
    nh = np.array([len(q["neg_hard"]) for q in inK]) if inK else np.array([0])
    print(f"查询总数 {len(queries)}（K内 {len(inK)} / E1 {len(queries)-len(inK)}）")
    print(f"金标规模：中位 {np.median(gs):.0f} | 均值 {gs.mean():.1f} | p90 {np.percentile(gs,90):.0f} | 最大 {gs.max()}")
    print(f"K内难负例：均值 {nh.mean():.1f} | 无难负例查询 {(nh==0).mean():.1%}")
    print(f"ctx 分布：{Counter(q['ctx'] for q in inK).most_common()}")
    print(f"角色分布：{Counter(q['role'] for q in inK).most_common()}")


if __name__ == "__main__":
    main()
