# -*- coding: utf-8 -*-
"""
M5 任务B - 查询集构建（初稿 5.5：任务驱动的角色匹配推荐）
用法：python pipeline/build_queries.py [--n-hard 50] [--n-easy 50] [--seed 42]

查询四元组：(任务描述, 所需方法 k, 所需角色 r, 实际承担作者 a*)
来源：t*=2024-01-01 之后的 250 篇 PLOS Comput Biol（articles_post/authorships_post/
      mention2entity_post），实际承担作者来自贡献声明的唯一持有者判定：
      某 (k, r) 在论文内只有一位作者声明角色 r（排除纯编辑角色者）→ 无歧义正例，
      且该作者须在 author_history 中有 t* 前历史（冷启动作者不构成可评测查询）。
候选池（4.5.3 hard negative 策略）：
  hard negatives：历史先验 w_ak>0（有能力却未承担）的非本文作者，按 w 降序取 n_hard
  easy negatives：无该方法历史的随机作者 n_easy 个
产物：data/processed/queries.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

WORK_ROLES = ["methodology", "software", "formal-analysis",
              "investigation", "data-curation", "validation"]
EDITORIAL = {"Section Editor", "Academic Editor"}


def parse_roles(raw):
    roles = json.loads(raw) if isinstance(raw, str) else raw
    return [(x.get("role_text", ""), x.get("role_canonical", "")) for x in roles]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hard", type=int, default=50)
    ap.add_argument("--n-easy", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    arts = pd.read_pickle(OUT_DIR / "articles_post.pkl")
    aus = pd.read_pickle(OUT_DIR / "authorships_post.pkl")
    m2e = pd.read_pickle(OUT_DIR / "mention2entity_post.pkl")
    hist = pd.read_pickle(OUT_DIR / "author_history.pkl")
    pri = pd.read_pickle(OUT_DIR / "priors.pkl")
    ents = pd.read_pickle(OUT_DIR / "entities.pkl")
    ename = dict(zip(ents.entity_id, ents.canonical))
    ep_path = OUT_DIR / "entities_post.pkl"
    if ep_path.exists():
        ep = pd.read_pickle(ep_path)
        ename.update(dict(zip(ep.entity_id, ep.canonical)))

    known_authors = set(hist.openalex_id.unique())
    # 先验索引：entity -> [(oid, w)] 降序
    pri_by_ent = defaultdict(list)
    for r in pri.itertuples():
        pri_by_ent[r.entity_id].append((r.openalex_id, float(r.weight)))
    for v in pri_by_ent.values():
        v.sort(key=lambda x: -x[1])
    pool = sorted(known_authors)

    title = dict(zip(arts.doi, arts.title))
    aus_by_doi = {d: g for d, g in aus.groupby("doi")}

    queries, seen = [], set()
    n_skip = defaultdict(int)
    for r in m2e.itertuples():
        if r.context_role not in ("developed", "applied"):
            n_skip["非工作语境实体"] += 1
            continue
        g = aus_by_doi.get(r.doi)
        if g is None:
            continue
        # 论文内各工作角色的唯一持有者（排除纯编辑）
        holders = defaultdict(list)  # role -> [oid]
        for a in g.itertuples():
            rl = parse_roles(a.roles)
            if rl and all(t in EDITORIAL for t, _ in rl):
                continue
            for _, rc in rl:
                if rc in WORK_ROLES and pd.notna(a.openalex_id):
                    holders[rc].append(a.openalex_id)
        for role in WORK_ROLES:
            hs = holders.get(role, [])
            if len(hs) != 1:
                n_skip["角色持有者有歧义或缺失"] += 1
                continue
            oid = hs[0]
            if oid not in known_authors:
                n_skip["正例作者无历史（冷启动）"] += 1
                continue
            key = (r.doi, r.entity_id, role, oid)
            if key in seen:
                continue
            seen.add(key)
            # 候选池
            coauthors = set(g.openalex_id.dropna())
            hard = [(o, w) for o, w in pri_by_ent.get(r.entity_id, [])
                    if o not in coauthors][:args.n_hard]
            hard_oids = [o for o, _ in hard]
            easy = rng.sample([o for o in pool if o not in coauthors
                               and o not in {h for h, _ in hard}],
                              min(args.n_easy, len(pool)))
            queries.append({
                "qid": f"Q{len(queries):05d}", "doi": r.doi,
                "title": title.get(r.doi, ""),
                "entity_id": r.entity_id, "canonical": ename.get(r.entity_id, "?"),
                "ctx": r.context_role,
                "role": role, "positive_oid": oid, "tier": "unique_holder",
                "neg_hard": hard_oids, "neg_easy": easy,
            })

    out = OUT_DIR / "queries.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    from collections import Counter
    rc = Counter(q["role"] for q in queries)
    inK = sum(1 for q in queries if not q["entity_id"].startswith("E1"))
    papers = len({q["doi"] for q in queries})
    print(f"===== 查询集构建报告 =====")
    print(f"查询数: {len(queries)}（来自 {papers} 篇评测期论文）")
    print(f"方法实体在既有空间 K 内: {inK} ({inK/max(len(queries),1):.1%})")
    print(f"角色分布: {dict(rc.most_common())}")
    print(f"平均每查询候选数: 1 + {args.n_hard} hard + {args.n_easy} easy ≈ {1+args.n_hard+args.n_easy}")
    print(f"跳过统计: {dict(n_skip)}")
    print(f"产物: {out}")


if __name__ == "__main__":
    main()
