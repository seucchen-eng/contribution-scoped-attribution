# -*- coding: utf-8 -*-
"""
M5 任务B - 评测期实体映射（只映射，不重建实体空间）
用法：python pipeline/map_entities_post.py [--threshold 0.85]
流程：
  1. 以训练期 entities.pkl 的 alias 建索引（同样的 normalize 规则）
  2. 评测期 method_mentions_post.jsonl 的唯一 canonical：
     a. 规范化精确命中 alias -> 直接映射到既有 entity_id
     b. 否则 TF-IDF（与 link_entities.py 同构的特征）对全部 alias 取最近邻，
        相似度 >= threshold 映射
     c. 否则分配新实体号（E1xxxx 段），写入 entities_post.pkl
产物：
  data/processed/mention2entity_post.pkl   (doi, entity, span) -> entity_id
  data/processed/entities_post.pkl         新增实体（既有空间之外的）
  data/processed/map_post_report.txt
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

STOP_PLURAL = re.compile(r"s$")


def normalize(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    toks = [t[:-1] if len(t) > 3 and STOP_PLURAL.search(t) else t for t in s.split()]
    return " ".join(sorted(t for t in toks if t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    ents = pd.read_pickle(OUT_DIR / "entities.pkl")
    alias2eid = {}
    alias_list, alias_eid = [], []
    for r in ents.itertuples():
        aliases = json.loads(r.aliases) if isinstance(r.aliases, str) else r.aliases
        for a in aliases:
            na = normalize(a)
            if not na:
                continue
            alias2eid.setdefault(na, r.entity_id)
            alias_list.append(na)
            alias_eid.append(r.entity_id)
    print(f"既有实体 {len(ents)} 个，alias {len(alias_list)} 条")

    recs = [json.loads(l) for l in
            (OUT_DIR / "method_mentions_post.jsonl").read_text(encoding="utf-8").splitlines()]
    freq = Counter(r["entity"] for r in recs)
    names = sorted(freq)
    print(f"评测期唯一 canonical: {len(names)}")

    # 1. 规范化精确命中
    mapping, todo = {}, []
    for n in names:
        nn = normalize(n)
        if nn in alias2eid:
            mapping[n] = alias2eid[nn]
        else:
            todo.append(n)
    print(f"精确命中: {len(mapping)} / {len(names)}")

    # 2. TF-IDF 最近邻（字符 3-5gram + 词 1-2gram，各半权，与 link_entities 同构）
    new_entities = []
    if todo:
        from scipy.sparse import hstack, vstack
        vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        all_names = alias_list + [normalize(n) for n in todo]
        Xc = vec_c.fit_transform(all_names)
        Xw = vec_w.fit_transform(all_names)
        X = hstack([Xc * 0.5, Xw * 0.5]).tocsr()
        Xn = X.multiply(1.0 / np.maximum(np.sqrt(X.multiply(X).sum(axis=1)), 1e-9)).tocsr()
        Q = Xn[len(alias_list):]
        A = Xn[:len(alias_list)]
        sims = (Q @ A.T).toarray()
        best = sims.argmax(axis=1)
        bestv = sims[np.arange(len(todo)), best]
        n_fuzzy = 0
        for i, n in enumerate(todo):
            if bestv[i] >= args.threshold:
                mapping[n] = alias_eid[best[i]]
                n_fuzzy += 1
            else:
                eid = f"E1{len(new_entities):04d}"
                mapping[n] = eid
                new_entities.append({"entity_id": eid, "canonical": n,
                                     "aliases": json.dumps([n], ensure_ascii=False),
                                     "freq": freq[n], "n_variants": 1})
        print(f"模糊映射: {n_fuzzy} | 新实体: {len(new_entities)}")

    # 3. 落盘
    rows = [{"doi": r["doi"], "entity": r["entity"], "span": r["span"],
             "type": r["type"], "context_role": r["context_role"],
             "entity_id": mapping[r["entity"]]} for r in recs]
    pd.DataFrame(rows).to_pickle(OUT_DIR / "mention2entity_post.pkl")
    if new_entities:
        pd.DataFrame(new_entities).to_pickle(OUT_DIR / "entities_post.pkl")

    n_exist = sum(1 for r in rows if not r["entity_id"].startswith("E1"))
    report = (f"评测期提及 {len(rows)} 条 -> 既有实体 {n_exist} 条 ({n_exist/len(rows):.1%})，"
              f"新实体 {len(rows)-n_exist} 条；唯一 canonical {len(names)}，"
              f"映射进既有空间 {len(names)-len(new_entities)} 个")
    print(report)
    with open(OUT_DIR / "map_post_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
