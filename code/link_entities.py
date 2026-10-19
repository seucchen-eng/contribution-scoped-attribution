# -*- coding: utf-8 -*-
"""
M2 证据提取 - 方法实体链接（归一到共享实体空间 K）
用法：python pipeline/link_entities.py [--threshold 0.85]
流程：
  1. 名称规范化（小写、去标点、单复数粗归一、词序排序）
  2. 精确合并同名变体
  3. TF-IDF（字符 3-5 gram + 词 1-2 gram）余弦相似度，分块计算
  4. 阈值 τ_link 并查集聚类；缩写 ↔ 全称规则补充
  5. 规范名 = 簇内频次最高的变体
产物：
  data/processed/entities.pkl          实体表（entity_id, canonical, aliases, freq）
  data/processed/mention2entity.pkl    提及 → entity_id 映射
  data/processed/link_samples.txt      人工阈值校准样本（高/中/低相似度各若干对）
"""
import argparse
import itertools
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


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def acronym(s: str) -> str:
    return "".join(t[0] for t in s.split() if t and t[0].isalpha())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            (OUT_DIR / "method_mentions.jsonl").read_text(encoding="utf-8").splitlines()]
    freq = Counter(r["entity"] for r in recs)          # canonical（LLM 给的规范名）出现频次
    names = sorted(freq)
    print(f"唯一实体名: {len(names)}")

    # 1-2. 规范化 + 精确合并
    norm_map = {}
    for n in names:
        norm_map.setdefault(normalize(n), []).append(n)
    groups = list(norm_map.values())
    rep = [sorted(g, key=lambda x: -freq[x]) for g in groups]  # 每组按频次降序
    print(f"规范化精确合并后: {len(groups)} 组")

    # 3. TF-IDF 相似度（组代表名之间）
    rep_names = [g[0] for g in rep]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xc = vec.fit_transform(rep_names)
    vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    Xw = vec_w.fit_transform(rep_names)
    # 字符与词特征各占一半权重
    from scipy.sparse import hstack
    X = hstack([Xc * 0.5, Xw * 0.5]).tocsr()
    Xn = X.multiply(1.0 / np.maximum(np.sqrt(X.multiply(X).sum(axis=1)), 1e-9)).tocsr()

    n = len(rep_names)
    dsu = DSU(n)
    samples = []  # (sim, a, b) 校准样本
    B = 1000
    for i0 in range(0, n, B):
        block = (Xn[i0:i0 + B] @ Xn.T).toarray()
        for bi in range(block.shape[0]):
            i = i0 + bi
            js = np.where(block[bi, i + 1:] >= args.threshold)[0] + i + 1
            for j in js:
                dsu.union(i, j)
                if len(samples) < 400:
                    samples.append((float(block[bi, j]), rep_names[i], rep_names[j]))

    # 4. 缩写 ↔ 全称补充规则
    acr_map = {}
    for i, nm in enumerate(rep_names):
        if re.fullmatch(r"[a-z0-9]{2,8}", nm.replace(" ", "")) and " " not in nm:
            acr_map.setdefault(nm, []).append(i)
    # 4. 缩写 ↔ 全称补充规则（仅当二者在同一篇论文中共同出现，即论文内有括号定义）
    # 先建 rep 组 -> 出现论文集合 的倒排
    name2rep = {}
    for i, g in enumerate(rep):
        for v in g:
            name2rep[v] = i
    rep_dois = {}
    for r in recs:
        ri = name2rep.get(r["entity"])
        if ri is not None:
            rep_dois.setdefault(ri, set()).add(r["doi"])
    for i, nm in enumerate(rep_names):
        a = acronym(nm)
        if len(a) >= 2 and a in acr_map and " " in nm:
            for j in acr_map[a]:
                shared = rep_dois.get(i, set()) & rep_dois.get(j, set())
                if shared:  # 同篇共现才合并，避免 pcr/srs/vt 类歧义
                    dsu.union(i, j)
                    samples.append((1.0, rep_names[i], rep_names[j] + "  [缩写规则·同篇共现]"))

    # 5. 生成实体表
    clusters = {}
    for i in range(n):
        clusters.setdefault(dsu.find(i), []).append(i)
    ent_rows, mention_map = [], {}
    for cid, idxs in enumerate(clusters.values()):
        variants = [v for i in idxs for v in rep[i]]
        canonical = max(variants, key=lambda x: freq[x])
        eid = f"E{cid:05d}"
        ent_rows.append({
            "entity_id": eid, "canonical": canonical,
            "aliases": json.dumps(sorted(set(variants), key=lambda x: -freq[x]), ensure_ascii=False),
            "freq": sum(freq[v] for v in variants),
            "n_variants": len(set(variants)),
        })
        for v in variants:
            mention_map[v] = eid
    df_e = pd.DataFrame(ent_rows).sort_values("freq", ascending=False)
    df_e.to_pickle(OUT_DIR / "entities.pkl")

    # 提及 → 实体映射（按 LLM canonical 名映射；span 原名也尝试）
    m2e = {}
    for r in recs:
        eid = mention_map.get(r["entity"])
        if eid is None:
            # 尝试 span 的规范化
            eid = mention_map.get(norm_map.get(normalize(r["span"]), [None])[0])
        if eid:
            m2e[f"{r['doi']}|{r['span'].lower()}"] = eid
    pd.Series(m2e).to_pickle(OUT_DIR / "mention2entity.pkl")

    # 校准样本
    samples.sort(key=lambda x: -x[0])
    with open(OUT_DIR / "link_samples.txt", "w", encoding="utf-8") as f:
        f.write("===== 高相似度对（应合并）抽样 =====\n")
        for sim, a, b in samples[:60]:
            f.write(f"{sim:.2f}  {a}  <->  {b}\n")
        f.write("\n===== 阈值附近对（需人工判断） =====\n")
        near = [s for s in samples if abs(s[0] - args.threshold) < 0.05][:60]
        for sim, a, b in near:
            f.write(f"{sim:.2f}  {a}  <->  {b}\n")

    print(f"\n===== 实体链接结果 =====")
    print(f"实体数 |K|: {len(df_e)}（合并前 {len(names)}）")
    print(f"实体频次分布：中位数 {df_e['freq'].median():.0f}，最大 {df_e['freq'].max()}")
    print(f"多变体实体: {(df_e['n_variants'] > 1).sum()} 个")
    print(f"Top-10 高频实体:")
    for _, r in df_e.head(10).iterrows():
        print(f"  [{r['freq']:>4}] {r['canonical']}")
    print(f"校准样本已写入 link_samples.txt（共 {len(samples)} 对）")


if __name__ == "__main__":
    main()
