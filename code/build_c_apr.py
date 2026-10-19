# -*- coding: utf-8 -*-
"""
M1 收尾（离线部分）：由 authorships.pkl 构建 C_apr 角色标注宽表与论文语料统计
产出：
  data/processed/c_apr.csv          论文×作者×CRediT角色 0/1 宽表（4.3 节粗监督的直接输入）
  data/processed/corpus_stats.txt   论文 5.1 节统计表的回填素材
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

CREDIT_14 = [
    "conceptualization", "data-curation", "formal-analysis", "funding-acquisition",
    "investigation", "methodology", "project-administration", "resources",
    "software", "supervision", "validation", "visualization",
    "writing-original-draft", "writing-review-editing",
]

df_a = pd.read_pickle(OUT_DIR / "articles.pkl")
df_au = pd.read_pickle(OUT_DIR / "authorships.pkl")

rows = []
for _, r in df_au.iterrows():
    roles = {x.get("role_canonical") for x in json.loads(r["roles"])}
    rec = {
        "doi": r["doi"],
        "position": r["position"],
        "author": f"{r['surname']}, {r['given_names']}",
        "openalex_id": r.get("openalex_id"),
    }
    rec.update({c: int(c in roles) for c in CREDIT_14})
    rows.append(rec)

df_c = pd.DataFrame(rows)
df_c.to_csv(OUT_DIR / "c_apr.csv", index=False, encoding="utf-8-sig")

# ---- 语料统计（论文 5.1 节回填素材） ----
n_papers = df_a["doi"].nunique()
c1_count = int(df_c[CREDIT_14].sum().sum())
role_per_author_paper = df_c[CREDIT_14].sum(axis=1).mean()
stats = f"""===== 语料统计（M1 验收 / 论文 5.1 节回填素材）=====
论文数: {n_papers}
去重后作者-论文记录: {len(df_c)}（平均每篇 {len(df_c)/n_papers:.1f} 位作者）
唯一作者数（按姓名）: {df_c['author'].nunique()}
OpenAlex 匹配成功: {df_c['openalex_id'].notna().sum()} 条（{df_c['openalex_id'].notna().mean():.1%}）
唯一 OpenAlex 作者: {df_c['openalex_id'].nunique()}
C_apr=1 标注总量: {c1_count}
平均每作者-论文的角色数: {role_per_author_paper:.2f}

--- CRediT 角色分布（C_apr=1 频次）---
""" + "\n".join(f"  {c}: {int(df_c[c].sum())}" for c in CREDIT_14)
print(stats)
with open(OUT_DIR / "corpus_stats.txt", "w", encoding="utf-8") as f:
    f.write(stats)
