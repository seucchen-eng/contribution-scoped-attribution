# -*- coding: utf-8 -*-
"""为深度裁决导出单篇论文的证据包：
作者角色、z2 格子、每个实体的提及上下文（前后 300 字符）、致谢/数据可用性段落。
用法: python pipeline/dump_evidence.py <doi后缀>  如 1010557
"""
import json, pickle, sys, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

def main(suffix):
    arts = pickle.load(open(BASE / "articles.pkl", "rb"))
    auth = pickle.load(open(BASE / "authorships.pkl", "rb"))
    ents = pickle.load(open(BASE / "entities.pkl", "rb"))
    tasks = json.loads((BASE / "z2_adjudication_tasks.json").read_text(encoding="utf-8"))
    tid = f"journal.pcbi.{suffix}"
    cells = tasks.get(tid)
    if not cells:
        print("no z2 cells for", tid); return
    row = arts[arts.doi.str.contains(suffix)].iloc[0]
    full_doi = row.doi
    a = auth[auth.doi == full_doi].sort_values("position")
    ename = dict(zip(ents.entity_id, ents.canonical))

    out = []
    out.append(f"# {row.title}\nDOI: {full_doi}  |  authors: {row.n_authors}\n")
    out.append("## 作者与角色")
    for r in a.itertuples():
        out.append(f"  [{r.position}] {r.given_names} {r.surname}: {r.roles}")
    if row.contrib_narrative:
        out.append("\n## 叙述性贡献声明\n" + row.contrib_narrative)

    # 全文关键段：致谢/基金/数据可用
    xml_path = BASE.parent / "raw_xml" / f"{full_doi.split('/')[-1]}.xml"
    xml = xml_path.read_text(encoding="utf-8", errors="ignore") if xml_path.exists() else ""
    for tag, label in [("ack", "致谢"), ("fn-group", "脚注组")]:
        for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S):
            txt = re.sub(r"<[^>]+>", " ", m.group(1))
            txt = re.sub(r"\s+", " ", txt).strip()
            out.append(f"\n## {label}\n{txt[:1500]}")
    m = re.search(r"Data [Aa]vailability(.{0,1200})", re.sub(r"<[^>]+>", " ", xml), re.S)
    if m:
        out.append("\n## 数据可用性\n" + re.sub(r"\s+", " ", m.group(1))[:1200])

    # 每个待裁决实体：所有提及上下文
    ent_ids = sorted({c["entity_id"] for c in cells})
    out.append("\n## 待裁决实体的提及上下文")
    for eid in ent_ids:
        cname = ename.get(eid, "?")
        out.append(f"\n### {eid}  {cname}")
        aliases_raw = ents[ents.entity_id == eid].iloc[0].aliases
        if isinstance(aliases_raw, str):
            aliases_raw = json.loads(aliases_raw)
        aliases = [str(a_).lower() for a_ in aliases_raw]
        found = []
        body = re.sub(r"<[^>]+>", " ", xml)
        body = re.sub(r"\s+", " ", body)
        for al in aliases:
            if len(al) < 3:
                continue
            for mm in re.finditer(re.escape(al), body, re.I):
                s = max(0, mm.start() - 300); e = min(len(body), mm.end() + 300)
                found.append({"ctx": body[s:e]})
        seen = set()
        for m_ in found[:12]:
            ctx = m_["ctx"]
            key = ctx[:80]
            if key in seen: continue
            seen.add(key)
            out.append(f"  - …{ctx}…")

    out.append("\n## 待裁决格子")
    for c in cells:
        out.append(f"  - [{c['position']}] {c['author']}  ×  {ename.get(c['entity_id'], '?')} ({c['entity_id']})")

    txt = "\n".join(out)
    p = BASE / f"evidence_{suffix}.md"
    p.write_text(txt, encoding="utf-8")
    print(p, len(txt), "chars,", len(cells), "cells,", len(ent_ids), "entities")

if __name__ == "__main__":
    main(sys.argv[1])
