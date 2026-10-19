# -*- coding: utf-8 -*-
"""
M1 数据底座 - PLOS 论文抓取与解析 pipeline
用法：
  python pipeline/fetch_plos.py --limit 30            # 试跑
  python pipeline/fetch_plos.py --limit 500           # 正式批量
产物：
  data/raw_xml/<doi 后缀>.xml          原始 JATS XML
  data/processed/articles.pkl          论文级信息
  data/processed/authorships.pkl       作者×角色长表（C_apr 的直接来源）
  data/processed/report.txt            统计报告
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_xml"
OUT_DIR = ROOT / "data" / "processed"

SEARCH_API = "http://api.plos.org/search"
XML_TPL = "https://journals.plos.org/{journal_path}/article/file?id={doi}&type=manuscript"

JOURNALS = {  # journal solr 名 -> 下载路径段
    "PLoS Computational Biology": "ploscompbiol",
    "PLoS ONE": "plosone",
}

HEADERS = {"User-Agent": "research-pipeline/0.1 (mailto:research@example.org)"}


def search_dois(journal: str, limit: int, offset: int, date_from: str, date_to: str) -> list[dict]:
    """从 PLOS Solr 检索论文元数据列表。"""
    q = f'journal:"{journal}" AND doc_type:full AND publication_date:[{date_from}T00:00:00Z TO {date_to}T23:59:59Z]'
    docs, start = [], 0
    while len(docs) < offset + limit:
        rows = min(500, offset + limit - len(docs))
        r = requests.get(SEARCH_API, params={
            "q": q, "rows": rows, "start": start,
            "fl": "id,title,publication_date", "wt": "json",
            "sort": "publication_date desc",
        }, headers=HEADERS, timeout=30)
        r.raise_for_status()
        batch = r.json()["response"]["docs"]
        if not batch:
            break
        docs.extend(batch)
        start += len(batch)
    return docs[offset:offset + limit]


def fetch_xml(doi: str, journal_path: str, retries: int = 3) -> bytes | None:
    out = RAW_DIR / (doi.split("/")[-1] + ".xml")
    if out.exists() and out.stat().st_size > 1000:
        return out.read_bytes()
    url = XML_TPL.format(journal_path=journal_path, doi=doi)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 200 and len(r.content) > 1000:
                out.write_bytes(r.content)
                return r.content
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def parse_article(doi: str, xml_bytes: bytes) -> tuple[dict, list[dict]] | None:
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    if not soup.find("article"):
        return None

    title = _text(soup.find("article-title"))
    pub = soup.find("pub-date", attrs={"pub-type": "epub"}) or soup.find("pub-date")
    pub_date = ""
    if pub:
        y, m, d = pub.find("year"), pub.find("month"), pub.find("day")
        pub_date = f"{_text(y)}-{int(_text(m) or 1):02d}-{int(_text(d) or 1):02d}"

    abstract = _text(soup.find("abstract"))

    # 作者与 CRediT 角色（结构化）
    authorships = []
    for pos, c in enumerate(soup.find_all("contrib", attrs={"contrib-type": "author"}), start=1):
        name = c.find("name")
        if not name:
            continue
        surname, given = _text(name.find("surname")), _text(name.find("given-names"))
        roles = []
        for role in c.find_all("role"):
            ct = role.get("content-type", "")
            canonical = ct.rstrip("/").split("/")[-1] if "credit.niso.org" in ct else ""
            roles.append({
                "role_text": _text(role),
                "role_canonical": canonical,
            })
        authorships.append({
            "doi": doi, "position": pos,
            "surname": surname, "given_names": given,
            "roles": json.dumps(roles, ensure_ascii=False),
            "n_roles": len(roles),
        })

    # 叙述型贡献声明（部分论文同时存在）
    narrative = ""
    for s in soup.find_all(["sec", "fn"]):
        t = _text(s.find("title")) if s.name == "sec" else ""
        label = _text(s.find("label")) if s.name == "fn" else ""
        if "author contribution" in (t + " " + label).lower():
            narrative = _text(s)
            break

    # 章节结构与方法小节
    body = soup.find("body")
    sections, methods_text = [], []
    if body:
        for sec in body.find_all("sec", recursive=True):
            t = sec.find("title")
            if t:
                sections.append(_text(t))
        for sec in body.find_all("sec"):
            t = _text(sec.find("title")).lower()
            if re.search(r"\b(methods?|materials and methods?|model|approach)\b", t):
                methods_text.append(_text(sec))
    full_text = _text(body)

    article = {
        "doi": doi, "title": title, "pub_date": pub_date,
        "abstract": abstract,
        "n_authors": len(authorships),
        "n_authors_with_roles": sum(1 for a in authorships if a["n_roles"] > 0),
        "has_narrative_contrib": bool(narrative),
        "contrib_narrative": narrative[:5000],
        "sections": json.dumps(sections, ensure_ascii=False),
        "n_sections": len(sections),
        "methods_text": "\n\n".join(methods_text),
        "methods_len": sum(len(m) for m in methods_text),
        "fulltext_len": len(full_text),
    }
    return article, authorships


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="PLoS Computational Biology", choices=list(JOURNALS))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--date-from", default="2015-01-01")
    ap.add_argument("--date-to", default="2023-12-31", help="t*=2024-01-01，评测期之后的论文不进入训练底座")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀，如 _post 会把结果写入 articles_post.pkl/authorships_post.pkl，"
                         "用于 t* 之后的评测期论文，避免污染训练底座")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 检索 {args.journal} {args.date_from}~{args.date_to}，目标 {args.limit} 篇（offset={args.offset}）")
    docs = search_dois(args.journal, args.limit, args.offset, args.date_from, args.date_to)
    print(f"      检索到 {len(docs)} 篇")

    journal_path = JOURNALS[args.journal]
    articles, authorships, failed = [], [], []

    print(f"[2/3] 下载并解析 XML（workers={args.workers}）")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_xml, d["id"], journal_path): d["id"] for d in docs}
        for i, fut in enumerate(as_completed(futs), 1):
            doi = futs[fut]
            xml = fut.result()
            if xml is None:
                failed.append(doi)
                continue
            parsed = parse_article(doi, xml)
            if parsed is None:
                failed.append(doi)
                continue
            a, au = parsed
            articles.append(a)
            authorships.extend(au)
            if i % 25 == 0 or i == len(docs):
                print(f"      进度 {i}/{len(docs)}，耗时 {time.time()-t0:.0f}s")

    df_a = pd.DataFrame(articles)
    df_au = pd.DataFrame(authorships)

    # 与已有结果合并（支持分批累积）；--suffix 时读写带后缀的独立文件
    pa = OUT_DIR / f"articles{args.suffix}.pkl"
    pu = OUT_DIR / f"authorships{args.suffix}.pkl"
    if pa.exists():
        df_a = pd.concat([pd.read_pickle(pa), df_a]).drop_duplicates("doi", keep="last")
    if pu.exists() and not df_au.empty:
        df_au = pd.concat([pd.read_pickle(pu), df_au]).drop_duplicates(
            ["doi", "position"], keep="last")
    df_a.to_pickle(pa)
    if not df_au.empty:
        df_au.to_pickle(pu)

    print("[3/3] 统计报告")
    n = len(df_a)
    role_cov = (df_a["n_authors_with_roles"] > 0).mean() if n else 0
    full_role = (df_a["n_authors_with_roles"] == df_a["n_authors"]).mean() if n else 0
    report = f"""
===== PLOS 抓取统计报告 =====
期刊: {args.journal} | 时间窗: {args.date_from} ~ {args.date_to}
本次目标: {args.limit} 篇 (offset={args.offset})，失败 {len(failed)} 篇
库内累计论文数: {n}
库内累计作者-论文记录: {len(df_au)}

--- 完整性指标 ---
结构化 CRediT 角色覆盖率（至少 1 位作者有角色）: {role_cov:.1%}
全体作者均有角色的论文占比: {full_role:.1%}
含叙述型贡献声明的论文占比: {df_a['has_narrative_contrib'].mean():.1%}
有 Methods 类小节的论文占比: {(df_a['methods_len'] > 0).mean():.1%}
全文长度中位数: {int(df_a['fulltext_len'].median())} 字符
Methods 长度中位数: {int(df_a['methods_len'].median())} 字符
平均作者数/篇: {df_a['n_authors'].mean():.1f}
"""
    print(report)
    with open(OUT_DIR / "report.txt", "a", encoding="utf-8") as f:
        f.write(report)
    if failed:
        print("失败 DOI:", failed[:10], "..." if len(failed) > 10 else "")


if __name__ == "__main__":
    sys.exit(main())
