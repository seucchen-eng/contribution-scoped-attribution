# -*- coding: utf-8 -*-
"""
M1 数据底座 - OpenAlex 作者匹配与发表历史抓取
用法：
  python pipeline/fetch_openalex.py --stage resolve            # 阶段1：按 DOI 匹配作者 OpenAlex ID
  python pipeline/fetch_openalex.py --stage history --limit 100  # 阶段2：拉作者历史（可分批）
产物：
  data/processed/authorships.pkl     增加 openalex_id 列
  data/processed/author_history.pkl  作者×历史论文长表
  data/processed/report_openalex.txt
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

OA_WORKS = "https://api.openalex.org/works"
MAILTO = "research@example.org"  # OpenAlex polite pool
HEADERS = {"User-Agent": f"research-pipeline/0.1 (mailto:{MAILTO})"}
API_KEY = ""  # 由 --api-key 或环境变量 OPENALEX_API_KEY 注入

INV_INDEX_MAX = 3000  # 摘要倒排索引重建时的长度上限，防止超长文献撑爆


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def resolve_one(doi: str, plos_authors: list[tuple[int, str, str]]) -> list[dict] | None:
    """按 DOI 拉 OpenAlex work，把 PLOS 作者（按署名顺序+姓名）匹配到 openalex_id。"""
    for attempt in range(3):
        try:
            r = requests.get(f"{OA_WORKS}/doi:{doi}", params={"mailto": MAILTO, "api_key": API_KEY},
                             headers=HEADERS, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code == 404:
                return []
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    else:
        return None
    work = r.json()
    oa_auths = work.get("authorships", [])
    out = []
    for pos, surname, given in plos_authors:
        target = norm(surname) + "|" + norm(given)
        oa_id, oa_name, matched = None, None, False
        # 首选：同位置作者；备选：全体作者中姓名匹配
        cands = [oa_auths[pos - 1]] if 0 < pos <= len(oa_auths) else []
        cands += oa_auths
        for a in cands:
            au = a.get("author") or {}
            raw = au.get("display_name") or ""
            parts = raw.rsplit(" ", 1)
            cand = (norm(parts[-1]) + "|" + norm(parts[0])) if len(parts) == 2 else norm(raw)
            if cand == target or (norm(surname) and norm(surname) in norm(raw)
                                  and norm(given)[:1] and norm(given)[0] == norm(raw)[0:1]):
                oa_id, oa_name, matched = au.get("id"), raw, True
                break
        out.append({"doi": doi, "position": pos, "openalex_id": oa_id,
                    "openalex_name": oa_name, "matched": matched})
    return out


def stage_resolve(workers: int, suffix: str = ""):
    df_a = pd.read_pickle(OUT_DIR / f"articles{suffix}.pkl")
    df_au = pd.read_pickle(OUT_DIR / f"authorships{suffix}.pkl")
    need = df_au[~df_au.get("openalex_id", pd.Series([None] * len(df_au))).notna().values] \
        if "openalex_id" in df_au.columns else df_au
    dois = need["doi"].unique().tolist()
    print(f"[resolve] 待匹配论文 {len(dois)} 篇")
    t0, results, failed = time.time(), [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for doi in dois:
            sub = df_au[df_au["doi"] == doi]
            plist = list(zip(sub["position"], sub["surname"], sub["given_names"]))
            futs[ex.submit(resolve_one, doi, plist)] = doi
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is None:
                failed.append(futs[fut])
            else:
                results.extend(res)
            if i % 100 == 0 or i == len(dois):
                print(f"      进度 {i}/{len(dois)}，耗时 {time.time()-t0:.0f}s")
    df_m = pd.DataFrame(results)
    df_au = df_au.drop(columns=["openalex_id", "openalex_name", "matched"], errors="ignore") \
        .merge(df_m, on=["doi", "position"], how="left")
    df_au.to_pickle(OUT_DIR / f"authorships{suffix}.pkl")
    n, nm = len(df_au), df_au["matched"].sum()
    msg = f"[resolve] 作者-论文记录 {n} 条，成功匹配 OpenAlex 作者 {nm} 条（{nm/n:.1%}），失败论文 {len(failed)} 篇"
    print(msg)
    with open(OUT_DIR / "report_openalex.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pairs = [(pos, w) for w, poses in inv.items() for pos in poses]
    pairs.sort()
    words = [w for _, w in pairs[:INV_INDEX_MAX]]
    return " ".join(words)


def fetch_history_one(author_id: str) -> list[dict] | None:
    """拉取某作者 2024 之前的全部论文元数据（含摘要），最多 5 页×200 篇。"""
    rows, cursor = [], "*"
    for page in range(3):  # 每位作者最多 3 页×200 篇历史，足够构建 π_ak
        ok = False
        for attempt in range(6):
            try:
                r = requests.get(OA_WORKS, params={
                    "filter": f"author.id:{author_id},publication_year:2005-2023",
                    "per-page": 200, "cursor": cursor,
                    "select": "id,doi,title,publication_date,abstract_inverted_index,type",
                    "mailto": MAILTO, "api_key": API_KEY,
                }, headers=HEADERS, timeout=40)
                if r.status_code == 200:
                    ok = True
                    break
                if r.status_code == 429:  # 限流：尊重 Retry-After，指数退避
                    wait = float(r.headers.get("Retry-After", 0)) or 5 * (attempt + 1)
                    time.sleep(min(wait, 30))
                    continue
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        if not ok:
            return None
        data = r.json()
        for w in data.get("results", []):
            rows.append({
                "openalex_id": author_id,
                "work_id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("title"),
                "pub_date": w.get("publication_date"),
                "type": w.get("type"),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            })
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or not data.get("results"):
            break
    return rows


def stage_history(limit: int, offset: int, workers: int):
    ids = set()
    for name in ("authorships.pkl", "authorships_post.pkl"):
        p = OUT_DIR / name
        if p.exists():
            df_au = pd.read_pickle(p)
            ids.update(df_au.loc[df_au["matched"] == True, "openalex_id"].dropna().tolist())
    ids = sorted(ids)
    hp = OUT_DIR / "author_history.pkl"
    done = set(pd.read_pickle(hp)["openalex_id"].unique()) if hp.exists() else set()
    todo = [i for i in ids if i not in done][offset:offset + limit]
    print(f"[history] 唯一作者 {len(ids)}，已完成 {len(done)}，本批 {len(todo)}")
    t0, all_rows, failed = time.time(), [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_history_one, aid): aid for aid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is None:
                failed += 1
            else:
                all_rows.extend(res)
            if i % 100 == 0 or i == len(todo):
                print(f"      进度 {i}/{len(todo)}，耗时 {time.time()-t0:.0f}s")
            if all_rows and i % 50 == 0:  # 每 50 位作者增量落盘，防止超时丢数据
                df_inc = pd.DataFrame(all_rows)
                if hp.exists():
                    df_inc = pd.concat([pd.read_pickle(hp), df_inc]).drop_duplicates(
                        ["openalex_id", "work_id"], keep="last")
                df_inc.to_pickle(hp)
                all_rows.clear()
    if all_rows:
        df_h = pd.DataFrame(all_rows)
        if hp.exists():
            df_h = pd.concat([pd.read_pickle(hp), df_h]).drop_duplicates(
                ["openalex_id", "work_id"], keep="last")
        df_h.to_pickle(hp)
        msg = (f"[history] 本批作者 {len(todo)}，失败 {failed}；"
               f"库内作者 {df_h['openalex_id'].nunique()}，历史论文记录 {len(df_h)} 条，"
               f"含摘要占比 {df_h['abstract'].ne('').mean():.1%}")
    else:
        msg = f"[history] 本批无新数据，失败 {failed}"
    print(msg)
    with open(OUT_DIR / "report_openalex.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["resolve", "history"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--suffix", default="", help="读写带后缀的独立文件（如 _post 评测期论文）")
    args = ap.parse_args()
    global API_KEY
    import os
    API_KEY = args.api_key or os.environ.get("OPENALEX_API_KEY", "")
    if args.stage == "resolve":
        stage_resolve(args.workers, args.suffix)
    else:
        stage_history(args.limit, args.offset, args.workers)


if __name__ == "__main__":
    sys.exit(main())
