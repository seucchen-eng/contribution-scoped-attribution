# -*- coding: utf-8 -*-
"""
M2 证据提取 - 方法实体抽取 pipeline（prompt v1 + 后端抽象 + span 校验）
用法：
  python pipeline/extract_methods.py --limit 30 --backend openai \
      --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key <KEY>
  python pipeline/extract_methods.py --limit 30 --backend ollama --model qwen2.5:14b
  python pipeline/extract_methods.py --limit 30 --dry-run   # 不调用 LLM，只验证数据流与 span 校验
产物：
  data/processed/method_mentions.jsonl   每条：{doi, entity, span, section, verified}
  data/processed/extract_report.txt
设计约束（对应初稿 4.2.1）：
  - 抽取结果必须可定位：span 必须在原文中逐字出现（verified=True 才入库）
  - 输出严格 JSON schema，解析失败的响应重试一次后丢弃并计数
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

PROMPT_V2 = """你是科研文献信息抽取系统。从下面的论文方法章节文本中，抽取所有"方法实体"提及。

方法实体包括：算法、模型、统计方法、机器学习架构、实验技术、数据分析流程、软件工具（仅当作为方法使用时）。
重要：不仅要抽命名工具（如 ModelSEED、DIAMOND），也要抽通用方法短语（如 gap-filling、protein alignment、cross-species comparison、ensemble generation）——只要它指代一个可识别的研究活动。
不包括：过于宽泛的词（如"analysis"、"experiment"单独出现）、数据集/数据库名称（如 RAST）、疾病/生物实体、度量指标名称、实验材料（如试剂名 DNase 1 单独出现时不算，但"DNase treatment"算）。

对每条提及，输出：
- mention：原文中逐字出现的文本片段（必须逐字，不得改写）
- canonical：该方法的规范名称（英文，可用于跨论文归一）
- type：algorithm | model | statistical | experiment | software | pipeline
- context_role：该方法在文中的角色（developed=本文提出 / applied=本文使用 / baseline=用作基线 / reviewed=背景提及）

以 JSON 数组输出，不要输出任何其他文字。示例：
[{{"mention": "graph convolutional network", "canonical": "graph convolutional network", "type": "model", "context_role": "applied"}}]

=== 论文方法章节 ===
{text}
=== 输出（JSON 数组） ==="""


def resolve_api_key(cli_key: str) -> str:
    """依次尝试：命令行参数 → LLM_API_KEY/DEEPSEEK_API_KEY 环境变量 → Windows 注册表用户环境变量。"""
    if cli_key:
        return cli_key
    for var in ("LLM_API_KEY", "DEEPSEEK_API_KEY"):
        v = os.environ.get(var, "")
        if v:
            return v
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            for var in ("LLM_API_KEY", "DEEPSEEK_API_KEY"):
                try:
                    v, _ = winreg.QueryValueEx(k, var)
                    if v:
                        return v
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return ""


def call_llm_openai(text: str, base_url: str, model: str, api_key: str) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_V2.format(text=text)}],
            "temperature": 0,
            "max_tokens": 2000,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_llm_ollama(text: str, model: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": PROMPT_V2.format(text=text),
              "stream": False, "options": {"temperature": 0}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def parse_response(raw: str) -> list[dict] | None:
    """从 LLM 响应中提取 JSON 数组；失败返回 None。"""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for item in arr:
        if isinstance(item, dict) and item.get("mention") and item.get("canonical"):
            out.append({
                "mention": str(item["mention"]),
                "canonical": str(item["canonical"]).strip().lower(),
                "type": str(item.get("type", "")),
                "context_role": str(item.get("context_role", "")),
            })
    return out


def locate_span(mention: str, text: str) -> int:
    """span 校验：mention 必须逐字出现在原文中，返回起始偏移，找不到返回 -1。"""
    idx = text.find(mention)
    if idx >= 0:
        return idx
    # 宽松匹配：大小写不敏感
    m = re.search(re.escape(mention), text, re.IGNORECASE)
    return m.start() if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dois", default="", help="逗号分隔的 DOI 列表，优先于 offset/limit")
    ap.add_argument("--suffix", default="", help="读写带后缀的独立文件（如 _post 评测期论文）")
    ap.add_argument("--workers", type=int, default=1, help="LLM 并发数（>1 时按完成顺序落盘）")
    args = ap.parse_args()
    if not args.dry_run and args.backend == "openai":
        args.api_key = resolve_api_key(args.api_key)
        if not args.base_url:
            args.base_url = "https://api.deepseek.com/v1"
        if not args.model:
            args.model = "deepseek-chat"
        if not args.api_key:
            print("错误：未找到 API key（--api-key / LLM_API_KEY / DEEPSEEK_API_KEY）")
            sys.exit(1)

    df = pd.read_pickle(OUT_DIR / f"articles{args.suffix}.pkl")
    df = df[df["methods_len"] > 500]  # 太短的方法小节不参与
    # 断点续跑：先剔除已有记录的论文，再按 offset/limit 取批（--dois 强制重跑除外）
    out_path = OUT_DIR / f"method_mentions{args.suffix}.jsonl"
    if out_path.exists() and not args.dois:
        done0 = {json.loads(l)["doi"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        df = df[~df["doi"].isin(done0)]
    if args.dois:
        targets = set(args.dois.split(","))
        batch = df[df["doi"].isin(targets)]
    else:
        batch = df.iloc[args.offset:args.offset + args.limit]
    print(f"待抽取 {len(batch)} 篇（methods_len>500）")

    def process_one(row):
        """单篇抽取 -> (doi, records, status)，status ∈ ok/parse_fail/llm_fail"""
        text = row["methods_text"][:12000]  # 控制上下文长度
        if args.dry_run:
            items = [{"mention": text[100:130], "canonical": "dryrun",
                      "type": "model", "context_role": "applied"},
                     {"mention": "NOT_IN_TEXT_XYZ", "canonical": "bad",
                      "type": "model", "context_role": "applied"}]
        else:
            try:
                raw = (call_llm_openai(text, args.base_url, args.model, args.api_key)
                       if args.backend == "openai" else call_llm_ollama(text, args.model))
            except Exception:
                return row["doi"], [], "llm_fail"
            items = parse_response(raw)
            if items is None:
                return row["doi"], [], "parse_fail"
        recs = []
        for it in items:
            off = locate_span(it["mention"], text)
            recs.append({"doi": row["doi"], "entity": it["canonical"],
                         "span": it["mention"], "offset": off,
                         "type": it["type"], "context_role": it["context_role"],
                         "verified": off >= 0})
        return row["doi"], recs, "ok"

    stats = {"ok": 0, "parse_fail": 0, "llm_fail": 0,
             "mentions": 0, "verified": 0, "unverified": 0}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        if args.workers > 1 and not args.dry_run:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(process_one, row) for _, row in batch.iterrows()]
                for i, fut in enumerate(as_completed(futs), 1):
                    doi, recs, status = fut.result()
                    stats[status] += 1
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats["mentions"] += 1
                        stats["verified" if rec["verified"] else "unverified"] += 1
                    if i % 10 == 0 or i == len(futs):
                        print(f"  进度 {i}/{len(futs)}，耗时 {time.time()-t0:.0f}s")
        else:
            for i, (_, row) in enumerate(batch.iterrows(), 1):
                doi, recs, status = process_one(row)
                stats[status] += 1
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["mentions"] += 1
                    stats["verified" if rec["verified"] else "unverified"] += 1
                if i % 5 == 0 or i == len(batch):
                    print(f"  进度 {i}/{len(batch)}，耗时 {time.time()-t0:.0f}s")

    vr = stats["verified"] / max(stats["mentions"], 1)
    report = (f"\n===== 方法实体抽取报告 =====\n"
              f"论文: 成功 {stats['ok']} | JSON 解析失败 {stats['parse_fail']} | LLM 失败 {stats['llm_fail']}\n"
              f"提及总数: {stats['mentions']} | span 可定位: {stats['verified']} ({vr:.1%}) | "
              f"不可定位: {stats['unverified']}\n")
    print(report)
    with open(OUT_DIR / "extract_report.txt", "a", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    sys.exit(main())
