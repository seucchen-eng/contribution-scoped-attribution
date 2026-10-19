# -*- coding: utf-8 -*-
"""
M3 标注 - 任务构建与标注服务器（Python 标准库，无额外依赖）
用法：
  python pipeline/annotation_server.py --build --n-papers 50   # 生成标注任务
  python pipeline/annotation_server.py --serve --port 8765     # 启动标注界面
标注界面设计（对应实施计划 M3 / 初稿 5.2 节）：
  - 不展示任何模型输出：只展示论文 Methods 上下文 + 作者声明的 CRediT 角色
  - 两步标注：先判定 作者×方法 是否参与（是/否/不确定），是则勾选角色
  - 角色候选只列该作者在该论文中声明过的角色（即待消歧的粗标签）
  - 保存为 annotations.jsonl，含标注员标识与时间戳
"""
import argparse
import json
import time
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
TASKS_PATH = OUT_DIR / "annotation_tasks.json"
ANNO_PATH = OUT_DIR / "annotations.jsonl"

MAX_ENTITIES_PER_PAPER = 8
TYPE_PRIORITY = {"model": 0, "algorithm": 1, "statistical": 2, "pipeline": 3,
                 "experiment": 4, "software": 5}
ROLE_PREFERRED = {"methodology", "software", "formal-analysis", "data-curation",
                  "investigation", "validation", "visualization"}


def build_tasks(n_papers: int, dois_file: str = None, out_path=None):
    df_a = pd.read_pickle(OUT_DIR / "articles.pkl")
    df_au = pd.read_pickle(OUT_DIR / "authorships.pkl")
    df_e = pd.read_pickle(OUT_DIR / "entities.pkl")
    e2name = dict(zip(df_e["entity_id"], df_e["canonical"]))

    mentions = [json.loads(l) for l in
                (OUT_DIR / "method_mentions.jsonl").read_text(encoding="utf-8").splitlines()]
    m2e = pd.read_pickle(OUT_DIR / "mention2entity.pkl").to_dict()

    # 论文 -> 提及列表（含实体链接结果）
    by_doi = defaultdict(list)
    for r in mentions:
        eid = m2e.get(f"{r['doi']}|{r['span'].lower()}")
        if eid and r["verified"]:
            by_doi[r["doi"]].append({**r, "entity_id": eid})

    cand = df_a[(df_a["n_authors"] >= 3) & (df_a["methods_len"] > 3000)]
    cand = cand[cand["doi"].isin(by_doi)]
    if dois_file:  # 指定 DOI 清单（正式批）
        sel = json.loads(Path(dois_file).read_text(encoding="utf-8"))
        cand = cand[cand["doi"].isin(sel)]
        order = {d: i for i, d in enumerate(sel)}
        cand = cand.assign(_o=cand["doi"].map(order)).sort_values("_o")
    else:
        cand = cand.sample(frac=1, random_state=42)
    tasks = []
    for _, row in cand.head(n_papers).iterrows():
        doi = row["doi"]
        # 实体排序：developed/applied 优先，类型优先级，频次
        ment = [m for m in by_doi[doi] if m["context_role"] in ("developed", "applied")]
        ment.sort(key=lambda m: (TYPE_PRIORITY.get(m["type"], 9),
                                 0 if m["context_role"] == "developed" else 1))
        seen_e, ents = set(), []
        for m in ment:
            if m["entity_id"] in seen_e:
                continue
            seen_e.add(m["entity_id"])
            ents.append({
                "entity_id": m["entity_id"],
                "name": e2name.get(m["entity_id"], m["entity"]),
                "span": m["span"],
                "context": row["methods_text"][max(0, m["offset"] - 180):m["offset"] + len(m["span"]) + 180],
                "type": m["type"], "role_ctx": m["context_role"],
            })
            if len(ents) >= MAX_ENTITIES_PER_PAPER:
                break
        if len(ents) < 2:
            continue
        # 作者与声明角色
        authors = []
        for _, a in df_au[df_au["doi"] == doi].sort_values("position").iterrows():
            roles = [x.get("role_canonical") for x in json.loads(a["roles"]) if x.get("role_canonical")]
            authors.append({
                "position": int(a["position"]),
                "name": f"{a['given_names']} {a['surname']}",
                "declared_roles": roles,
                "preferred_roles": [r for r in roles if r in ROLE_PREFERRED],
            })
        tasks.append({
            "task_id": doi.split("/")[-1],
            "doi": doi, "title": row["title"], "pub_date": row["pub_date"],
            "abstract": row["abstract"][:600],
            "methods_text": row["methods_text"][:9000],
            "entities": ents, "authors": authors,
        })
    out = Path(out_path) if out_path else TASKS_PATH
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cells = sum(len(t["entities"]) * len(t["authors"]) for t in tasks)
    print(f"已生成 {len(tasks)} 篇标注任务，判定单元共 {n_cells} 个（平均每篇 {n_cells/max(len(tasks),1):.0f}）")
    print(f"已写入 {out}")


PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>归属标注 - {title}</title><style>
body{{font-family:'Segoe UI',sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ccc;padding:6px 8px;font-size:13px;vertical-align:top}}
th{{background:#f4f6f8}}
.cell-yes{{background:#d8f0d8}} .cell-no{{background:#f6dada}} .cell-maybe{{background:#fdf3cd}}
button.tg{{margin:1px;padding:3px 8px;font-size:12px;cursor:pointer}}
.roles label{{margin-right:8px;font-size:12px;white-space:nowrap}}
details{{margin:10px 0}} .ctx{{background:#fafafa;padding:8px;font-size:13px;line-height:1.6}}
mark{{background:#ffe58a}} .head{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #333}}
.bar{{margin:14px 0}} .save{{padding:10px 24px;font-size:15px;background:#1a73e8;color:#fff;border:0;border-radius:4px;cursor:pointer}}
</style></head><body>
<div class="head"><b>{title}</b><br><span style="font-size:12px;color:#666">{doi} · {pub_date} · 标注员：{who}</span></div>
<details><summary><b>论文摘要与 Methods 全文</b>（点击展开）</summary>
<p class="ctx">{abstract}</p><div class="ctx" style="white-space:pre-wrap">{methods}</div></details>
<details open><summary><b>方法实体上下文</b>（原文定位证据）</summary>{entity_ctx}</details>
<h3>判定矩阵：作者是否参与了该方法的工作？</h3>
<p style="font-size:12px">点击单元格切换：<b>是</b>（有文本证据支持）→ <b>否</b>（证据指向其他人/无证据）→ <b>不确定</b>。
标“是”后，在该行勾选具体角色（候选来自该作者的声明角色）。</p>
{matrix}
<div class="bar"><button class="save" onclick="save()">保存本篇标注</button>
<span id="msg" style="margin-left:12px;color:green"></span></div>
<script>
const DATA = {data_json};
function tg(pos, eid, btn, val) {{
  const cell = document.getElementById(`c-${{pos}}-${{eid}}`);
  cell.className = val===1?'cell-yes':(val===0?'cell-no':'cell-maybe');
  cell.dataset.v = val;
  const rl = document.getElementById(`r-${{pos}}-${{eid}}`);
  if (rl) rl.style.display = val===1?'block':'none';
}}
function save() {{
  const items = [];
  document.querySelectorAll('td[data-v]').forEach(td => {{
    const v = parseInt(td.dataset.v);
    if (v === -1) return;
    const [pos, eid] = td.id.slice(2).split('-');
    const roles = [];
    document.querySelectorAll(`#r-${{pos}}-${{eid}} input:checked`).forEach(cb => roles.push(cb.value));
    items.push({{position: parseInt(pos), entity_id: eid, z: v, roles: roles}});
  }});
  fetch('/save', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{task_id: DATA.task_id, doi: DATA.doi, annotator: DATA.who, items: items}})}})
  .then(r => r.text()).then(t => {{document.getElementById('msg').textContent = '✓ 已保存 ' + items.length + ' 条';}});
}}
</script></body></html>"""


def render_paper(task: dict, who: str) -> str:
    # 实体上下文（高亮 span）
    ctxs = []
    for e in task["entities"]:
        ctx = e["context"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        sp = e["span"].replace("&", "&amp;").replace("<", "&lt;")
        ctx = ctx.replace(sp, f"<mark>{sp}</mark>", 1)
        ctxs.append(f"<p class='ctx'><b>[{e['type']}] {e['name']}</b><br>…{ctx}…</p>")
    # 矩阵
    head = "<tr><th>作者（声明角色）</th>" + "".join(
        f"<th>{e['name']}</th>" for e in task["entities"]) + "</tr>"
    rows = []
    for a in task["authors"]:
        cells = []
        for e in task["entities"]:
            rl = "".join(
                f"<label><input type='checkbox' value='{r}'>{r}</label>"
                for r in (a["preferred_roles"] or a["declared_roles"]))
            cells.append(
                f"<td id='c-{a['position']}-{e['entity_id']}' data-v='-1'>"
                f"<button class='tg' onclick=\"tg({a['position']},'{e['entity_id']}',this,1)\">是</button>"
                f"<button class='tg' onclick=\"tg({a['position']},'{e['entity_id']}',this,0)\">否</button>"
                f"<button class='tg' onclick=\"tg({a['position']},'{e['entity_id']}',this,2)\">?</button>"
                f"<div class='roles' id='r-{a['position']}-{e['entity_id']}' style='display:none'>{rl}</div></td>")
        roles_txt = ", ".join(a["declared_roles"])
        rows.append(f"<tr><td><b>{a['name']}</b><br><span style='font-size:11px;color:#666'>{roles_txt}</span></td>"
                    + "".join(cells) + "</tr>")
    matrix = f"<table>{head}{''.join(rows)}</table>"
    data = {"task_id": task["task_id"], "doi": task["doi"], "who": who}
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return PAGE.format(title=esc(task["title"]), doi=task["doi"], pub_date=task["pub_date"],
                       who=who, abstract=esc(task["abstract"]),
                       methods=esc(task["methods_text"]),
                       entity_ctx="".join(ctxs), matrix=matrix,
                       data_json=json.dumps(data))


class Handler(BaseHTTPRequestHandler):
    tasks = []
    anno_done = defaultdict(set)

    def log_message(self, *a):
        pass

    def _send(self, body: str, ctype="text/html; charset=utf-8"):
        b = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        who = q.get("who", ["A"])[0]
        if u.path == "/":
            lis = []
            for t in self.tasks:
                n_done = len(self.anno_done[t["task_id"]])
                lis.append(f"<li><a href='/paper/{t['task_id']}?who={who}'>{t['title'][:90]}</a> "
                           f"<span style='color:#666;font-size:12px'>{t['pub_date']} · "
                           f"{len(t['authors'])}作者×{len(t['entities'])}方法 · 已有 {n_done} 人标注</span></li>")
            self._send(f"<html><head><meta charset='utf-8'><title>标注任务列表</title></head><body>"
                       f"<h2>归属标注任务（当前标注员：{who}）</h2>"
                       f"<p>切换标注员：<a href='/?who=A'>A</a> | <a href='/?who=B'>B</a></p><ul>{''.join(lis)}</ul></body></html>")
        elif u.path.startswith("/paper/"):
            tid = u.path.split("/")[-1]
            task = next((t for t in self.tasks if t["task_id"] == tid), None)
            if task:
                self._send(render_paper(task, who))
            else:
                self._send("任务不存在")
        else:
            self._send("404")

    def do_POST(self):
        if urlparse(self.path).path == "/save":
            body = self.rfile.read(int(self.headers["Content-Length"]))
            rec = json.loads(body)
            rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(ANNO_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.anno_done[rec["task_id"]].add(rec["annotator"])
            self._send("ok", "text/plain")


def serve(port: int, tasks_path: str = None):
    Handler.tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8")) if tasks_path else json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if ANNO_PATH.exists():
        for l in ANNO_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                Handler.anno_done[r["task_id"]].add(r.get("annotator", "A"))
    print(f"标注服务器已启动：http://localhost:{port}/  （Ctrl+C 停止）")
    print(f"任务 {len(Handler.tasks)} 篇；已完成标注：{sum(len(v) for v in Handler.anno_done.values())} 人次")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--n-papers", type=int, default=50)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dois-file", default=None)
    ap.add_argument("--tasks", default=None, help="serve 时使用的任务文件")
    ap.add_argument("--out", default=None, help="build 时的输出路径")
    args = ap.parse_args()
    if args.build:
        build_tasks(args.n_papers, args.dois_file, args.out)
    if args.serve:
        serve(args.port, args.tasks)
