# -*- coding: utf-8 -*-
"""
L1 层数据抓取：eldenring.wiki.gg → l1_wiki.jsonl

产出：每个 chunk 一行 JSON
    {
      "id": "wiki:malenia_blade_of_miquella#003",
      "title": "Malenia, Blade of Miquella",
      "title_zh": "「米凯拉的锋刃」玛莲妮亚",   # 来自官方术语表，可能为空
      "category": "Bosses",
      "section": "Phase 2 - Malenia, Goddess of Rot",
      "url": "https://eldenring.wiki.gg/wiki/Malenia,_Blade_of_Miquella",
      "text": "Once her health reaches 0, a cutscene will play...",
      "lang": "en",
      "chars": 412
    }

用法:
    python scripts/fetch_wiki.py                          # 抓全部默认分类
    python scripts/fetch_wiki.py --cats Bosses            # 只抓 Boss
    python scripts/fetch_wiki.py --limit 20               # 每分类只抓 20 页（测试用）
    python scripts/fetch_wiki.py --dry-run                # 只看抓到什么，不落盘

设计要点:
    - wiki.gg 有 UA 检查与限流：必须用完整浏览器 UA + 请求间隔，短 UA 直接返回 Blocked 页面
    - 断点续传：已抓过的 title 记录在 .cache/fetched_titles.txt，中断后重跑自动跳过
    - 官方中文名：从 data/elden_ring/glossary_flat.json 查表补全（查不到则留空）
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "elden_ring" / "l1_wiki.jsonl"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
TITLES_CACHE = CACHE_DIR / "fetched_titles.txt"
GLOSSARY = ROOT / "data" / "elden_ring" / "glossary_flat.json"

API = "https://eldenring.wiki.gg/api.php"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# 抓取优先级：Boss/NPC/地点是攻略问答的主战场，武器护符量太大且对"怎么打"帮助小
DEFAULT_CATS = ["Bosses", "Characters", "Locations", "Ashes_of_War", "Talismans"]

MIN_CHARS = 80      # 短于此的 chunk 丢弃（多半是模板残渣）
TARGET_CHARS = 500  # chunk 目标长度
MAX_CHARS = 900     # 单个 chunk 上限


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 网络

def api_get(params, retries=4):
    """带退避重试的 GET。返回 dict，失败抛异常。"""
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = f"{API}?{urllib.parse.urlencode(params)}"

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            if raw.lstrip().startswith("<"):   # 被挡：返回的是 HTML
                raise RuntimeError("被限流(Blocked)")
            return json.loads(raw)
        except Exception as e:
            last = e
            wait = (2 ** attempt) + random.uniform(0, 1)
            if attempt < retries - 1:
                time.sleep(wait)
    raise RuntimeError(f"请求失败 {retries} 次: {last}")


def fetch_category(cat, limit=None):
    """取某分类下所有页面标题"""
    titles, cont = [], {}
    while True:
        p = {"action": "query", "list": "categorymembers",
             "cmtitle": f"Category:{cat}", "cmlimit": 500, "cmtype": "page"}
        p.update(cont)
        d = api_get(p)
        for m in d.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        if limit and len(titles) >= limit:
            return titles[:limit]
        if "continue" in d:
            cont = d["continue"]
            time.sleep(1.2)
        else:
            return titles
    return titles


def fetch_pages(titles):
    """批量取 wikitext。MediaWiki 限制：rvprop=content 单次最多 50 页"""
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        d = api_get({
            "action": "query",
            "titles": "|".join(batch),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
        })
        for p in d.get("query", {}).get("pages", []):
            if p.get("missing"):
                continue
            try:
                out[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
            except (KeyError, IndexError):
                pass
        time.sleep(1.2)
    return out


# ---------------------------------------------------------------- 清洗

def strip_templates(w):
    """剥离 {{...}}，支持嵌套。Wiki 模板语法的主要噪音来源"""
    out, depth = [], 0
    for ch in w:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def clean_wikitext(w):
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)        # 注释
    w = re.sub(r"<ref[^>]*/>", "", w)                    # 自闭合引用
    w = re.sub(r"<ref.*?</ref>", "", w, flags=re.S)      # 引用块
    w = re.sub(r"<[^>]+>", "", w)                        # 残余 HTML
    w = strip_templates(w)                               # 模板
    w = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", w)   # 链接取显示名
    w = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", w)      # 外链
    w = re.sub(r"\[https?://\S+\]", "", w)
    w = re.sub(r"'{2,}", "", w)                          # 粗斜体标记
    w = re.sub(r"^[=]{1,6}\s*(.*?)\s*[=]{1,6}$", r"\n## \1\n", w, flags=re.M)  # 小标题
    w = re.sub(r"[ \t]+", " ", w)
    w = re.sub(r"\n{3,}", "\n\n", w)
    w = re.sub(r"\n +", "\n", w)
    return w.strip()


def is_index_page(text):
    """
    判定"分类索引页"——内容是一长串 '* 条目名' 的目录页。
    这类页面对问答毫无价值（只有名字没有信息），必须丢弃而非切分。
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    bullets = sum(1 for l in lines if l.strip().startswith("*"))
    return bullets / len(lines) > 0.5


def split_chunks(text):
    """按 ## 小节切，超长段落再按句切"""
    chunks, cur_sec = [], ""
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            first, _, rest = block.partition("\n")
            cur_sec = first[3:].strip()
            block = rest.strip()
        if not block:
            continue
        if len(block) <= MAX_CHARS:
            chunks.append((cur_sec, block))
            continue
        # 超长：按句累积
        buf = ""
        for sent in re.split(r"(?<=[.!?。！？])\s+", block):
            if buf and len(buf) + len(sent) > TARGET_CHARS:
                chunks.append((cur_sec, buf.strip()))
                buf = sent
            else:
                buf = f"{buf} {sent}" if buf else sent
        if buf.strip():
            chunks.append((cur_sec, buf.strip()))

    # 兜底：仍超上限的（无标点的表格/长串），按长度硬切
    final = []
    for sec, c in chunks:
        while len(c) > MAX_CHARS:
            cut = c.rfind(" ", 0, MAX_CHARS)
            if cut < MAX_CHARS // 2:      # 找不到合适空格就强制断
                cut = MAX_CHARS
            final.append((sec, c[:cut].strip()))
            c = c[cut:].strip()
        if c:
            final.append((sec, c))
    return [(s, c) for s, c in final if len(c) >= MIN_CHARS]


# ---------------------------------------------------------------- 中文名

def load_glossary():
    if not GLOSSARY.exists():
        log("  ⚠️ 未找到术语表，跳过中文名补全")
        return {}
    with open(GLOSSARY, encoding="utf-8") as f:
        return json.load(f)


def to_zh(title, gloss):
    """精确匹配 → 去掉后缀修饰（'X, Blade of Y' → 'X'）→ 逐词匹配"""
    key = title.lower().strip()
    if key in gloss:
        return gloss[key]
    base = re.split(r"[,（(]", title)[0].strip().lower()
    if base in gloss:
        return gloss[base]
    for tok in re.findall(r"[A-Za-z][A-Za-z'’\-]{3,}", title):
        k2 = tok.lower()
        if k2 in gloss:
            return gloss[k2]
    return ""


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=DEFAULT_CATS)
    ap.add_argument("--limit", type=int, default=None, help="每个分类最多抓几页（测试用）")
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="追加到已有 jsonl（按 id 去重），而不是覆盖")
    ap.add_argument("--delay", type=float, default=1.2, help="请求间隔秒")
    args = ap.parse_args()

    log("=" * 60)
    log("L1 数据抓取 · eldenring.wiki.gg")
    log("=" * 60)

    gloss = load_glossary()
    log(f"  术语表: {len(gloss):,} 条")

    done = set()
    if TITLES_CACHE.exists():
        done = {l.strip() for l in
                TITLES_CACHE.read_text(encoding="utf-8").splitlines() if l.strip()}
    log(f"  断点缓存: 已抓 {len(done)} 页")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    stats = {}

    for cat in args.cats:
        log(f"\n── 分类 {cat} ──")
        try:
            titles = fetch_category(cat, args.limit)
        except Exception as e:
            log(f"  取列表失败: {e}")
            continue
        todo = [t for t in titles if f"{cat}::{t}" not in done]
        log(f"  共 {len(titles)} 页，待抓 {len(todo)} 页")
        if not todo:
            continue

        n_chunk = 0
        skipped = 0
        for i in range(0, len(todo), 20):
            batch = todo[i:i + 20]
            try:
                pages = fetch_pages(batch)
            except Exception as e:
                log(f"  批次失败，跳过: {e}")
                continue
            for title, raw in pages.items():
                text = clean_wikitext(raw)
                if len(text) < MIN_CHARS:
                    continue
                if is_index_page(text):
                    skipped += 1
                    continue
                url = "https://eldenring.wiki.gg/wiki/" + urllib.parse.quote(
                    title.replace(" ", "_"))
                zh = to_zh(title, gloss)
                for idx, (sec, chunk) in enumerate(split_chunks(text)):
                    results.append({
                        "id": f"wiki:{re.sub(r'[^a-z0-9]+', '_', title.lower())}#{idx:03d}",
                        "title": title,
                        "title_zh": zh,
                        "category": cat,
                        "section": sec,
                        "url": url,
                        "text": chunk,
                        "lang": "en",
                        "chars": len(chunk),
                    })
                    n_chunk += 1
            with open(TITLES_CACHE, "a", encoding="utf-8") as f:
                for t in batch:
                    f.write(f"{cat}::{t}\n")
            done.update(f"{cat}::{t}" for t in batch)
            log(f"    进度 {min(i + 20, len(todo))}/{len(todo)}  累计 {n_chunk} chunks")
            time.sleep(args.delay)

        stats[cat] = n_chunk
        log(f"  ✅ {cat}: {n_chunk} chunks"
            + (f"（丢弃 {skipped} 个索引页）" if skipped else ""))

    log("")
    log("=" * 60)
    total = len(results)
    with_zh = sum(1 for r in results if r["title_zh"])
    log(f"合计 {total:,} chunks，{with_zh:,} 条带官方中文名 "
        f"({with_zh * 100 // max(total, 1)}%)")
    for c, n in stats.items():
        log(f"  {c:<20} {n:>6}")
    log("=" * 60)

    if args.dry_run:
        log("\n[dry-run] 不写文件。前 2 条预览：")
        for r in results[:2]:
            log(f"\n  id={r['id']}\n  {r['title']} / {r['title_zh']}\n"
                f"  [{r['section']}] {r['text'][:200]}...")
        return 0

    if total == 0:
        log("无数据，不覆盖已有文件")
        return 1

    rows = results
    if args.merge and out_path.exists():
        old = [json.loads(l) for l in
               out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        seen = {r["id"] for r in old}
        added = [r for r in results if r["id"] not in seen]
        rows = old + added
        log(f"\n  merge: 已有 {len(old):,} 行，新增 {len(added):,} 行，"
            f"去重跳过 {len(results) - len(added):,} 行")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"✅ 已写入 {out_path}  ({len(rows):,} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
