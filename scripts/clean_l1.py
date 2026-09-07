# -*- coding: utf-8 -*-
"""
L1 数据后处理：清洗已抓好的 jsonl，不必重新抓取。

做三件事：
    1. 丢弃索引页（一长串 '* 名字' 的目录页，只有名字没有信息）
    2. 二次切分超长 chunk（超过 MAX_CHARS 的硬切）
    3. 丢弃过短残片

用法:
    python scripts/clean_l1.py                      # 原地整理 l1_wiki.jsonl
    python scripts/clean_l1.py --dry-run             # 只看会变什么
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wiki import MAX_CHARS, MIN_CHARS, is_index_page, split_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data" / "elden_ring" / "l1_wiki.jsonl"


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        log(f"❌ 未找到 {src}")
        return 1

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    log("=" * 60)
    log(f"L1 后处理 · {src.name}")
    log("=" * 60)
    log(f"\n输入 {len(rows):,} 行")

    out, n_index, n_split, n_short = [], 0, 0, 0
    for r in rows:
        text = r["text"]
        if is_index_page(text):
            n_index += 1
            continue
        if len(text) <= MAX_CHARS:
            if len(text) >= MIN_CHARS:
                out.append(r)
            else:
                n_short += 1
            continue
        # 超长：重新切分，保留原 section
        sec = r.get("section", "")
        parts = split_chunks(text) if not sec else split_chunks(f"## {sec}\n{text}")
        if not parts:
            n_short += 1
            continue
        n_split += 1
        for i, (_s, c) in enumerate(parts):
            nr = dict(r)
            nr["id"] = f"{r['id'].split('#')[0]}#{i:03d}"
            nr["text"] = c
            nr["chars"] = len(c)
            out.append(nr)

    # id 去重（二次切分可能产生重复）
    seen, dedup = set(), []
    for r in out:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        dedup.append(r)

    log(f"  丢弃索引页     {n_index}")
    log(f"  二次切分       {n_split}")
    log(f"  丢弃过短残片   {n_short}")
    log(f"  id 去重        {len(out) - len(dedup)}")
    log(f"\n输出 {len(dedup):,} 行（净变化 {len(dedup) - len(rows):+,}）")

    if dedup:
        lens = [r["chars"] for r in dedup]
        log(f"chunk 长度: 平均 {sum(lens) // len(lens)}, "
            f"最小 {min(lens)}, 最大 {max(lens)}")
        over = sum(1 for x in lens if x > MAX_CHARS)
        log(f"仍超 {MAX_CHARS} 字符: {over}")

    if args.dry_run:
        log("\n[dry-run] 未写文件")
        return 0

    with open(src, "w", encoding="utf-8") as f:
        for r in dedup:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\n✅ 已写入 {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
