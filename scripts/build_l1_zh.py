# -*- coding: utf-8 -*-
"""
把官方游戏文本里的中文描述导出为 L1 中文补充语料。

来源：data/elden_ring/glossary.json（官方中英对照表）
其中 Caption / Info 类不是名词，而是官方中文的**整段物品描述**，例如：

    「绘有黄色古龙图案的护符。能提升雷属性减伤率。
      在黄金树尚未出现的史前时代，据说古龙是那时代的主宰……」

作用：
    L1 主库来自英文 wiki（eldenring.wiki.gg）。中文提问检索英文语料会有跨语言损失，
    这批官方中文文本可以作为**中文侧的补充召回源**，且它是官方翻译、质量远高于机翻。

产出：data/elden_ring/l1_zh_game.jsonl，schema 与 l1_wiki.jsonl 完全一致（lang="zh"）

用法:
    python scripts/build_l1_zh.py
    python scripts/build_l1_zh.py --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "data" / "elden_ring" / "glossary.json"
OUT = ROOT / "data" / "elden_ring" / "l1_zh_game.jsonl"

# 这些类别是"整段描述"而非名词，才是有效语料
DESC_CATS = {
    "WeaponCaption": "武器",
    "GoodsCaption": "道具",
    "AccessoryCaption": "护符",
    "ProtectorCaption": "防具",
    "MagicCaption": "魔法",
    "ArtsCaption": "战灰",
    "GoodsInfo": "道具",
    "WeaponInfo": "武器",
}

MIN_CHARS = 40


def log(m):
    print(m, flush=True)


def norm(t):
    """官方文本里有大量换行与全角空格，压成连贯段落"""
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s*\n\s*", "", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not GLOSSARY.exists():
        log(f"❌ 未找到 {GLOSSARY}，请先运行 scripts/build_glossary.py")
        return 1

    with open(GLOSSARY, encoding="utf-8") as f:
        g = json.load(f)

    log("=" * 60)
    log("L1 中文补充语料 · 官方游戏文本")
    log("=" * 60)

    rows, stats = [], {}
    for cat, pairs in g.items():
        if cat not in DESC_CATS:
            continue
        n = 0
        for _id, v in pairs.items():
            zh = norm(v.get("zh", ""))
            en = norm(v.get("en", ""))
            if len(zh) < MIN_CHARS:
                continue
            rows.append({
                "id": f"game:{cat.lower()}:{_id}",
                "title": en[:80] or zh[:20],
                "title_zh": zh[:20],
                "category": f"GameText-{DESC_CATS[cat]}",
                "section": "",
                "url": "",
                "text": zh,
                "text_en": en,
                "lang": "zh",
                "chars": len(zh),
            })
            n += 1
        if n:
            stats[cat] = n

    total = len(rows)
    log(f"\n合计 {total:,} 条中文语料")
    for c, n in sorted(stats.items(), key=lambda x: -x[1]):
        log(f"  {c:<20} {n:>6}")

    if args.dry_run:
        log("\n[dry-run] 前 3 条预览：")
        for r in rows[:3]:
            log(f"\n  [{r['category']}] {r['text'][:120]}...")
        return 0

    if total == 0:
        log("无数据")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    avg = sum(r["chars"] for r in rows) / total
    log(f"\n✅ 已写入 {out}")
    log(f"   平均长度 {avg:.0f} 字")
    return 0


if __name__ == "__main__":
    sys.exit(main())
