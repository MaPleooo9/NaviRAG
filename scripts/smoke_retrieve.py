# -*- coding: utf-8 -*-
"""
检索冒烟测试（零依赖，不需要 embedding 模型）

目的：在装好 sentence-transformers 之前，先验证一件更基础的事——
      **数据到底能不能被查到**。

重点验证跨语言链路：
    中文提问「玛莲妮亚怎么打」 → 能否命中英文 wiki 的 Malenia 页面？
    纯字符匹配做不到（"玛莲妮亚" 和 "Malenia" 没有公共字符），
    必须靠 title_zh 字段做桥接——这正是官方术语表的价值所在。

这个脚本里的关键词检索同时是**降级方案**：
    当 embedding 模型加载失败或 Ollama 挂了，系统退回这条路仍能出结果。

用法:
    python scripts/smoke_retrieve.py
    python scripts/smoke_retrieve.py -q "玛莲妮亚二阶段怎么躲"
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "elden_ring"

# 权重：标题命中比正文命中重要得多
W_TITLE_ZH = 12.0   # 中文标题命中（跨语言桥接的关键）
W_TITLE = 8.0
W_REQ = 4.0         # L2 的"需要"字段
W_TEXT = 1.0
W_SECTION = 3.0


def log(m):
    print(m, flush=True)


def load_all():
    rows = []
    for name in ["l1_wiki.jsonl", "l1_zh_game.jsonl", "l2_cheese.jsonl"]:
        p = DATA / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_src"] = name
                    rows.append(r)
    return rows


def tokenize(q):
    """中英文混合分词：英文按词，中文按 2-gram（避免单字噪音）"""
    toks = []
    for w in re.findall(r"[A-Za-z][A-Za-z'’\-]{2,}", q):
        toks.append(w.lower())
    for seg in re.findall(r"[\u4e00-\u9fff]+", q):
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks += [seg[i:i + 2] for i in range(len(seg) - 1)]
    return toks


def score(row, toks, q_lower, entities=()):
    s = 0.0
    src = row["_src"]
    title_zh = row.get("title_zh", "") or ""
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""
    section = row.get("section", "") or ""

    # 完整实体名命中：这是跨语言检索的命脉。
    # 「玛莲妮亚」完整出现在 title_zh 里 → 远胜于 2-gram 碎片碰巧撞上
    for e in entities:
        if len(e) >= 2:
            if e in title_zh:
                s += 30.0
            if e in title.lower():
                s += 18.0
        # L2 的 target 字段本身就是 boss 名
        if e in (row.get("target", "") or ""):
            s += 25.0

    for t in toks:
        if t in title_zh:
            s += W_TITLE_ZH
        if t in title.lower():
            s += W_TITLE
        if t in section.lower():
            s += W_SECTION
        for r in row.get("requires", []) or []:
            if t in str(r).lower():
                s += W_REQ
                break
        # 正文按出现次数计，但设上限防止刷分
        n = text.lower().count(t)
        if n:
            s += W_TEXT * min(n, 5)

    # 完整短语直接命中，加权
    if len(q_lower) > 3 and q_lower in text.lower():
        s += 6

    # 分层权重：这是整个系统的核心设计
    #   L2 逃课  = 人工整理的核心资产，最该被看到
    #   L1 wiki  = 攻略正文，主力
    #   L1 游戏文本 = 物品描述，只是补充语料，极易靠名字刷分，必须重罚
    if src == "l2_cheese.jsonl":
        s *= 1.8
    elif src == "l1_zh_game.jsonl":
        s *= 0.30
    return s


def search(rows, query, topk=5):
    toks = tokenize(query)
    q_lower = query.lower()
    entities = re.findall(r"[\u4e00-\u9fff]{2,}", query)

    scored = [(score(r, toks, q_lower, entities), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]
    scored.sort(key=lambda x: -x[0])

    # 去重：同一段文本只保留得分最高的一条
    seen, out = set(), []
    for s, r in scored:
        key = r["text"][:80]
        if key in seen:
            continue
        seen.add(key)
        out.append((s, r))
        if len(out) >= topk:
            break
    return out


def show(r, s, i):
    src = {"l1_wiki.jsonl": "L1-wiki(英)",
           "l1_zh_game.jsonl": "L1-游戏文本(中)",
           "l2_cheese.jsonl": "L2-逃课"}[r["_src"]]
    head = r.get("title_zh") or r.get("title", "")
    log(f"\n  [{i}] {src}  score={s:.1f}")
    log(f"      {head}" + (f"  <{r['title']}>" if r.get("title_zh") and r.get("title") else ""))
    if r.get("section"):
        log(f"      小节: {r['section']}")
    if r.get("brain_level"):
        log(f"      无脑度 {r['brain_level']}/5  需要: {'、'.join(r.get('requires', []))}")
    txt = r["text"].replace("\n", " ")
    log(f"      {txt[:170]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", default=None)
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    rows = load_all()
    if not rows:
        log("❌ 没有数据，请先跑 fetch_wiki.py / build_l1_zh.py / build_l2.py")
        return 1

    from collections import Counter
    c = Counter(r["_src"] for r in rows)
    log("=" * 64)
    log("检索冒烟测试（关键词模式，无需 embedding）")
    log("=" * 64)
    log(f"\n载入 {len(rows):,} 条：")
    for k, v in c.items():
        log(f"  {k:<20} {v:>6}")

    queries = [args.query] if args.query else [
        "玛莲妮亚二阶段怎么躲",
        "拉塔恩怎么逃课",
        "铃珠猎人",
        "最无脑的打法",
        "腐败吐息",
    ]

    for q in queries:
        log("\n" + "=" * 64)
        log(f"查询：{q}")
        log("=" * 64)
        hits = search(rows, q, args.topk)
        if not hits:
            log("  （无命中）")
            continue
        for i, (s, r) in enumerate(hits, 1):
            show(r, s, i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
