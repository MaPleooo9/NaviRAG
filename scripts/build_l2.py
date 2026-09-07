# -*- coding: utf-8 -*-
"""
L2 逃课层：Markdown → JSONL

为什么用 Markdown 而不是直接写 JSON：
    L2 要人工写 200+ 条，让人手写 JSON 是反人类的。Markdown 里一条只需 6-8 行，
    改起来也直观。这个脚本负责把它转成检索需要的结构化格式。

用法:
    python scripts/build_l2.py                 # 解析并写入 l2_cheese.jsonl
    python scripts/build_l2.py --check         # 只检查格式问题，不写文件

输出字段:
    id            唯一标识
    game          固定 elden_ring
    target        中文目标名（boss / 区域）
    target_en     英文官方名，用于关联 L1 wiki 页面
    title         打法名
    brain_level   无脑度 1-5
    level_req     建议等级
    requires      所需道具列表
    steps         步骤列表
    risk          翻车点
    verified      是否亲自验证
    text          拼好的检索文本（embedding 用）
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "elden_ring" / "l2_cheese.md"
OUT = ROOT / "data" / "elden_ring" / "l2_cheese.jsonl"


def log(m):
    print(m, flush=True)


def strip_tail(s):
    """去掉结尾多余的分隔符：AI 生成的条目常在每步末尾留逗号"""
    return re.sub(r"[,，;；]\s*$", "", s.strip()).strip()


def slug(s):
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s.lower())
    return s.strip("_")


def parse(md):
    items, target, target_en = [], "", ""
    cur = None
    in_steps = False

    def flush():
        if cur and cur.get("title"):
            items.append(cur)

    for raw in md.splitlines():
        line = raw.rstrip()

        if line.startswith("## ") and not line.startswith("###"):
            flush()
            cur = None
            head = line[3:].strip()
            if "|" in head:
                target, target_en = [x.strip() for x in head.split("|", 1)]
            else:
                target, target_en = head, ""
            in_steps = False
            continue

        if line.startswith("### "):
            flush()
            cur = {
                "id": "", "game": "elden_ring",
                "target": target, "target_en": target_en,
                "title": line[4:].strip(),
                "brain_level": 0, "level_req": 0,
                "requires": [], "steps": [], "risk": "", "verified": False,
            }
            in_steps = False
            continue

        if cur is None:
            continue

        # 兼容 "- 字段:" 与 "- **字段**:" 两种写法（Markdown 粗体是常见变体）
        m = re.match(r"^-\s*\*{0,2}(无脑度|等级|需要|风险|验证)\*{0,2}\s*[:：]\s*(.*)$", line)
        if m:
            in_steps = False
            k, v = m.group(1), strip_tail(m.group(2))
            if k == "无脑度":
                cur["brain_level"] = int(re.sub(r"[^0-9]", "", v) or 0)
            elif k == "等级":
                # "无需求"/"任意"/"无" 视为无门槛，记 0
                cur["level_req"] = 0 if re.match(
                    r"^(无|任意|不限|无需求|-)$", v) else int(
                    re.sub(r"[^0-9]", "", v) or 0)
            elif k == "需要":
                cur["requires"] = [x.strip() for x in re.split(r"[,，]", v) if x.strip()]
            elif k == "风险":
                cur["risk"] = v
            elif k == "验证":
                cur["verified"] = v.startswith(("是", "y", "Y", "1", "true", "True"))
            continue

        if re.match(r"^-\s*\*{0,2}步骤\*{0,2}\s*[:：]?\s*$", line):
            in_steps = True
            continue

        if in_steps:
            m = re.match(r"^\s*\d+[.、)]\s*(.+)$", line)
            if m:
                cur["steps"].append(strip_tail(m.group(1)))
            elif line.strip() and not line.startswith(("#", "-")):
                cur["steps"].append(strip_tail(line))

    flush()
    return items


def build_text(it):
    """拼成一段适合 embedding 的自然文本"""
    parts = [f"【{it['target']}】{it['title']}。"]
    if it["requires"]:
        parts.append("需要：" + "、".join(it["requires"]) + "。")
    if it["steps"]:
        parts.append("打法：" + " ".join(
            f"{i + 1}. {s}" for i, s in enumerate(it["steps"])) + "。")
    if it["risk"]:
        parts.append("注意：" + it["risk"] + "。")
    lv = f"建议 {it['level_req']} 级" if it["level_req"] else "无等级需求"
    parts.append(f"无脑度 {it['brain_level']}/5，{lv}。")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查不写文件")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        log(f"❌ 未找到 {src}")
        return 1

    md = src.read_text(encoding="utf-8")
    # 只解析 DATA-START 标记之后的内容，标记上方是说明文档与示例
    marker = "<!-- DATA-START"
    if marker in md:
        md = md.split(marker, 1)[1].split("-->", 1)[-1]
    items = parse(md)

    log("=" * 60)
    log("L2 逃课层 · Markdown → JSONL")
    log("=" * 60)

    problems = []
    for i, it in enumerate(items):
        it["id"] = f"er_{slug(it['target_en'] or it['target'])}_{i:03d}"
        it["text"] = build_text(it)
        if not it["target"]:
            problems.append(f"[{i}] 缺少目标名")
        if not it["steps"]:
            problems.append(f"[{i}] {it['title']}：没有步骤")
        if not 1 <= it["brain_level"] <= 5:
            problems.append(f"[{i}] {it['title']}：无脑度异常 ({it['brain_level']})")
        if not it["target_en"]:
            problems.append(f"[{i}] {it['target']}：缺英文名，无法关联 L1 wiki")

    log(f"\n解析出 {len(items)} 条")

    # 丢条目是最容易发生又最难发现的问题：比对 ### 标题数与实际解析数
    n_h3 = len(re.findall(r"(?m)^###\s+\S", md))
    if n_h3 != len(items):
        log(f"  ⚠️  文件里有 {n_h3} 个 '###' 标题，但只解析出 {len(items)} 条")
        log(f"      → 有 {n_h3 - len(items)} 条未被识别，多半是漏了 '##' 或 '###' 标记")

    # 统计
    targets = {}
    for it in items:
        targets[it["target"]] = targets.get(it["target"], 0) + 1
    log(f"覆盖 {len(targets)} 个目标：")
    for t, n in sorted(targets.items(), key=lambda x: -x[1]):
        log(f"  {t:<20} {n} 条")

    lv = [it["brain_level"] for it in items if it["brain_level"]]
    if lv:
        log(f"\n无脑度分布: 平均 {sum(lv) / len(lv):.1f}，最高 {max(lv)}，最低 {min(lv)}")
    v = sum(1 for it in items if it["verified"])
    log(f"已验证: {v}/{len(items)}")

    if problems:
        log(f"\n⚠️  {len(problems)} 处待完善：")
        for p in problems[:20]:
            log(f"   {p}")
        if len(problems) > 20:
            log(f"   ...还有 {len(problems) - 20} 处")
    else:
        log("\n✅ 格式检查全部通过")

    if args.check:
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log(f"\n✅ 已写入 {out}  ({len(items)} 条)")

    if len(items) < 100:
        log(f"\n💡 目标 100 条起步，还差 {100 - len(items)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
