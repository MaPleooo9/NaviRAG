# -*- coding: utf-8 -*-
"""
L3 常规打法层：Markdown → JSONL

为什么和 build_l2.py 分开而不是加个参数：
    L2 和 L3 的核心字段不一样（无脑度 vs 难度、风险 vs 核心思路），
    校验规则也不一样（L2 必须有无脑度，L3 必须有核心思路）。
    硬塞进一个解析器，两边都会长出大量 if 分支——分开反而更好维护。

用法:
    python scripts/build_l3.py                 # 解析并写入 l3_normal.jsonl
    python scripts/build_l3.py --check         # 只检查格式问题，不写文件

输出字段:
    id            唯一标识（er_l3_ 前缀，与 L2 区分）
    game          固定 elden_ring
    target        中文目标名
    target_en     英文官方名，用于关联 L1 wiki 页面
    title         打法名
    difficulty    难度 1-5（L2 是 brain_level 无脑度）
    level_req     建议等级
    requires      所需道具
    idea          核心思路 —— L3 独有，也是这层的精华
    steps         步骤列表
    risk          翻车点
    verified      是否亲自验证
    text          拼好的检索文本
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "elden_ring" / "l3_normal.md"
OUT = ROOT / "data" / "elden_ring" / "l3_normal.jsonl"


def log(m):
    print(m, flush=True)


def strip_tail(s):
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
                "difficulty": 0, "level_req": 0,
                "requires": [], "idea": "", "steps": [], "risk": "",
                "verified": False,
            }
            in_steps = False
            continue

        if cur is None:
            continue

        # 兼容 "- 字段:" 与 "- **字段**:" 两种写法
        m = re.match(
            r"^-\s*\*{0,2}(难度|等级|需要|核心思路|翻车点|验证)\*{0,2}\s*[:：]\s*(.*)$",
            line)
        if m:
            in_steps = False
            k, v = m.group(1), strip_tail(m.group(2))
            if k == "难度":
                cur["difficulty"] = int(re.sub(r"[^0-9]", "", v) or 0)
            elif k == "等级":
                cur["level_req"] = 0 if re.match(
                    r"^(无|任意|不限|无需求|-)$", v) else int(
                    re.sub(r"[^0-9]", "", v) or 0)
            elif k == "需要":
                cur["requires"] = [x.strip() for x in re.split(r"[,，]", v) if x.strip()]
            elif k == "核心思路":
                cur["idea"] = v
            elif k == "翻车点":
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
    """拼成一段适合 embedding 的自然文本。

    刻意把「常规打法」这个词写进文本——用户问「怎么打」而不是「怎么逃课」时，
    这条要能被语义检索捞出来。
    """
    parts = [f"【{it['target']}】常规打法：{it['title']}。"]
    if it["requires"]:
        parts.append("需要：" + "、".join(it["requires"]) + "。")
    if it["idea"]:
        parts.append(f"核心思路：{it['idea']}。")
    if it["steps"]:
        parts.append("打法：" + " ".join(
            f"{i + 1}. {s}" for i, s in enumerate(it["steps"])) + "。")
    if it["risk"]:
        parts.append(f"翻车点：{it['risk']}。")
    lv = f"建议 {it['level_req']} 级" if it["level_req"] else "无等级需求"
    parts.append(f"难度 {it['difficulty']}/5，{lv}。")
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
    marker = "<!-- DATA-START"
    if marker in md:
        md = md.split(marker, 1)[1].split("-->", 1)[-1]
    items = parse(md)

    log("=" * 60)
    log("L3 常规打法层 · Markdown → JSONL")
    log("=" * 60)

    problems = []
    for i, it in enumerate(items):
        it["id"] = f"er_l3_{slug(it['target_en'] or it['target'])}_{i:03d}"
        it["text"] = build_text(it)
        if not it["target"]:
            problems.append(f"[{i}] 缺少目标名")
        if not it["steps"]:
            problems.append(f"[{i}] {it['title']}：没有步骤")
        if not 1 <= it["difficulty"] <= 5:
            problems.append(f"[{i}] {it['title']}：难度异常 ({it['difficulty']})")
        if not it["idea"]:
            problems.append(f"[{i}] {it['title']}：缺核心思路（L3 的精华就是这句）")
        if not it["target_en"]:
            problems.append(f"[{i}] {it['target']}：缺英文名，无法关联 L1 wiki")

    log(f"\n解析出 {len(items)} 条")

    n_h3 = len(re.findall(r"(?m)^###\s+\S", md))
    if n_h3 != len(items):
        log(f"  ⚠️  文件里有 {n_h3} 个 '###' 标题，但只解析出 {len(items)} 条")
        log(f"      → 有 {n_h3 - len(items)} 条未被识别，多半是漏了 '##' 或 '###' 标记")

    targets = {}
    for it in items:
        targets[it["target"]] = targets.get(it["target"], 0) + 1
    log(f"覆盖 {len(targets)} 个目标：")
    for t, n in sorted(targets.items(), key=lambda x: -x[1]):
        log(f"  {t:<24} {n} 条")

    lv = [it["difficulty"] for it in items if it["difficulty"]]
    if lv:
        log(f"\n难度分布: 平均 {sum(lv) / len(lv):.1f}，最高 {max(lv)}，最低 {min(lv)}")
    v = sum(1 for it in items if it["verified"])
    log(f"已验证: {v}/{len(items)}")

    if problems:
        log(f"\n⚠️  {len(problems)} 处待完善：")
        for p in problems[:20]:
            log(f"   {p}")
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
    log("\n💡 下一步：重建索引才会生效 —— "
        "python -c \"import sys;sys.path.insert(0,'core');"
        "from retriever import build_index;build_index(force=True)\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
