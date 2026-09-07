# -*- coding: utf-8 -*-
"""
构建《艾尔登法环》官方中英术语对照表。

数据源: github.com/elden-ring-data/msg
  - engus/item.msgbnd.dcx.json  官方英文文本
  - zhocn/item.msgbnd.dcx.json  官方简体中文文本

原理: 两份文件键为 {"<绝对路径>\\<类别>.fmg": {"<ID>": "<文本>"}}
      中英路径对称、ID 一致，按 (类别, ID) 对齐即得官方译名。

用法:
    python scripts/build_glossary.py
输出:
    data/elden_ring/glossary.json     结构化对照表
    data/elden_ring/glossary.tsv      便于人工校对
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "_raw"
OUT_DIR = ROOT / "data" / "elden_ring"

BASE = "https://raw.githubusercontent.com/elden-ring-data/msg/master"
FILES = {
    "en": ("engus/item.msgbnd.dcx.json", 2_076_602),
    "zh": ("zhocn/item.msgbnd.dcx.json", 1_861_632),
}
EXPECT_MIN_RATIO = 0.90   # 低于预期体积 90% 视为下载不完整
MAX_RETRY = 5


def log(msg):
    print(msg, flush=True)


def download(rel_path, expect_size, dest: Path):
    """带重试与体积校验的下载。大文件易被截断，必须校验。"""
    if dest.exists() and dest.stat().st_size >= expect_size * EXPECT_MIN_RATIO:
        log(f"  [缓存] {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest

    url = f"{BASE}/{rel_path}"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            log(f"  [下载] 第 {attempt} 次  {rel_path}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            size = len(data)
            if size < expect_size * EXPECT_MIN_RATIO:
                log(f"    不完整 {size:,} < 预期 {expect_size:,}，重试")
                time.sleep(2)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            log(f"    完成 {size:,} bytes")
            return dest
        except Exception as e:
            log(f"    失败: {type(e).__name__}: {e}")
            time.sleep(3)
    raise RuntimeError(f"下载失败（已重试 {MAX_RETRY} 次）: {url}")


def category_of(path: str) -> str:
    """从 'N:\\GR\\...\\engUS\\WeaponName.fmg' 提取 'WeaponName'"""
    name = re.split(r"[\\/]", path)[-1]
    return name.replace(".fmg", "")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 56)
    log("步骤 1/3  下载官方游戏文本")
    log("=" * 56)
    raw = {}
    for lang, (rel, size) in FILES.items():
        raw[lang] = download(rel, size, CACHE / f"item_{lang}.json")

    log("")
    log("=" * 56)
    log("步骤 2/3  解析并对齐")
    log("=" * 56)
    en_raw, zh_raw = load(raw["en"]), load(raw["zh"])

    # 路径 -> 类别，同名类别合并
    def index(d):
        out = {}
        for path, entries in d.items():
            if not isinstance(entries, dict):
                continue
            out.setdefault(category_of(path), {}).update(entries)
        return out

    en, zh = index(en_raw), index(zh_raw)
    shared = sorted(set(en) & set(zh))
    log(f"  英文类别 {len(en)} 个 / 中文类别 {len(zh)} 个 / 可对齐 {len(shared)} 个")

    glossary = {}
    stats = {}
    for cat in shared:
        ids = set(en[cat]) & set(zh[cat])
        pairs = {}
        for i in ids:
            e, z = en[cat].get(i), zh[cat].get(i)
            if not isinstance(e, str) or not isinstance(z, str):
                continue
            e, z = e.strip(), z.strip()
            # 跳过空值、纯占位符、以及中英完全相同的（多为未翻译的编号串）
            if not e or not z:
                continue
            if e == z:
                continue
            if set(e) <= set("0123456789-_%s") :
                continue
            pairs[i] = {"en": e, "zh": z}
        if pairs:
            glossary[cat] = pairs
            stats[cat] = len(pairs)

    total = sum(stats.values())
    log(f"  对齐成功 {total} 条，分布：")
    for cat, n in sorted(stats.items(), key=lambda x: -x[1]):
        log(f"    {cat:<28} {n:>5}")

    log("")
    log("=" * 56)
    log("步骤 3/3  写出结果")
    log("=" * 56)
    out_json = OUT_DIR / "glossary.json"
    out_json.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"  {out_json.relative_to(ROOT)}")

    out_tsv = OUT_DIR / "glossary.tsv"
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("category\tid\ten\tzh\n")
        for cat, pairs in glossary.items():
            for i, p in pairs.items():
                f.write(f"{cat}\t{i}\t{p['en']}\t{p['zh']}\n")
    log(f"  {out_tsv.relative_to(ROOT)}")

    # 便于检索的快速映射（英文小写 -> 中文）
    flat = {}
    for cat, pairs in glossary.items():
        for i, p in pairs.items():
            flat.setdefault(p["en"].lower(), p["zh"])
    out_flat = OUT_DIR / "glossary_flat.json"
    out_flat.write_text(
        json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"  {out_flat.relative_to(ROOT)}  ({len(flat)} 条去重映射)")

    log("")
    log(f"✅ 完成，共 {total} 条官方术语对照")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"\n❌ 失败: {type(e).__name__}: {e}")
        sys.exit(1)
