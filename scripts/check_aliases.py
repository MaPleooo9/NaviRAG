# -*- coding: utf-8 -*-
"""别名表校验：aliases.json 的 key 必须真的能对上条目，否则静默失效。

为什么需要这个脚本：
    别名表是「key = 实体片段」的字典，检索时用 target 拆出的片段去查它。
    如果 key 写成 target 原文（"“黑剑”玛利喀斯"）而不是片段（"玛利喀斯"），
    查不到 → 别名完全不生效，但程序不报任何错。这类静默失效在评估里
    表现为"指标没变化"，很容易被误判成"别名没用"。
    实际第一次写这张表时，7 条带引号的 target 就全中招了。

    所以在这里把 key 与真实 target 做一次对账，顺便把跨 boss 的重名别名
    挑出来（同名不同 boss 会让两条目打平、排序交给向量，等于别名失效）。

用法:
    python scripts/check_aliases.py            # 校验并报告
    python scripts/check_aliases.py --probe    # 额外演示几条俗称的命中分数
"""

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.retriever import ALIAS_SCORE, _entity_hit, load_aliases, target_names  # noqa: E402

DATA = ROOT / "data" / "elden_ring"
FILES = {"L2": "l2_cheese.jsonl", "L3": "l3_normal.jsonl"}

# 非 boss 的 target（通用技巧 / 跳关），不参与覆盖度统计
NON_BOSS_PREFIX = ("通用：", "铁处女")


def norm_entity(s):
    """去掉“…”称号前缀，得到本名：『“黑剑”玛利喀斯』→『玛利喀斯』"""
    return re.sub(r"^[“「『][^”」』]*[”」』]\s*", "", (s or "").strip())


def same_boss(t1, t2):
    """两个 target 是否指同一个 boss。

    同一 boss 在 L2/L3 常有两套写法，本名互为子串：
      『“血王”蒙格』= 本名蒙格, 『蒙格』= 蒙格          → 相等
      『“接肢”葛瑞克』本名葛瑞克 ⊂ 『接肢葛瑞克』        → 包含
    而『观星骑士罗蕾塔』与『圣树骑士罗蕾塔』互不包含 → 判定为两个 boss。
    """
    n1, n2 = norm_entity(t1), norm_entity(t2)
    return n1 == n2 or n1 in n2 or n2 in n1


def load_rows():
    rows = []
    for layer, name in FILES.items():
        p = DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((layer, json.loads(line)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="演示俗称命中分数")
    args = ap.parse_args()

    rows = load_rows()
    aliases = load_aliases()

    known = defaultdict(set)      # 实体片段 → {它来自哪些 target 原文}
    boss_targets = set()
    for _layer, r in rows:
        tgt, title = r.get("target", ""), r.get("title_zh", "")
        for n in target_names(tgt, title):
            known[n].add(tgt)
        if tgt and not tgt.startswith(NON_BOSS_PREFIX):
            boss_targets.add(tgt)

    errors, warns = [], []

    # ---- 1. key 必须能对上真实条目，否则别名永远不会生效
    for key in aliases:
        if key not in known:
            guess = difflib.get_close_matches(key, list(known), n=2, cutoff=0.5)
            hint = f"（是否想写 {' / '.join(guess)}？）" if guess else ""
            errors.append(f'key "{key}" 对不上任何条目的 target 片段，别名不会生效{hint}')

    # ---- 2. 别名撞上"别的 boss"的正式名才警告；撞上自己不算问题
    #         （"铁棘"本来就是『“铁棘”艾隆梅尔』拆出来的片段）
    for key, alist in aliases.items():
        own_targets = known.get(key, set())
        own = set()
        for t in own_targets:
            own |= set(target_names(t))
        dupes = []
        for a in alist:
            if a not in known or a in own:
                continue
            if not all(same_boss(t, u) for u in known[a] for t in own_targets):
                dupes.append(a)
        if dupes:
            warns.append(f'"{key}" 的别名 {dupes} 是另一个 boss 的正式名，可能误伤')

    # ---- 3. 同一别名挂到多个实体：只有跨 boss 才警告。
    #         同一 boss 在 L2/L3 的两套写法（“接肢”葛瑞克 / 接肢葛瑞克）
    #         共用一个别名是正常的——它们本来就该一起被置顶。
    owner = defaultdict(list)
    for key, alist in aliases.items():
        for a in alist:
            owner[a].append(key)
    for a, keys in sorted(owner.items()):
        if len(keys) < 2:
            continue
        tgts = [t for k in keys for t in known.get(k, ())]
        if all(same_boss(t, u) for t in tgts for u in tgts):
            continue
        warns.append(f'别名 "{a}" 挂在 {len(keys)} 个不同 boss 上：{" / ".join(keys)}'
                     f'（两条目会同分，排序退回向量）')

    # ---- 4. 覆盖度
    covered = set()
    for key, alist in aliases.items():
        if alist:
            covered |= known.get(key, set())
    uncovered = sorted(boss_targets - covered)

    # ---- 输出
    n_alias = sum(len(v) for v in aliases.values())
    print("=" * 64)
    print("别名表校验 · data/elden_ring/aliases.json")
    print("=" * 64)
    print(f"条目: L2 {sum(1 for l, _ in rows if l == 'L2')} 条 / "
          f"L3 {sum(1 for l, _ in rows if l == 'L3')} 条")
    print(f"别名表: {len(aliases)} 个实体 / {n_alias} 个别名")
    print(f"命中分值: {ALIAS_SCORE}（全名 2.0 / 短名 1.5 / 部分 1.0~1.5）\n")

    for e in errors:
        print(f"❌ {e}")
    for w in warns:
        print(f"⚠  {w}")
    if not errors and not warns:
        print("✅ key 全部对账通过，无跨 boss 重名警告")
    elif not errors:
        print(f"\n（{len(warns)} 条警告，均为同名不同形态，不影响检索）")

    if uncovered:
        print(f"\n未登记别名的 boss（{len(uncovered)} 个，仅作提示）：")
        for t in uncovered:
            print(f"   {t}")
    print(f"\n覆盖: {len(boss_targets) - len(uncovered)}/{len(boss_targets)} 个 boss 有别名")

    if args.probe:
        print("\n" + "=" * 64)
        print("俗称命中演示（分数 / 问句 → 目标条目）")
        print("=" * 64)
        cases = [
            ("女武神怎么逃课", "玛莲妮亚"),
            ("米凯拉的锋刃怎么逃课", "玛莲妮亚"),
            ("血王怎么逃课", "蒙格"),
            ("碎星怎么无脑打", "拉塔恩"),
            ("接肢怎么正常打", "接肢葛瑞克"),
            ("黑剑怎么正常打", "“黑剑”玛利喀斯"),
            ("黑剑眷属怎么逃课", "黑剑眷属"),
            ("满月女王怎么打", "满月女王蕾娜菈"),
            ("大蛇怎么打", "噬神大蛇"),
            ("大树守卫怎么逃课", "大树守卫"),
            ("龙装大树守卫怎么打", "龙装大树守卫"),
        ]
        for q, tgt in cases:
            s = _entity_hit(q, {"target": tgt, "title_zh": ""}, aliases)
            print(f"  {s:.2f}  {q:<20} → {tgt}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
