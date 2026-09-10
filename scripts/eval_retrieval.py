# -*- coding: utf-8 -*-
"""
检索层评估：Hit@K / MRR / 层准确率 + 消融实验 + 权重扫描。

为什么自己写而不用 RAGAS：
    RAGAS 评估的是"生成答案"的质量，核心是用 LLM 打分。但本项目要证明的是
    **检索层的分层设计有效**——问题在"召回哪条"，不在"话说得好不好"。
    而且 RAGAS 需要额外 LLM 调用，本地 3b 模型打分不可信。
    这里用可判定的硬指标：期望层 + 期望目标，命中就算对。

指标:
    Hit@K   Top-K 里出现期望结果的比例（K=1/3/5）
    MRR     首个正确结果的排名倒数，综合反映"排得够不够前"
    层准确率@1  Top1 的层是不是期望的层（层判错比排错更严重）

用法:
    python scripts/eval_retrieval.py                  # 全套评估
    python scripts/eval_retrieval.py --sweep          # 额外跑权重扫描
    python scripts/eval_retrieval.py --out docs/eval.md
"""

from __future__ import annotations

# 必须赶在 import torch 之前设置：多线程 BLAS 的浮点归约顺序不确定，
# 会让同一条 query 的 embedding 在不同进程里出现微小差异，
# 进而让 HNSW 候选边界上的排序翻转。
# 实测同一份代码连跑三次，单路纯向量的层准确率在 38.0% / 39.4% 之间跳。
# 评估脚本宁可慢一点也要可复现，所以锁死单线程。
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.retriever import (  # noqa: E402
    build_index, enhance_query, load_zh2en, search)

EVAL_SET = ROOT / "data" / "elden_ring" / "eval_set.json"


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------- 判定

def norm_target(s: str) -> str:
    """去掉称号前缀：”血王”蒙格 → 蒙格。

    必须去掉，否则 "蒙格" 和 "血王蒙格" 会被判成两个 boss。
    """
    return re.sub(r"^[“「『][^”」』]*[”」』]\s*", "", (s or "").strip())


def expect_layers(case: dict) -> list[str]:
    """期望层统一成列表。

    L3 扩充后「正常打法」命中 L3 条目比命中 L1 wiki 页更好，
    所以这类用例的 expect_layer 写成 ["l1", "l3"]，
    判定时必须按列表处理。

    曾经踩过的坑：层准确率统计用 `==` 直接比 expect_layer，
    遇列表时 `"l1" == ["l1","l3"]` 恒为 False，导致这 2 条
    无论排多准都被判错。Hit@1 走的是 is_hit（支持列表），
    于是出现「层准确率(97.2%) < Hit@1(98.6%)」这种不该有的倒挂。
    """
    layers = case["expect_layer"]
    return [layers] if isinstance(layers, str) else list(layers)


def layer_match(meta: dict, case: dict) -> bool:
    """结果的层是否符合期望。is_hit 与层准确率统计共用以保证口径一致。"""
    return meta["layer"] in expect_layers(case)


def page_main(meta: dict) -> str:
    """L1 wiki 页面的主实体名：去称号前缀 + 去括号后缀。

    L1 条目没有 target 字段，只能拿标题当实体名。
    「玛莲妮亚的追忆」是物品页，不是「玛莲妮亚」本体页——
    严格口径下必须能区分这两者，否则「标题里沾到名字就算对」，
    L1 组的指标会被系统性高估。
    """
    t = (meta.get("title_zh") or "") or (meta.get("title") or "")
    return re.split(r"[（(]", norm_target(t))[0].strip()


def is_hit(meta: dict, case: dict, strict: bool = False) -> bool:
    """这条结果算不算命中期望答案

    expect_layer 支持传列表：L3 扩充后「正常打法」命中验证过的 L3 条目
    比命中 L1 wiki 页更好，所以这类用例期望 ["l1", "l3"] 都算对。
    """
    if not layer_match(meta, case):
        return False
    want = norm_target(case.get("expect_target", ""))
    if not want:
        return True                      # 只判层，不判具体 boss
    if meta["layer"] in ("l2", "l3"):
        # 规范名必须完全相等——用 in 的话"大树守卫"会误判命中"龙装大树守卫"
        return norm_target(meta.get("target", "")) == want
    if strict:
        # 严格：页面主实体就是它本人，名字里带它的其他页面不算
        return page_main(meta) == want
    # 宽松：标题包含即可（L1 是 wiki 条目，页面名可能有前后缀）
    hay = (meta.get("title_zh") or "") + (meta.get("title") or "")
    return want in hay


# ---------------------------------------------------------------- 检索方案

def make_runners(col, model, zh2en):
    """四种方案，逐层叠加能力，用来量化每一层设计各自的贡献"""

    def pure(query, k):
        """A. 单路纯向量：不分路、不增强、不过滤——任何 RAG 教程的起点"""
        emb = model.encode([query], normalize_embeddings=True)[0].tolist()
        res = col.query(query_embeddings=[emb], n_results=k,
                        include=["metadatas"])
        return [{"meta": m} for m in res["metadatas"][0]]

    def pure_enhanced(query, k):
        """B. A + 官方术语表查询增强（玛莲妮亚 → 玛莲妮亚 Malenia）"""
        q = enhance_query(query, zh2en)
        emb = model.encode([q], normalize_embeddings=True)[0].tolist()
        res = col.query(query_embeddings=[emb], n_results=k,
                        include=["metadatas"])
        return [{"meta": m} for m in res["metadatas"][0]]

    def layered(query, k):
        """C. B + 三路分层召回 + 规则路由加权（但不开实体命中置顶）"""
        hits, _ = search(col, model, query, k=k, zh2en=zh2en,
                         entity_boost=False)
        return hits

    def full_no_enhance(query, k):
        """D−enhance：分层 + 实体置顶，但**不做**术语增强。

        B 方案已经证明「在单路检索里注入英文名是负收益」，可线上却一直开着
        增强——这是个说不过去的点。这一组单独在「分层体系」下关掉增强，
        才能回答：增强在完整方案里到底是帮忙还是帮倒忙。
        """
        hits, _ = search(col, model, query, k=k, entity_boost=True)
        return hits

    def full(query, k):
        """D. C + 实体命中硬置顶 —— 线上实际使用的方案"""
        hits, _ = search(col, model, query, k=k, zh2en=zh2en,
                         entity_boost=True)
        return hits

    return [
        ("A 单路纯向量", pure),
        ("B +术语增强", pure_enhanced),
        ("C +分层加权", layered),
        ("D−enhance(关增强)", full_no_enhance),
        ("D +实体命中(线上)", full),
    ]


def make_rule_runner(col):
    """E. 纯实体规则：一次向量都不查，只看实体能不能从问句里被认出来。

    这是把 98.6% 拆开的关键对照组——如果 E 已经很高，说明这个数字
    主要来自「词典命中」而不是「语义检索」。

    只在「期望 L2/L3」的用例子集上有意义：L1 条目没有 target 字段，
    纯规则无法定位到具体 wiki 页面。
    """
    from core.retriever import _entity_hit

    cache = {}

    def all_items(layer):
        if layer not in cache:
            got = col.get(where={"layer": layer}, include=["metadatas"])
            cache[layer] = got["metadatas"]
        return cache[layer]

    def rule_only(query, k):
        hits = []
        for layer in ("l2", "l3"):
            for meta in all_items(layer):
                eh = _entity_hit(query, meta)
                if eh > 0:
                    hits.append({"meta": meta, "eh": eh})
        hits.sort(key=lambda x: -x["eh"])
        return hits[:k]

    return rule_only


# ---------------------------------------------------------------- 评估

def evaluate(runner, cases, k=5):
    n = len(cases)
    stat = {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0, "layer1": 0,
            "hit1_s": 0, "hit5_s": 0, "mrr_s": 0.0}
    by_group = defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0})
    misses = []

    for c in cases:
        results = runner(c["query"], k)
        rank = 0
        rank_s = 0
        for i, r in enumerate(results[:k], 1):
            if not rank and is_hit(r["meta"], c):
                rank = i
            if not rank_s and is_hit(r["meta"], c, strict=True):
                rank_s = i

        g = by_group[c["group"]]
        g["n"] += 1
        if rank:
            stat["mrr"] += 1.0 / rank
            if rank == 1:
                stat["hit1"] += 1
                g["hit1"] += 1
            if rank <= 3:
                stat["hit3"] += 1
            stat["hit5"] += 1
            g["hit5"] += 1
        else:
            top = results[0]["meta"] if results else {}
            misses.append((c, top.get("layer", "-"),
                           norm_target(top.get("target", "")) or
                           (top.get("title_zh") or top.get("title") or "")[:24]))

        # 严格口径独立统计：不与宽松口径混用，指标表两边并列呈现
        if rank_s:
            stat["mrr_s"] += 1.0 / rank_s
            if rank_s == 1:
                stat["hit1_s"] += 1
            stat["hit5_s"] += 1

        # 层准确率：必须与 is_hit 用同一套层判定，否则会出现
        # 「层准确率 < Hit@1」的倒挂（列表型 expect_layer 被永久判错）
        if results and layer_match(results[0]["meta"], c):
            stat["layer1"] += 1

    return {
        "hit1": stat["hit1"] / n, "hit3": stat["hit3"] / n,
        "hit5": stat["hit5"] / n, "mrr": stat["mrr"] / n,
        "layer1": stat["layer1"] / n,
        "hit1_s": stat["hit1_s"] / n, "hit5_s": stat["hit5_s"] / n,
        "mrr_s": stat["mrr_s"] / n,
        "by_group": dict(by_group), "misses": misses, "n": n,
    }


def sweep(col, model, zh2en, cases, k=5):
    """扫描 L2/L3 权重配比，看当前默认值是不是最优。

    固定 w_l1=0.2，让 w_l2 + w_l3 = 0.8，扫一遍两者的比例。
    """
    rows = []
    for w2 in [x / 10 for x in range(1, 8)]:
        w3 = round(0.8 - w2, 2)
        if w3 < 0.1:
            continue

        def runner(query, k, w2=w2, w3=w3):
            hits, _ = search(col, model, query, k=k, zh2en=zh2en,
                             w_l1=0.2, w_l2=w2, w_l3=w3)
            return hits

        r = evaluate(runner, cases, k)
        rows.append((w2, w3, r["hit1"], r["hit5"], r["mrr"]))
    return rows


# ---------------------------------------------------------------- 输出

def report(all_res, sub_res, rule_cases, sweep_rows, cases, out_path):
    groups = sorted({c["group"] for c in cases},
                    key=lambda g: -sum(1 for c in cases if c["group"] == g))
    lines = ["# 检索层评估报告\n",
             f"评估集：{len(cases)} 条人工标注问法，"
             f"分 {len(groups)} 组：{' / '.join(groups)}\n",
             "判定标准：Top-K 中出现「期望层 + 期望目标」才算命中。\n",
             "\n两套判定口径并列呈现（以前只有宽松口径，容易被误读）：\n",
             "- **宽松**：L2/L3 要求 `target` 与期望规范名完全相等；L1 无 `target` 字段，"
             "降级为「标题包含期望名」——同名词页面（如「玛莲妮亚的追忆」）也算对。\n",
             "- **严格**：L1 升级为「页面主实体就是它本人」（去称号、去括号后缀后全等），"
             "L2/L3 不变。两组数字差得越多，说明虚高部分越大。\n",
             "## 整体指标（K=5）\n",
             "| 方案 | Hit@1 | Hit@1(严格) | Hit@3 | Hit@5 | MRR | MRR(严格) | 层准确率@1 |",
             "|---|---|---|---|---|---|---|---|"]

    for name, r in all_res:
        lines.append(
            f"| {name} | {r['hit1']:.1%} | {r['hit1_s']:.1%} | {r['hit3']:.1%} | "
            f"{r['hit5']:.1%} | {r['mrr']:.3f} | {r['mrr_s']:.3f} | "
            f"{r['layer1']:.1%} |")

    best = all_res[-1][1]

    if sub_res:
        n_sub = len(rule_cases)
        lines += [
            "\n## 归因拆分：线上方案的成绩里，规则和向量各占多少\n",
            f"统一在「期望 L2/L3」的 {n_sub} 条子集上比较——L1 条目没有 "
            "`target` 字段，纯规则无法定位到具体 wiki 页面，"
            "混在全集上比不公平。\n",
            "| 方案 | Hit@1 | Hit@5 | MRR |", "|---|---|---|---|",
        ]
        for name in ["A 单路纯向量", "C +分层加权", "E 纯实体规则(无向量)",
                     "D−enhance(关增强)", "D +实体命中(线上)"]:
            r = sub_res[name]
            lines.append(
                f"| {name} | {r['hit1']:.1%} | {r['hit5']:.1%} | "
                f"{r['mrr']:.3f} |")

        e, c = sub_res["E 纯实体规则(无向量)"], sub_res["C +分层加权"]
        d_sub = sub_res["D +实体命中(线上)"]
        ne_sub = sub_res["D−enhance(关增强)"]
        full = dict(all_res)
        d = full["D +实体命中(线上)"]
        ne = full["D−enhance(关增强)"]
        lines += [
            "",
            f"- **一次向量都不查**的纯实体规则，已经能把正确答案捞进候选框"
            f"（Hit@5 {e['hit5']:.1%}），但排不准首位（Hit@1 只有 "
            f"{e['hit1']:.1%}）；向量+分层把前者的问题补上，"
            f"Hit@1 拉到 {d_sub['hit1']:.1%}。"
            f"所以是**规则主召回、向量主精排**，二者互补而非替代——"
            f"单独拆开谁都不够看（纯向量只有 {c['hit1']:.1%}）。",
            f"- 顺带了结一个旧疑点：B 已说明「术语增强在**单路**检索里有害」"
            f"（MRR {full['A 单路纯向量']['mrr']:.3f} → "
            f"{full['B +术语增强']['mrr']:.3f}）；"
            f"但在**分层体系**里它的影响落在噪声级——Hit@1 与 MRR 和关掉时"
            f"完全一致（{d['hit1']:.1%} / {d['mrr']:.3f} vs "
            f"{ne['hit1']:.1%} / {ne['mrr']:.3f}），"
            f"Hit@5 各赢一半（全量 {d['hit5']:.1%} vs {ne['hit5']:.1%}，"
            f"L2/L3 子集 {d_sub['hit5']:.1%} vs {ne_sub['hit5']:.1%}）。"
            f"即：增强在完整链路里既没被证明有用、也没造成明显损害——"
            f"比起保留一个无法证明价值的模块，更该评估直接关掉它以简化链路。",
        ]

    lines += [
        "\n## 分组表现（线上方案 D）\n",
        "「逃课 / 常规 / 否定」这类组的问句写法与知识库同源（写题时就知道库里有"
        "哪个 boss 的条目），高分是必然的；**「口语俗称」和「描述提问」才是难度组**"
        "——前者测俗称别名覆盖，后者整句不含任何实体名，实体硬置顶完全失效、"
        "只能靠语义召回。这两组的分数才代表真实泛化能力。\n",
        "| 分组 | 条数 | Hit@1 | Hit@5 |", "|---|---|---|---|",
    ]
    for g, s in sorted(best["by_group"].items(), key=lambda x: -x[1]["n"]):
        lines.append(f"| {g} | {s['n']} | {s['hit1'] / s['n']:.0%} | "
                     f"{s['hit5'] / s['n']:.0%} |")

    if best["misses"]:
        lines += ["\n## 未命中案例（Top1 实际给了什么）\n",
                  "| 提问 | 期望 | Top1 实际 |", "|---|---|---|"]
        for c, lay, tgt in best["misses"]:
            lines.append(f"| {c['query']} | {c['expect_layer']}/"
                         f"{norm_target(c['expect_target']) or '任意'} | "
                         f"{lay}/{tgt} |")

    if sweep_rows:
        lines += ["\n## 权重扫描（固定 w_l1=0.2）\n",
                  "| w_l2 | w_l3 | Hit@1 | Hit@5 | MRR |", "|---|---|---|---|---|"]
        for w2, w3, h1, h5, mrr in sweep_rows:
            lines.append(f"| {w2:.1f} | {w3:.1f} | {h1:.1%} | {h5:.1%} | {mrr:.3f} |")
        b = max(sweep_rows, key=lambda x: (x[2], x[4]))
        lines.append(f"\n最优配比：w_l2={b[0]:.1f} / w_l3={b[1]:.1f}"
                     f"（Hit@1 {b[2]:.1%}）")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"\n📄 报告已写入 {out_path}")


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(EVAL_SET))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="额外跑权重扫描")
    ap.add_argument("--out", default=None, help="结果写入 markdown")
    args = ap.parse_args()

    log("=" * 66)
    log("NaviRAG 检索层评估")
    log("=" * 66)

    cases = json.loads(Path(args.set).read_text(encoding="utf-8"))["cases"]
    log(f"\n评估集 {len(cases)} 条")

    t0 = time.time()
    _client, col, model = build_index()
    zh2en = load_zh2en()
    log(f"索引 {col.count():,} 条，加载 {time.time() - t0:.1f}s\n")

    runners = dict(make_runners(col, model, zh2en))
    all_res = []
    for name, runner in runners.items():
        t1 = time.time()
        r = evaluate(runner, cases, args.k)
        all_res.append((name, r))
        log(f"{name:<20} Hit@1 {r['hit1']:6.1%} | 严格 {r['hit1_s']:6.1%} | "
            f"Hit@3 {r['hit3']:6.1%} | Hit@5 {r['hit5']:6.1%} | "
            f"MRR {r['mrr']:.3f} | 层准确率 {r['layer1']:6.1%}  ({time.time() - t1:.1f}s)")

    # ---- 归因拆分：把线上方案的分数拆到「规则」和「向量」两个因子上 ----
    # 只在期望 L2/L3 的用例上比，因为纯规则方案无法定位 L1 wiki 页面。
    rule_cases = [c for c in cases if set(expect_layers(c)) & {"l2", "l3"}]
    sub_res = {name: evaluate(runner, rule_cases, args.k)
               for name, runner in runners.items()}
    sub_res["E 纯实体规则(无向量)"] = evaluate(
        make_rule_runner(col), rule_cases, args.k)

    log(f"\n归因拆分（在「期望 L2/L3」的 {len(rule_cases)} 条子集上，"
        f"各组可比）：")
    for name in ["A 单路纯向量", "C +分层加权", "E 纯实体规则(无向量)",
                 "D−enhance(关增强)", "D +实体命中(线上)"]:
        r = sub_res[name]
        log(f"  {name:<20} Hit@1 {r['hit1']:6.1%} | Hit@5 {r['hit5']:6.1%} "
            f"| MRR {r['mrr']:.3f}")

    log("\n分组表现（线上方案 D）：")
    for g, s in sorted(all_res[-1][1]["by_group"].items(),
                       key=lambda x: -x[1]["n"]):
        log(f"  {g:<10} {s['hit1']}/{s['n']} Hit@1  |  {s['hit5']}/{s['n']} Hit@5")

    misses = all_res[-1][1]["misses"]
    if misses:
        log(f"\n未命中 {len(misses)} 条：")
        for c, lay, tgt in misses:
            log(f"  ❌ {c['query']:<22} 期望 {c['expect_layer']}/"
                f"{norm_target(c['expect_target']) or '任意':<8} "
                f"实际 {lay}/{tgt}")

    sweep_rows = []
    if args.sweep:
        log("\n权重扫描（固定 w_l1=0.2）：")
        sweep_rows = sweep(col, model, zh2en, cases, args.k)
        for w2, w3, h1, h5, mrr in sweep_rows:
            log(f"  w_l2 {w2:.1f} / w_l3 {w3:.1f}  →  "
                f"Hit@1 {h1:6.1%} | MRR {mrr:.3f}")

    if args.out:
        report(all_res, sub_res, rule_cases, sweep_rows, cases,
               Path(ROOT / args.out))

    a, d = all_res[0][1], all_res[-1][1]
    log("\n" + "=" * 66)
    log(f"结论：分层设计把 Hit@1 从 {a['hit1']:.1%} 提到 {d['hit1']:.1%}"
        f"（+{d['hit1'] - a['hit1']:.1%}），MRR {a['mrr']:.3f} → {d['mrr']:.3f}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
