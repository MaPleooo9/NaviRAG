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


def is_hit(meta: dict, case: dict) -> bool:
    """这条结果算不算命中期望答案"""
    if meta["layer"] != case["expect_layer"]:
        return False
    want = norm_target(case.get("expect_target", ""))
    if not want:
        return True                      # 只判层，不判具体 boss
    if meta["layer"] in ("l2", "l3"):
        # 规范名必须完全相等——用 in 的话"大树守卫"会误判命中"龙装大树守卫"
        return norm_target(meta.get("target", "")) == want
    # L1 是 wiki 条目，没有 target 字段，用标题包含判断
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

    def full(query, k):
        """D. C + 实体命中硬置顶 —— 线上实际使用的方案"""
        hits, _ = search(col, model, query, k=k, zh2en=zh2en,
                         entity_boost=True)
        return hits

    return [
        ("A 单路纯向量", pure),
        ("B +术语增强", pure_enhanced),
        ("C +分层加权", layered),
        ("D +实体命中(线上)", full),
    ]


# ---------------------------------------------------------------- 评估

def evaluate(runner, cases, k=5):
    n = len(cases)
    stat = {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0, "layer1": 0}
    by_group = defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0})
    misses = []

    for c in cases:
        results = runner(c["query"], k)
        rank = 0
        for i, r in enumerate(results[:k], 1):
            if is_hit(r["meta"], c):
                rank = i
                break

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

        if results and results[0]["meta"]["layer"] == c["expect_layer"]:
            stat["layer1"] += 1

    return {
        "hit1": stat["hit1"] / n, "hit3": stat["hit3"] / n,
        "hit5": stat["hit5"] / n, "mrr": stat["mrr"] / n,
        "layer1": stat["layer1"] / n,
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

def report(all_res, sweep_rows, cases, out_path):
    lines = ["# 检索层评估报告\n",
             f"评估集：{len(cases)} 条人工标注问法"
             "（逃课 / 常规 / 否定表达 / 无脑度 / 等级过滤 / 知识问答）\n",
             "判定标准：Top-K 中出现「期望层 + 期望目标」才算命中。\n",
             "## 整体指标（K=5）\n",
             "| 方案 | Hit@1 | Hit@3 | Hit@5 | MRR | 层准确率@1 |",
             "|---|---|---|---|---|---|"]

    for name, r in all_res:
        lines.append(
            f"| {name} | {r['hit1']:.1%} | {r['hit3']:.1%} | "
            f"{r['hit5']:.1%} | {r['mrr']:.3f} | {r['layer1']:.1%} |")

    best = all_res[-1][1]
    lines += [
        "\n## 分组表现（线上方案 D）\n",
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

    all_res = []
    for name, runner in make_runners(col, model, zh2en):
        t1 = time.time()
        r = evaluate(runner, cases, args.k)
        all_res.append((name, r))
        log(f"{name:<20} Hit@1 {r['hit1']:6.1%} | Hit@3 {r['hit3']:6.1%} | "
            f"Hit@5 {r['hit5']:6.1%} | MRR {r['mrr']:.3f} | "
            f"层准确率 {r['layer1']:6.1%}  ({time.time() - t1:.1f}s)")

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
        report(all_res, sweep_rows, cases, Path(ROOT / args.out))

    a, d = all_res[0][1], all_res[-1][1]
    log("\n" + "=" * 66)
    log(f"结论：分层设计把 Hit@1 从 {a['hit1']:.1%} 提到 {d['hit1']:.1%}"
        f"（+{d['hit1'] - a['hit1']:.1%}），MRR {a['mrr']:.3f} → {d['mrr']:.3f}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
