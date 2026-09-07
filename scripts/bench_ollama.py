# -*- coding: utf-8 -*-
"""
Ollama 本地推理速度实测。

目的: 用真实数字回答"这个模型在我机器上到底多快"，
      避免凭感觉选模型，也避免面试时被问"多快"答不上来。

用法:
    python scripts/bench_ollama.py                      # 测默认模型
    python scripts/bench_ollama.py --models qwen2.5:3b qwen2.5:7b
    python scripts/bench_ollama.py --runs 5 --out docs/bench.md

输出指标:
    - TTFT  首字延迟 (秒)   决定"等多久才开始出字"
    - tok/s 生成速度        决定"出字有多快"
    - 250字中文预计耗时      最直观的体感指标
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://localhost:11434"
DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:7b"]

# 贴近真实场景的中文 prompt（约 40 字，模拟 RAG 问答的输入长度）
PROMPT = (
    "你是《艾尔登法环》的攻略助手。请根据以下资料，用中文简要说明"
    "“米凯拉的锋刃”玛莲妮亚的二阶段应对要点，控制在 250 字以内。\n\n"
    "资料：玛莲妮亚二阶段会以猩红腐败的花朵开场，水鸟乱舞有明确前摇，"
    "玩家需在她跃起的瞬间向后翻滚躲避第一段，随后侧向翻滚躲开后续两段。"
    "推荐使用冻伤或出血手段打断其节奏，并配合仿身泪滴吸引仇恨。"
)


def log(m):
    print(m, flush=True)


def api_post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def check_server():
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as r:
            return json.loads(r.read().decode("utf-8")).get("models", [])
    except urllib.error.URLError:
        return None


def ensure_model(name, installed):
    names = {m.get("name") for m in installed}
    if any(name == n or n.startswith(name + ":") for n in names):
        return True
    log(f"    模型 {name} 未拉取，开始下载（可能数分钟）…")
    try:
        api_post("/api/pull", {"name": name}, timeout=3600)
        return True
    except Exception as e:
        log(f"    拉取失败: {e}")
        return False


def bench_once(model):
    """单次测量：TTFT + 生成速度（流式）"""
    payload = {"model": model, "prompt": PROMPT, "stream": True}
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    gen_tokens = 0
    eval_ns = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ttft is None and d.get("response"):
                ttft = time.time() - t0
            if d.get("done"):
                gen_tokens = d.get("eval_count", 0) or 0
                eval_ns = d.get("eval_duration", 0) or 0
                break
    total = time.time() - t0
    tps = (gen_tokens / (eval_ns / 1e9)) if eval_ns else 0
    return {"ttft": ttft or total, "total": total, "tokens": gen_tokens, "tps": tps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=int, default=3, help="每个模型测试次数，取中位数")
    ap.add_argument("--out", default=None, help="结果写入 markdown 文件")
    args = ap.parse_args()

    log("=" * 60)
    log("Ollama 速度实测")
    log("=" * 60)

    installed = check_server()
    if installed is None:
        log("\n❌ 连不上 Ollama（localhost:11434）")
        log("   请先启动 Ollama，或运行 OllamaSetup.exe 完成安装")
        return 1
    log(f"✅ Ollama 已运行，本机已有 {len(installed)} 个模型")
    for m in installed:
        log(f"   - {m.get('name')}")

    results = {}
    for model in args.models:
        log("")
        log(f"── 测试 {model} ──")
        if not ensure_model(model, installed):
            continue
        runs = []
        for i in range(1, args.runs + 1):
            try:
                r = bench_once(model)
            except Exception as e:
                log(f"    第 {i} 次失败: {type(e).__name__}: {e}")
                continue
            runs.append(r)
            log(
                f"    第 {i} 次  TTFT {r['ttft']:5.2f}s | "
                f"{r['tps']:5.2f} tok/s | {r['tokens']:4d} tokens | 总 {r['total']:5.2f}s"
            )
        if not runs:
            continue
        runs.sort(key=lambda x: x["tps"])
        med = runs[len(runs) // 2]
        results[model] = med

        # 250 字中文 ≈ 380 tokens（中文 1 字约 1.5 token）
        est = 380 / med["tps"] if med["tps"] else 0
        log(f"    中位数  TTFT {med['ttft']:.2f}s | {med['tps']:.2f} tok/s")
        log(f"    → 生成 250 字中文约需 {est:.0f} 秒")

    log("")
    log("=" * 60)
    log("结论")
    log("=" * 60)
    if not results:
        log("无有效结果")
        return 1
    for model, r in sorted(results.items(), key=lambda x: -x[1]["tps"]):
        est = 380 / r["tps"] if r["tps"] else 0
        log(f"  {model:<16} {r['tps']:5.2f} tok/s   "
            f"首字 {r['ttft']:5.2f}s   250字约 {est:5.1f}s")
    best = max(results.items(), key=lambda x: x[1]["tps"])
    log(f"\n  建议默认模型: {best[0]}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Ollama 本地推理实测\n", "| 模型 | 生成速度 | 首字延迟 | 250字中文耗时 |",
                 "|---|---|---|---|"]
        for model, r in sorted(results.items(), key=lambda x: -x[1]["tps"]):
            est = 380 / r["tps"] if r["tps"] else 0
            lines.append(
                f"| {model} | {r['tps']:.2f} tok/s | {r['ttft']:.2f}s | {est:.1f}s |"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"\n  结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
