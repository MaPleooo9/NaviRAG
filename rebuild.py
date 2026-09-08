# -*- coding: utf-8 -*-
"""
NaviRAG 攻略库更新入口：md → jsonl → 重建向量索引。

为什么需要这个入口（方案 A 的最后一环）：
    打包成 exe 后，data/ 和 .vectorstore/ 外置在 exe 旁边。
    想加新攻略：改 data/elden_ring/ 下的 md → 跑一次更新 → 新内容即刻可问。
    全程不需要 Python 环境、不需要重新打包。

用法:
    python rebuild.py               # 开发环境
    NaviRAG.exe --rebuild           # 打包后（桌面端检测到参数后调用 rebuild()）
"""
import os
import time

# 冻结环境（PyInstaller）里 torch 常因 OpenMP 运行时重复加载在首次推理时原生崩溃，
# 必须在 torch 被导入前设置（rebuild 被调用时 torch 还没进内存，这里正好赶上）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from core.retriever import DATA, MODEL_NAME, build_index


def rebuild(log=print) -> dict:
    """完整更新流程，返回摘要 dict。任何一步失败直接抛异常，由调用方展示。"""
    from scripts.build_l2 import convert as conv_l2
    from scripts.build_l3 import convert as conv_l3

    t0 = time.time()
    problems = []

    log("① 解析 L2 逃课攻略（Markdown → JSONL）…")
    try:
        _, p = conv_l2(DATA / "l2_cheese.md", DATA / "l2_cheese.jsonl", log)
        problems += p
    except FileNotFoundError as e:
        log(f"  跳过：{e}")

    log("② 解析 L3 常规打法（Markdown → JSONL）…")
    try:
        _, p = conv_l3(DATA / "l3_normal.md", DATA / "l3_normal.jsonl", log)
        problems += p
    except FileNotFoundError as e:
        log(f"  跳过：{e}")

    for p in problems:
        log(f"  ⚠️ {p}")

    log("③ 重建向量索引（全量重嵌，约 1-3 分钟）…")
    # 先单独加载模型再建索引：模型加载和批量编码分开记日志，
    # 冻结环境若在这两步崩，日志能直接定位是哪一步。
    log("  ③-1 加载 embedding 模型…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    log("  ③-2 模型就绪，开始全量编码…")
    _client, col, _model = build_index(force=True, model=model)
    n = col.count()

    log(f"✅ 更新完成：索引 {n:,} 条，用时 {time.time() - t0:.0f} 秒。")
    log("   新攻略已生效，直接提问即可。")
    return {"count": n, "problems": problems, "elapsed": time.time() - t0}


if __name__ == "__main__":
    rebuild()
