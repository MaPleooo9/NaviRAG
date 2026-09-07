# -*- coding: utf-8 -*-
"""
手动下载 embedding 模型到本地，绕过 huggingface_hub 客户端。

背景：hf-mirror 镜像与新版 huggingface_hub 的 Xet 协议不兼容，
python 客户端下载会把空文件写进缓存（JSONDecodeError: Expecting value）。
但镜像的 HTTP resolve 直链本身是好的——所以直接用 urllib 拉。

产出：models/paraphrase-multilingual-MiniLM-L12-v2/（SentenceTransformer 可直接加载）

用法:
    python scripts/download_model.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
BASE = ("https://hf-mirror.com/sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2/resolve/main/")

FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "1_Pooling/config.json",
    "model.safetensors",   # ~470MB，放最后
]

# 可选文件：404 不算失败（不同模型仓库结构略有差异）
OPTIONAL = {"vocab.txt"}

MIN_SIZE = {"model.safetensors": 400_000_000}   # 大文件校验


def log(m):
    print(m, flush=True)


def fetch(name, retries=4):
    url = BASE + name
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= MIN_SIZE.get(name, 20):
        log(f"  ✓ 已存在 {name} ({dest.stat().st_size:,} B)")
        return True

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        log(f"    {name}: {done * 100 // total}%")
            # JSON 文件校验有效性（镜像偶尔返回错误页）
            if name.endswith(".json") and dest.stat().st_size < 10_000:
                json.loads(dest.read_text(encoding="utf-8"))   # 会抛异常
            if dest.stat().st_size < MIN_SIZE.get(name, 20):
                raise RuntimeError(f"文件过小: {dest.stat().st_size} B")
            log(f"  ✓ {name} ({dest.stat().st_size:,} B)")
            return True
        except Exception as e:
            last = e
            dest.unlink(missing_ok=True)      # 删掉半残文件
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                log(f"    重试 {attempt + 1}: {e}")
    log(f"  ✗ {name} 失败: {last}")
    return False


def main():
    log("=" * 60)
    log(f"下载 embedding 模型 → {DEST}")
    log("=" * 60)
    ok = True
    for f in FILES:
        if not fetch(f):
            if f in OPTIONAL:
                log(f"    （{f} 为可选文件，跳过）")
            else:
                ok = False
    if not ok:
        log("\n❌ 有文件下载失败，重跑本脚本会断点续传")
        return 1
    log(f"\n✅ 模型就绪: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
