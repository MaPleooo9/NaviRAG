# -*- coding: utf-8 -*-
"""
Ollama 本地大模型客户端。

三个必须自己解决的问题（直接用裸 API 会踩）：
    1. 冷启动慢：第一次请求要把 2GB 模型加载进显存/内存，实测 31.75s，
       之后同样请求只要 0.18s。解法是启动时 warmup 一次 + keep_alive=-1 常驻。
    2. 必须流式：用户等 30 秒再一次性看到答案 = 体感崩溃。逐 token yield 出来，
       首字延迟只有 1-2 秒。
    3. 必须可降级：Ollama 没开 / 模型没拉 / 生成超时，都不能让页面白屏，
       要有明确的降级路径（见 core/qa.py 的 fallback）。

用法:
    from core.llm import Ollama
    llm = Ollama()
    if llm.available():
        for chunk in llm.stream("你好"):
            print(chunk, end="")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"

# 生成参数：攻略问答要的是"准确复述资料"而不是"自由发挥"，
# 所以温度压到 0.3；num_predict 限长，防止 3b 模型绕圈停不下来。
DEFAULT_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_predict": 600,
    "num_ctx": 4096,
    "repeat_penalty": 1.1,
}


class LLMError(RuntimeError):
    """Ollama 不可用或生成失败。调用方应捕获并降级。"""


@dataclass
class GenStat:
    """一次生成的性能指标，用于在 UI 上显示和做评估。"""

    ttft: float = 0.0        # 首字延迟（秒）
    total: float = 0.0       # 总耗时（秒）
    tokens: int = 0          # 生成 token 数
    tps: float = 0.0         # 生成速度（token/秒）
    model: str = ""
    fallback: bool = False   # 是否走了降级路径
    note: str = ""
    elapsed: float = field(default=0.0)

    def as_text(self) -> str:
        if self.fallback:
            return f"降级模式 · {self.note}"
        return (f"{self.model} · 首字 {self.ttft:.1f}s · "
                f"{self.tps:.1f} tok/s · 共 {self.tokens} tokens / {self.total:.1f}s")


class Ollama:
    """极简 Ollama 客户端，只依赖 requests。"""

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
                 timeout: tuple[float, float] = (5, 300), keep_alive: int = -1):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        # keep_alive=-1：模型常驻内存不卸载。默认 5 分钟就卸载，
        # 用户聊两句去上个厕所回来又要冷启动 30 秒。
        self.keep_alive = keep_alive
        self._session = requests.Session()

    # ------------------------------------------------------------ 健康检查

    def available(self, timeout: float = 3.0) -> bool:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def models(self) -> list[str]:
        """本机已拉取的模型名列表"""
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return []

    def has_model(self, name: str | None = None) -> bool:
        name = name or self.model
        return any(n == name or n.startswith(name + ":") for n in self.models())

    # ------------------------------------------------------------ 预热

    def warmup(self, model: str | None = None) -> float:
        """把模型加载进内存并返回耗时。冷启动一次，之后全程热。

        用 keep_alive=-1 发一个极短请求：模型会被加载并常驻。
        返回耗时（秒），失败返回 -1。
        """
        model = model or self.model
        t0 = time.time()
        try:
            r = self._session.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "hi", "stream": False,
                      "keep_alive": self.keep_alive,
                      "options": {"num_predict": 1}},
                timeout=(10, 600),
            )
            r.raise_for_status()
            return time.time() - t0
        except requests.RequestException:
            return -1.0

    # ------------------------------------------------------------ 生成

    def stream(self, prompt: str, system: str | None = None,
               options: dict | None = None,
               model: str | None = None) -> Iterator[str]:
        """流式生成，逐段 yield 文本。

        抛 LLMError 表示失败，调用方负责降级。

        think 参数（qwen3 系列支持）:
            None → 不发该字段，走 Ollama 默认（qwen3 默认开思考）
            False → 关闭思考，首字延迟从 ~8s 降到 <1s，适合攻略复述类任务
            True → 开思考，质量略升但整体耗时翻倍
            思考内容走独立字段，不会混进 response，客户端无需清洗。
        """
        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {**DEFAULT_OPTIONS, **(options or {})},
        }
        if system:
            payload["system"] = system
        think = (options or {}).pop("_think", None) if options else None
        if model.startswith("qwen3"):
            # qwen3 在 Ollama 里默认开思考，这里显式默认关闭（攻略复述不需要推理链）
            payload["think"] = bool(think) if think is not None else False

        try:
            r = self._session.post(f"{self.host}/api/generate", json=payload,
                                   stream=True, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"无法连接 Ollama（{self.host}）：{e}") from e

        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("error"):
                    raise LLMError(d["error"])
                if d.get("response"):
                    yield d["response"]
        finally:
            r.close()

    def generate(self, prompt: str, system: str | None = None,
                 options: dict | None = None,
                 model: str | None = None) -> tuple[str, GenStat]:
        """非流式生成，返回 (文本, 统计)。用于预热后的短回答和脚本调用。"""
        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {**DEFAULT_OPTIONS, **(options or {})},
        }
        if system:
            payload["system"] = system

        t0 = time.time()
        try:
            r = self._session.post(f"{self.host}/api/generate", json=payload,
                                   timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"无法连接 Ollama（{self.host}）：{e}") from e

        d = r.json()
        if d.get("error"):
            raise LLMError(d["error"])
        total = time.time() - t0
        tokens = d.get("eval_count", 0) or 0
        eval_ns = d.get("eval_duration", 0) or 0
        stat = GenStat(
            ttft=total, total=total, tokens=tokens,
            tps=(tokens / (eval_ns / 1e9)) if eval_ns else 0.0,
            model=model,
        )
        return d.get("response", "").strip(), stat


def pick_model(host: str = DEFAULT_HOST, prefer: list[str] | None = None) -> str:
    """在本机已安装的模型里挑一个能用的，优先 prefer 顺序。

    用户在自己机器上跑时，模型名可能不同（qwen2.5:3b / qwen3:8b / llama3.2 等），
    硬编码一个名字会直接开天窗，所以按优先级挑第一个存在的。
    环境变量 NAVIRAG_MODEL 优先级最高，方便不改代码切换模型做对比实验。
    """
    import os

    env = os.environ.get("NAVIRAG_MODEL")
    if env:
        return env
    prefer = prefer or ["qwen3:8b", "qwen2.5:3b", "qwen2.5:7b",
                        "qwen2.5:1.5b", "llama3.2", "llama3.1"]
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return prefer[0]

    for p in prefer:
        if any(n == p or n.startswith(p + ":") for n in installed):
            return p
    return installed[0] if installed else prefer[0]
