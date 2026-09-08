# -*- coding: utf-8 -*-
"""
问答编排层：检索 → 组装 Prompt → 流式生成（或降级）。

这一层存在的意义：把"检索"和"生成"解耦。
    - app.py 只管画界面，不知道 ChromaDB 和 Ollama 的存在
    - 换成别的 LLM（vLLM / llama.cpp）只需改 core/llm.py
    - 评估脚本可以只调 retrieve()，不触发生成，跑得飞快

三个真实问题在这里处理：
    1. 上下文塞不下：3b 模型 num_ctx 只有 4096，5 条资料每条必须截断
    2. 小模型爱编造：system prompt 强制"只用资料、找不到就说找不到"
    3. LLM 挂了不能白屏：fallback() 返回纯检索结果的结构化摘要
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import DEFAULT_MODEL, GenStat, LLMError, Ollama, pick_model  # noqa: E402
from core.retriever import build_index, load_zh2en, route, search  # noqa: E402

# 每条资料最多送多少字进上下文。3b 模型 num_ctx=4096，
# 5 条 × 500 字 + prompt 骨架 ≈ 3000 token，留足生成空间。
SNIPPET_MAX = 500
MAX_HITS_IN_PROMPT = 5

SYSTEM_PROMPT = """你是《艾尔登法环》的中文攻略助手，代号 NaviRAG。

严格规则：
1. 只使用【资料】中给出的信息回答。资料里没有的数值、道具名、地点名，绝对不要编造。
2. 用简体中文回答，简洁直接，不要复述问题，不要说"根据资料"这类废话。
3. 讲打法时用「需要 → 步骤 → 翻车点」的顺序，步骤用数字列表。
4. 如果资料不足以回答问题，直接说明"资料里没找到"，然后列出资料中最相关的条目名称和一句话摘要。
5. 不要超出资料范围推荐配装、加点或剧情路线。"""

INTENT_HINT = {
    "brain": "用户明确想要最省事/最无脑的打法。请优先挑无脑度最高的那条，并说明为什么它最省事。",
    "cheese": "用户想逃课。请优先给出逃课打法（L2 层资料），把「需要什么」和「具体怎么做」讲清楚。",
    "level": "用户关心当前等级能不能打。请说明建议等级、等级不够时的替代方案。",
    "normal": "用户想了解常规攻略信息。请综合资料给出准确、有条理的说明。",
}


class QASystem:
    """持有检索资源 + LLM 客户端，对外只暴露 retrieve / stream / fallback。"""

    def __init__(self, model_name: str | None = None, host: str | None = None):
        from core.llm import DEFAULT_HOST
        self.host = host or DEFAULT_HOST
        self.llm = Ollama(host=self.host, model=model_name or DEFAULT_MODEL)
        self.model_name = model_name or pick_model(self.host) or DEFAULT_MODEL
        self.llm.model = self.model_name

        self._col = None
        self._emb = None
        self._zh2en = None
        self.index_error = ""

    # ------------------------------------------------------------ 资源

    def load(self):
        """加载向量库和 embedding 模型（首次较慢，调用方应缓存）。"""
        if self._col is not None:
            return
        try:
            _client, self._col, self._emb = build_index()
            self._zh2en = load_zh2en()
        except Exception as e:  # 索引损坏 / 模型缺失，都要给出可读原因
            self.index_error = f"{type(e).__name__}: {e}"
            raise

    @property
    def ready(self) -> bool:
        return self._col is not None

    def count(self) -> int:
        return self._col.count() if self._col else 0

    # ------------------------------------------------------------ 检索

    def retrieve(self, query: str, k: int = 6) -> tuple[list[dict], dict]:
        self.load()
        hits, diag = search(self._col, self._emb, query, k=k, zh2en=self._zh2en)
        return hits, diag

    # ------------------------------------------------------------ Prompt

    @staticmethod
    def _format_hit(i: int, h: dict) -> str:
        m = h["meta"]
        parts = [f"[{i}]"]
        if m["layer"] == "l2":
            parts.append(f"★逃课打法｜目标：{m['target'] or '通用'}")
            parts.append(f"打法：{m['title_zh'] or m['title']}")
            parts.append(f"无脑度：{m['brain_level']}/5")
            if m["level_req"]:
                parts.append(f"建议等级：{m['level_req']}")
            parts.append("状态：本人已验证" if m["verified"] else "状态：未验证")
        else:
            label = m["title_zh"] or m["title"] or m["category"]
            parts.append(f"攻略条目：{label}")
            if m["url"]:
                parts.append(f"来源：{m['url']}")
        head = "｜".join(parts)
        body = h["text"].strip().replace("\n", " ")
        if len(body) > SNIPPET_MAX:
            body = body[:SNIPPET_MAX] + "…"
        return f"{head}\n{body}"

    def build_messages(self, query: str, hits: list[dict], diag: dict) -> tuple[str, str]:
        """返回 (system, user_prompt)"""
        if not hits:
            return SYSTEM_PROMPT, (
                f"【问题】{query}\n\n【资料】（无）\n\n"
                "资料库里没有任何相关内容。请直接说明没找到，"
                "不要编造答案，并建议用户换一个说法或换个 boss 名再问。"
            )

        blocks = [
            self._format_hit(i, h)
            for i, h in enumerate(hits[:MAX_HITS_IN_PROMPT], 1)
        ]
        hint = INTENT_HINT.get(diag.get("intent", "normal"), "")
        user = f"【问题】{query}\n\n"
        if hint:
            user += f"{hint}\n\n"
        user += "【资料】\n" + "\n\n".join(blocks)
        user += "\n\n【要求】用资料回答问题，控制在 300 字以内。"
        return SYSTEM_PROMPT, user

    # ------------------------------------------------------------ 生成

    def stream(self, query: str, hits: list[dict], diag: dict,
               options: dict | None = None) -> Iterator[str]:
        """流式生成。Ollama 不可用时自动切 fallback（作为整段文本 yield 一次）。"""
        system, user = self.build_messages(query, hits, diag)
        try:
            for chunk in self.llm.stream(user, system=system, options=options):
                yield chunk
        except LLMError as e:
            yield self.fallback(query, hits, diag, reason=str(e))

    def fallback(self, query: str, hits: list[dict], diag: dict,
                 reason: str = "") -> str:
        """本地大模型不可用时的降级：直接把检索结果整理成可读文本。

        这不是"凑合"。RAG 系统里检索层才是核心资产——模型挂了还能给出
        准确的原文片段，比让小模型编一个听起来对的错误答案强得多。
        """
        if not hits:
            return ("⚠️ 本地大模型未启动，且资料库没有检索到相关内容。\n\n"
                    f"原因：{reason or '未知'}\n"
                    "建议：先启动 Ollama（运行 `ollama serve`），或换个说法再问。")

        lines = [
            "⚠️ **本地大模型不可用，已降级为「纯检索模式」** —— "
            "以下是从资料库原文检索到的内容，未经模型整理，但都是原文，可信。",
            f"\n> 降级原因：{reason or 'Ollama 未响应'}",
            f"> 路由：{diag.get('intent', 'normal')}｜"
            f"召回 L1 {diag.get('n_l1', 0)} 条 / L2 {diag.get('n_l2', 0)} 条\n",
        ]
        for i, h in enumerate(hits[:5], 1):
            m = h["meta"]
            if m["layer"] == "l2":
                head = (f"**{i}. ★{m['title_zh'] or m['title']}**"
                        f"（目标：{m['target'] or '通用'}｜"
                        f"无脑度 {m['brain_level']}/5"
                        + (f"｜建议 {m['level_req']} 级" if m["level_req"] else "")
                        + "）")
            else:
                head = f"**{i}. {m['title_zh'] or m['title'] or m['category']}**"
            body = h["text"].strip().replace("\n", " ")
            if len(body) > 320:
                body = body[:320] + "…"
            lines.append(f"{head}\n{body}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------ 诊断

    def sources(self, hits: list[dict]) -> list[dict]:
        """给 UI 用的来源卡片数据"""
        out = []
        for h in hits:
            m = h["meta"]
            out.append({
                "layer": m["layer"],
                "title": m["title_zh"] or m["title"] or m["category"] or "(无标题)",
                "target": m.get("target", ""),
                "brain_level": m.get("brain_level", 0),
                "level_req": m.get("level_req", 0),
                "verified": m.get("verified", False),
                "url": m.get("url", ""),
                "score": round(h["score"], 3),
                "text": h["text"],
            })
        return out
