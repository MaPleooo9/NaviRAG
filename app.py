# -*- coding: utf-8 -*-
"""
NaviRAG · 本地游戏攻略 RAG 问答（Streamlit 界面）

运行:
    streamlit run app.py

设计取舍：
    - 界面只做三件事：拿问题、流式显示答案、把检索来源摊开给你看。
      第三件是本项目最能体现工程性的地方——RAG 系统如果不敢把来源亮出来，
      就等于让用户盲信一个会编造的小模型。
    - 所有重资源（向量库、embedding 模型）用 st.cache_resource 持有，
      避免 Streamlit 每次交互都重新加载（一次 20 秒，谁都受不了）。
    - 本地大模型用 Ollama；连不上时自动降级为纯检索模式，绝不白屏。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.llm import LLMError  # noqa: E402
from core.qa import QASystem  # noqa: E402
from core.retriever import route  # noqa: E402

# ---------------------------------------------------------------- 页面配置

st.set_page_config(
    page_title="NaviRAG · 艾尔登法环攻略助手",
    page_icon="🗡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; max-width: 1180px; }
  .src-card {
      border-left: 3px solid #d4a24a; padding: 0.55rem 0.9rem; margin: 0.5rem 0;
      background: rgba(212,162,74,0.06); border-radius: 0 6px 6px 0; font-size: 0.9rem;
  }
  .src-card.l1 { border-left-color: #6b8fbf; background: rgba(107,143,191,0.06); }
  .src-card.l3 { border-left-color: #7bb07b; background: rgba(123,176,123,0.06); }
  .src-title { font-weight: 600; margin-bottom: 0.2rem; }
  .src-meta { color: #8a8a8a; font-size: 0.8rem; margin-bottom: 0.35rem; }
  .src-body { color: #d0d0d0; line-height: 1.6; white-space: pre-wrap; }
  .diag { color: #8a8a8a; font-size: 0.78rem; font-family: ui-monospace, Consolas, monospace; }
  .hero { color: #8a8a8a; font-size: 0.92rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "玛莲妮亚怎么逃课？",
    "玛莲妮亚怎么打（不逃课）",
    "拉塔恩怎么打",
    "黑剑玛利喀斯怎么打",
    "给我最无脑的打法",
    "我 60 级能打大树守卫吗",
]


# ---------------------------------------------------------------- 资源加载

@st.cache_resource(show_spinner=False)
def get_qa() -> QASystem:
    """全局唯一的 QASystem（跨会话复用，避免重复加载 20 秒的模型）。"""
    qa = QASystem()
    qa.load()
    return qa


@st.cache_resource(show_spinner=False)
def warmup_once(_qa: QASystem) -> float:
    """首次把模型加载进内存。返回耗时（秒），失败返回 -1。"""
    return _qa.llm.warmup()


def boot():
    """首次运行时的初始化：加载索引 → 预热模型。结果存 session_state。"""
    if st.session_state.get("booted"):
        return True

    with st.spinner("正在加载向量索引与本地模型（首次约 20-40 秒，之后全程秒开）…"):
        try:
            qa = get_qa()
        except Exception as e:
            st.session_state.boot_error = str(e)
            return False

        st.session_state.qa = qa
        t0 = time.time()
        warm = warmup_once(qa)
        st.session_state.warm_sec = warm
        st.session_state.boot_sec = time.time() - t0
        st.session_state.booted = True
    return True


# ---------------------------------------------------------------- 渲染

def render_sources(sources: list[dict], diag: dict):
    """把检索来源摊开给用户看——RAG 系统的诚实性全靠这一块。"""
    intent = diag.get("intent", "normal")
    intent_zh = {"cheese": "逃课意图", "brain": "求最无脑", "level": "等级过滤",
                 "knowledge": "知识问答", "normal": "常规攻略"}.get(intent, intent)
    lv = f"｜等级上限 {diag['max_level']}" if diag.get("max_level") else ""
    st.markdown(
        f'<p class="diag">路由 {intent_zh}｜权重 L1 {diag["w_l1"]} / '
        f'L2 {diag["w_l2"]} / L3 {diag["w_l3"]}{lv}｜'
        f'召回 L1 {diag["n_l1"]} 条 / L2 {diag["n_l2"]} 条 / '
        f'L3 {diag["n_l3"]} 条</p>',
        unsafe_allow_html=True,
    )

    with st.expander(f"📚 检索来源（{len(sources)} 条）", expanded=False):
        for i, s in enumerate(sources, 1):
            if s["layer"] == "l2":
                stars = "★" * s["brain_level"] + "☆" * (5 - s["brain_level"])
                meta = f"L2 逃课 · 无脑度 {stars}"
                cls = "src-card"
            elif s["layer"] == "l3":
                stars = "▲" * s["difficulty"] + "△" * (5 - s["difficulty"])
                meta = f"L3 常规 · 难度 {stars}"
                cls = "src-card l3"
            else:
                meta = "L1 攻略"
                cls = "src-card l1"

            if s["layer"] in ("l2", "l3"):
                if s["level_req"]:
                    meta += f" · 建议 {s['level_req']} 级"
                meta += f" · {'✅ 已验证' if s['verified'] else '⚠️ 未验证'}"
                if s["target"]:
                    meta += f" · 目标 {s['target']}"
            elif s["url"]:
                meta += f" · [原文]({s['url']})"
            meta += f" · 得分 {s['score']}"

            body = s["text"].strip()
            if len(body) > 420:
                body = body[:420] + "…"
            st.markdown(
                f'<div class="{cls}">'
                f'<div class="src-title">{i}. {s["title"]}</div>'
                f'<div class="src-meta">{meta}</div>'
                f'<div class="src-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


def sidebar(qa: QASystem | None):
    with st.sidebar:
        st.markdown("## ⚙️ 控制台")

        # ---- 系统状态
        st.markdown("**系统状态**")
        if qa is None:
            st.error("检索层未就绪")
        else:
            counts = qa.layer_counts()
            st.success(
                f"向量库 {qa.count():,} 条  \n"
                f"L1 攻略 {counts.get('l1', 0):,} ｜ "
                f"L2 逃课 {counts.get('l2', 0)} ｜ "
                f"L3 常规 {counts.get('l3', 0)}"
            )
            llm_ok = qa.llm.available()
            if llm_ok:
                warm = st.session_state.get("warm_sec", -1)
                wtxt = f" · 预热 {warm:.1f}s" if warm and warm > 0 else ""
                st.success(f"Ollama 在线 · {qa.model_name}{wtxt}")
            else:
                st.warning("Ollama 离线 → 纯检索模式")
                st.caption("启动方式：终端执行 `ollama serve`")

        st.divider()

        # ---- 检索参数
        st.markdown("**检索参数**")
        topk = st.slider("返回条数", 3, 10, 6,
                         help="送给大模型和展示在来源里的条目数")
        min_brain = st.slider(
            "最低无脑度", 0, 5, 0,
            help=">0 时只保留无脑度达到该值的逃课打法（L1 攻略不受影响）")
        show_src = st.checkbox("显示检索来源", value=True)

        st.divider()

        # ---- 示例问题
        st.markdown("**试试这些问题**")
        for q in EXAMPLES:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.pending = q

        st.divider()
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "NaviRAG · 本地 RAG 攻略问答\n\n"
            "L1 英文 wiki + 官方中文语料 ｜ L2 人工逃课打法\n\n"
            "全部推理在本机完成，不联网、不上传。"
        )
        return topk, min_brain, show_src


# ---------------------------------------------------------------- 主流程

def main():
    st.markdown("# 🗡️ NaviRAG")
    st.markdown(
        '<p class="hero">本地运行的《艾尔登法环》攻略问答 · '
        '分层检索（通用攻略 + 人工逃课打法）· 答案附带可核对的原文来源</p>',
        unsafe_allow_html=True,
    )

    if not boot():
        st.error(
            "检索层加载失败，无法启动。\n\n"
            f"```\n{st.session_state.get('boot_error', '未知错误')}\n```\n\n"
            "常见原因：数据文件缺失（先跑 `python scripts/build_l2.py`），"
            "或向量索引损坏（重建：`python -c \"import sys;sys.path.insert(0,'core');"
            "from retriever import build_index;build_index(force=True)\"`）。"
        )
        return

    qa: QASystem = st.session_state.qa
    topk, min_brain, show_src = sidebar(qa)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_src and msg.get("sources"):
                render_sources(msg["sources"], msg["diag"])

    # 输入：聊天框 or 侧边栏示例
    user_q = st.chat_input("想问哪个 boss？比如「玛莲妮亚怎么逃课」")
    if not user_q and st.session_state.get("pending"):
        user_q = st.session_state.pop("pending")

    if not user_q:
        return

    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner("检索中…"):
            try:
                hits, diag = qa.retrieve(user_q, k=topk)
            except Exception as e:
                st.error(f"检索失败：{type(e).__name__}: {e}")
                return

        # 无脑度过滤：只作用于 L2，且全部被过滤掉时自动放宽，避免给出空答案
        if min_brain > 0:
            filtered = [h for h in hits
                        if h["meta"]["layer"] != "l2"
                        or h["meta"]["brain_level"] >= min_brain]
            if filtered:
                hits = filtered

        gen = qa.stream(user_q, hits, diag)
        answer = st.write_stream(gen)

        sources = qa.sources(hits)
        if show_src:
            render_sources(sources, diag)
        st.caption(f"耗时 {time.time() - t0:.1f}s · {qa.model_name}")

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "diag": diag,
    })


if __name__ == "__main__":
    main()
