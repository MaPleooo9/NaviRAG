# -*- coding: utf-8 -*-
"""
NaviRAG 桌面端：PySide6（Qt）原生窗口，不是网页。

为什么单独做一个桌面壳而不是复用 app.py：
    app.py（Streamlit）是网页形态，演示"产品"用；这个桌面窗口是
    "工具"形态——双击就开、离线可用、不占浏览器。两层共用同一个
    编排层 core/qa.py，检索 / 生成 / 降级逻辑完全一致，只换壳。

用法:
    python desktop_chat.py        # 或双击 run_desktop.bat

架构（线程模型，面试可讲）:
    主线程  —— Qt 事件循环，只做渲染（QTextBrowser 追加 HTML）
    后台线程 —— 加载索引 / 检索 / Ollama 流式请求，通过 Qt Signal
               把结果推回主线程（跨线程 Signal 是队列投递，安全）
    好处：模型生成 30 秒窗口也不卡死；关窗即退，无残留进程。
"""
from __future__ import annotations

import html
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget,
)

from core.qa import QASystem  # noqa: E402

LAYER_TAG = {"l1": "L1攻略", "l2": "L2逃课", "l3": "L3常规"}


# ---------------------------------------------------------------- 渲染工具

def md_lite(text: str) -> str:
    """模型输出的轻量 markdown → HTML（粗体 / 三级标题 / 换行）。

    只处理这三样，因为 system prompt 就约束了输出是
    「短段落 + 数字步骤 + 粗体字段」，多了反而是攻击面。
    """
    s = html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?m)^#{1,3}\s*(.+)$", r"<b>\1</b>", s)
    return s.replace("\n", "<br>")


def sources_html(sources: list[dict]) -> str:
    """来源列表 → 灰色小字 HTML。RAG 不亮出检索原文 = 让用户盲信模型。"""
    if not sources:
        return ""
    rows = []
    for i, s in enumerate(sources, 1):
        tag = LAYER_TAG.get(s["layer"], s["layer"])
        head = f"{i}. [{tag}] {s['title']}"
        extras = []
        if s["target"]:
            extras.append(f"目标 {s['target']}")
        if s["layer"] == "l2":
            extras.append(f"无脑度 {s['brain_level']}/5")
        if s["layer"] == "l3":
            extras.append(f"难度 {s['difficulty']}/5")
        if s["level_req"]:
            extras.append(f"建议 {s['level_req']} 级")
        extras.append("✅已验证" if s["verified"] else "⚠️未验证")
        extras.append(f"得分 {s['score']}")
        rows.append(f"{head}　" + "｜".join(extras))
    body = "<br>".join(html.escape(r) for r in rows)
    return (f'<span style="color:#8a8a8a;font-size:9pt;">'
            f"📚 检索来源<br>{body}</span>")


# ---------------------------------------------------------------- 后台桥

class Bridge(QObject):
    """后台线程 → 主线程 的信号桥。跨线程 emit 走队列投递，安全。"""

    status = Signal(str)      # 状态栏文本
    chunk = Signal(str)       # 生成片段
    sources = Signal(str)     # 来源 HTML
    busy = Signal(bool)       # 输入区是否锁定
    banner = Signal(str)      # 聊天区直接追加的 HTML


def make_qa() -> QASystem:
    qa = QASystem()
    qa.load()
    return qa


def warmup_thread(qa: QASystem, bridge: Bridge) -> None:
    """启动预热：加载索引 + 让模型常驻（keep_alive=-1），不阻塞 UI。"""
    bridge.status.emit("正在加载向量库与 embedding 模型…")
    try:
        qa.load()
    except Exception as e:
        bridge.status.emit(f"❌ 索引加载失败：{e}")
        return
    online = qa.llm.available()
    if online:
        bridge.status.emit("正在预热本地模型（首次约 6 秒）…")
        try:
            qa.llm.warmup()
        except Exception:
            pass  # 预热失败不致命，首问会自然慢一次
    layers = qa.layer_counts()
    lay_txt = " / ".join(f"{k.upper()} {v:,}" for k, v in layers.items())
    state = "● 在线" if online else "○ 离线（将降级为纯检索模式）"
    bridge.status.emit(
        f"向量库 {qa.count():,} 条（{lay_txt}）｜ 模型 {qa.model_name} ｜ Ollama {state}"
    )
    bridge.banner.emit(
        "<span style='color:#8a8a8a;'>"
        "试试：「玛莲妮亚怎么逃课」「玛莲妮亚怎么打（不逃课）」「我 60 级能打大树守卫吗」"
        "「玛莲妮亚的追忆能换什么」</span><br>"
    )


def ask_thread(qa: QASystem, bridge: Bridge, query: str) -> None:
    """一次问答：检索 → 流式生成 → 来源。任何一步失败都不白屏。"""
    try:
        hits, diag = qa.retrieve(query, k=6)
        intent = diag.get("intent", "normal")
        bridge.chunk.emit(f"\n<b>你</b>：{html.escape(query)}\n")
        bridge.chunk.emit(
            f"<span style='color:#8a8a8a;font-size:9pt;'>"
            f"路由 {intent}｜召回 L1 {diag.get('n_l1', 0)} / "
            f"L2 {diag.get('n_l2', 0)} / L3 {diag.get('n_l3', 0)}</span><br>"
        )
        n = 0
        for c in qa.stream(query, hits, diag):
            bridge.chunk.emit(html.escape(c).replace("\n", "<br>"))
            n += len(c)
        if n == 0:
            bridge.chunk.emit("<span style='color:#b05050;'>（模型没有返回内容）</span>")
        bridge.sources.emit(sources_html(qa.sources(hits)))
        bridge.chunk.emit("<hr>")
    except Exception as e:
        bridge.chunk.emit(
            f"<span style='color:#b05050;'>出错了：{html.escape(str(e))}<br>"
            "请确认 Ollama 正在运行，或重试。</span><hr>"
        )
    finally:
        bridge.busy.emit(False)


# ---------------------------------------------------------------- 主窗口

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.qa = QASystem()
        self.bridge = Bridge()
        self.bridge.chunk.connect(self._append)
        self.bridge.sources.connect(self._append)
        self.bridge.banner.connect(self._append)
        self.bridge.busy.connect(self._set_busy)

        self.setWindowTitle("NaviRAG · 本地攻略问答")
        self.resize(780, 660)
        self._build_ui()

        threading.Thread(target=warmup_thread, args=(self.qa, self.bridge),
                         daemon=True).start()

    # ---------------- UI

    def _build_ui(self):
        font = QFont("Microsoft YaHei UI", 10)
        self.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        layout.addWidget(self.chat, 1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("输入问题，回车发送…")
        self.input.returnPressed.connect(self.send)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        layout.addLayout(row)

        self.status = QLabel("正在启动…")
        self.status.setStyleSheet("color:#666;font-size:9pt;")
        layout.addWidget(self.status)

    # ---------------- 槽

    def _append(self, html_text: str):
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertHtml(html_text)
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _set_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.send_btn.setText("回答中…" if busy else "发送")
        if not busy:
            self.input.setFocus()

    # ---------------- 动作

    def send(self):
        text = self.input.text().strip()
        if not text:
            return
        if not self.qa.ready:
            self._append("<span style='color:#b05050;'>索引还没加载完，稍等几秒。</span><br>")
            return
        self.input.clear()
        self._set_busy(True)
        threading.Thread(target=ask_thread, args=(self.qa, self.bridge, text),
                         daemon=True).start()


def main():
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
