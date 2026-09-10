# -*- coding: utf-8 -*-
"""
NaviRAG 桌面端：PySide6 + QFluentWidgets（Win11 Fluent 风格），原生窗口，不是网页。

为什么单独做一个桌面壳而不是复用 app.py：
    app.py（Streamlit）是网页形态，演示"产品"用；这个桌面窗口是
    "工具"形态——双击就开、离线可用、不占浏览器。两层共用同一个
    编排层 core/qa.py，检索 / 生成 / 降级逻辑完全一致，只换壳。

用法:
    python desktop_chat.py        # 或双击 run_desktop.bat

架构（线程模型，面试可讲）:
    主线程  —— Qt 事件循环，只做渲染（卡片气泡更新）
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

LAYER_TAG = {"l1": "L1攻略", "l2": "L2逃课", "l3": "L3常规"}

WINDOW_TITLE = "NaviRAG · 本地攻略问答"
_MUTEX_DESKTOP = "NaviRAG.Desktop.SingleInstance"
_MUTEX_REBUILD = "NaviRAG.Rebuild.SingleInstance"


# ---------------------------------------------------------------- 单实例
# 注意这一节的**位置**：它刻意排在下面所有重依赖之前。
# 下面 from core.qa import QASystem 会连带把 torch / sentence-transformers /
# chromadb 全拉起来，冷启动十几秒。闸门若放在 main() 里，重复启动的第二个
# 进程要先白烧十几秒 CPU、占几百 MB 内存才走到检查；放在模块顶部，它不到
# 一秒就自己消失，用户只会看到已有窗口被唤到前台。

_guard_handles: list = []   # 必须持有互斥量句柄到进程结束，被 GC 回收 = 锁没了


def acquire_single_instance(name: str) -> bool:
    """Windows 命名互斥量做单实例检测，返回 True 表示"已经有同名实例在跑"。

    为什么用互斥量而不是锁文件:
        锁文件在进程崩溃后会残留，用户得手动删掉文件才能再启动；命名互斥量
        由内核持有，进程一退（正常关闭 / 任务管理器结束 / 崩溃）立即释放，
        零残留，也不依赖任何第三方库，打包成 exe 后同样有效。
    为什么不用 Qt 的 QLocalServer:
        那个方案必须在 QApplication 起来之后才能判断，会白等一次 Qt 初始化；
        互斥量在 QApplication 之前就拦住了，越早拦越省事。
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE

    ctypes.set_last_error(0)
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return False                    # 建不出来就放行，别把用户挡在门外
    _guard_handles.append(handle)       # 句柄活到进程结束，互斥量才一直有效
    return ctypes.get_last_error() == ERROR_ALREADY_EXISTS


def focus_existing_window(title: str, timeout: float = 8.0) -> bool:
    """把已有窗口从最小化还原并抢到前台；找不到就轮询等一会，返回是否成功。

    为什么要轮询而不是查一次:
        这正是"手快连点两次"的真实时序——第一次点击后窗口还没创建出来，
        第二次点击时 FindWindowW 就查不到。只查一次会退化成"弹个框告诉用户
        程序已经开了"，而这个框本身又要人去点，比不加还烦。轮询到窗口出现
        为止，"连点两次"的体验就只有"窗口跳到前面"这一个结果。
    抢前台在这里是允许的——本进程正是用户刚点击的前台进程。
    """
    if sys.platform != "win32":
        return False
    import ctypes
    import time as _time
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND

    deadline = _time.monotonic() + timeout
    while True:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.ShowWindow(hwnd, 9)      # SW_RESTORE：最小化状态也拉回来
            user32.SetForegroundWindow(hwnd)
            return True
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.2)


def _enforce_single_instance() -> None:
    """闸门：重复启动就在这里结束本进程，第二个窗口永远不会被创建。

    正常返回 = 拿到了锁，可以继续加载 torch / Qt 这些重依赖。
    重复启动 = 唤起已有窗口后直接 sys.exit 退出。
    """
    mutex = _MUTEX_REBUILD if "--rebuild" in sys.argv else _MUTEX_DESKTOP
    if not acquire_single_instance(mutex):
        return

    if "--rebuild" in sys.argv:
        # 更新任务没有窗口可以"唤起"，而且用户是专门双击 bat 来的，
        # 这时候弹框告知是合理的（这条路径才去碰 Qt，平时不白付导入成本）。
        from PySide6.QtWidgets import QApplication, QMessageBox
        _ = QApplication(sys.argv)
        QMessageBox.information(
            None, "更新正在进行",
            "已经有一个攻略更新任务在跑了。\n等它弹出完成提示后再试。")
        sys.exit(0)

    # 主界面重复启动：把已有窗口唤到前台就够了，不再多弹任何框。
    # 轮询到最后还是没找到（窗口标题被改 / 跨会话）也静默退出——那个窗口
    # 就在任务栏里，弹一个需要手动关闭的框反而比什么都不做更烦人。
    focus_existing_window(WINDOW_TITLE)
    sys.exit(0)


if __name__ == "__main__":
    # 必须早于 torch / Qt 的导入——放在这里，重复启动的进程秒退
    _enforce_single_instance()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QSettings, QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, FluentIcon, FluentWindow,
    IndeterminateProgressBar, InfoBar, LineEdit, PrimaryPushButton,
    PushButton, ScrollArea, SwitchButton, Theme, TitleLabel, isDarkTheme,
    setTheme,
)

from core.qa import QASystem  # noqa: E402

# ---------------------------------------------------------------- 渲染工具

def md_lite(text: str) -> str:
    """模型输出的轻量 markdown → 富文本（粗体 / 标题），QLabel 直接吃。"""
    s = html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?m)^#{1,3}\s*(.+)$", r"<b>\1</b>", s)
    return s.replace("\n", "<br>")


def sources_lines(sources: list[dict], with_url: bool = False) -> str:
    """来源列表 → 多行文本。with_url=True 时附原文链接。"""
    if not sources:
        return ""
    rows = []
    for i, s in enumerate(sources, 1):
        tag = LAYER_TAG.get(s["layer"], s["layer"])
        extras = []
        if s["layer"] == "l2":
            extras.append(f"无脑度 {s['brain_level']}/5")
        if s["layer"] == "l3":
            extras.append(f"难度 {s['difficulty']}/5")
        if s["level_req"]:
            extras.append(f"建议 {s['level_req']} 级")
        extras.append("✅已验证" if s["verified"] else "⚠️未验证")
        extras.append(f"得分 {s['score']}")
        line = f"{i}. [{tag}] {s['title']}　" + "｜".join(extras)
        if with_url and s.get("url"):
            line += f"　<a href='{html.escape(s['url'], quote=True)}'>原文</a>"
        rows.append(line)
    return "\n".join(rows)


# ---------------------------------------------------------------- 后台桥

class Bridge(QObject):
    """后台线程 → 主线程 的信号桥（跨线程 emit 走队列投递，安全）。"""

    status = Signal(str)          # 状态栏文本
    chunk = Signal(str)           # 生成片段（原始文本）
    answer_done = Signal(list)    # 回答结束，附来源 dict 列表
    busy = Signal(bool)           # 输入区是否锁定


def warmup_thread(qa: QASystem, bridge: Bridge) -> None:
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
            pass
    layers = qa.layer_counts()
    lay_txt = " / ".join(f"{k.upper()} {v:,}" for k, v in layers.items())
    state = "● 在线" if online else "○ 离线（降级为纯检索模式）"
    bridge.status.emit(
        f"向量库 {qa.count():,} 条（{lay_txt}）｜ 模型 {qa.model_name} ｜ Ollama {state}"
    )


def ask_thread(qa: QASystem, bridge: Bridge, query: str) -> None:
    try:
        hits, diag = qa.retrieve(query, k=6)
        bridge.chunk.emit("")  # 触发创建回答气泡
        n = 0
        for c in qa.stream(query, hits, diag):
            bridge.chunk.emit(c)
            n += len(c)
        if n == 0:
            bridge.chunk.emit("（模型没有返回内容）")
        bridge.answer_done.emit(qa.sources(hits))
    except Exception as e:
        bridge.chunk.emit(f"出错了：{e}\n请确认 Ollama 正在运行，或重试。")
        bridge.answer_done.emit([])
    finally:
        bridge.busy.emit(False)


# ---------------------------------------------------------------- 聊天气泡

class Bubble(CardWidget):
    """一条消息卡片。role=user 右对齐主题色，role=bot 默认卡片。"""

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.setFixedWidth(600)
        self.label = BodyLabel(self)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse
                                           | Qt.LinksAccessibleByMouse)
        self.label.setOpenExternalLinks(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self.label)
        dark = isDarkTheme()
        if role == "user":
            bg = "#0078d4" if not dark else "#2b6cb8"
            self.setStyleSheet(f"CardWidget{{background:{bg};border-radius:8px;}}")
            self.label.setStyleSheet("color:white;font-size:14px;")
        else:
            self.setStyleSheet("CardWidget{border-radius:8px;}")
            self.label.setStyleSheet("font-size:14px;")

    def set_text(self, raw: str):
        self.label.setText(md_lite(raw))


# ---------------------------------------------------------------- 来源卡片

class SourcesCard(CardWidget):
    """检索来源卡片：默认只显示摘要，点「展开来源」看完整明细 + 原文链接。

    为什么默认折叠：来源是给"想较真的人"核对的，不是给普通用户阅读的，
    折叠让聊天流保持干净；但 RAG 必须能亮原文，所以一秒可达。
    """

    def __init__(self, sources: list[dict], parent=None):
        super().__init__(parent)
        self.setFixedWidth(600)
        self.setStyleSheet("CardWidget{border-radius:8px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(4)

        self.detail = CaptionLabel(sources_lines(sources, with_url=True), self)
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse
                                            | Qt.LinksAccessibleByMouse)
        self.detail.setOpenExternalLinks(True)
        self.detail.hide()

        self.summary = CaptionLabel(
            f"📚 本次回答依据 {len(sources)} 条检索来源（点击展开核对）", self)
        lay.addWidget(self.summary)
        lay.addWidget(self.detail)

        self.toggle = PushButton("展开来源", self)
        self.toggle.setFixedHeight(28)
        self.toggle.clicked.connect(self._flip)
        lay.addWidget(self.toggle)

    def _flip(self):
        shown = not self.detail.isVisible()
        self.detail.setVisible(shown)
        self.toggle.setText("收起来源" if shown else "展开来源")
        self.adjustSize()


# ---------------------------------------------------------------- 聊天界面

class ChatInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatInterface")
        self.qa = QASystem()
        self.bridge = Bridge()
        self._buf = ""
        self._answer_bubble: Bubble | None = None
        self.bridge.chunk.connect(self._on_chunk)
        self.bridge.answer_done.connect(self._on_done)
        self.bridge.busy.connect(self._set_busy)
        self._build_ui()
        # 状态栏在 _build_ui 里创建，连接必须放在它之后
        self.bridge.status.connect(self.status_label.setText)
        self._apply_saved_theme()
        threading.Thread(target=warmup_thread, args=(self.qa, self.bridge),
                         daemon=True).start()

    # ---------------- UI

    def _build_ui(self):
        self.setFont(QFont("Microsoft YaHei UI", 10))
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 12)
        root.setSpacing(10)

        # 顶部：标题 + 暗色开关
        head = QHBoxLayout()
        head.addWidget(TitleLabel("NaviRAG"))
        head.addSpacing(8)
        head.addWidget(CaptionLabel("本地多游戏攻略 RAG · 艾尔登法环"))
        head.addStretch(1)
        self.theme_switch = SwitchButton("暗色模式")
        self.theme_switch.checkedChanged.connect(self._toggle_theme)
        head.addWidget(self.theme_switch)
        root.addLayout(head)

        # 聊天滚动区
        self.scroll = ScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(self.scroll.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("chatInner")
        self.chat_lay = QVBoxLayout(inner)
        self.chat_lay.setContentsMargins(4, 4, 4, 4)
        self.chat_lay.setSpacing(10)
        self.chat_lay.addStretch(1)
        self.scroll.setWidget(inner)
        root.addWidget(self.scroll, 1)

        # 底部：输入 + 发送
        row = QHBoxLayout()
        self.input = LineEdit()
        self.input.setPlaceholderText("输入问题，回车发送…")
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self.send)
        self.send_btn = PrimaryPushButton(FluentIcon.SEND, "发送")
        self.send_btn.clicked.connect(self.send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        root.addLayout(row)

        self.progress = IndeterminateProgressBar(self)
        self.progress.hide()
        root.addWidget(self.progress)

        self.status_label = CaptionLabel("正在启动…")
        root.addWidget(self.status_label)

        self._welcome()

    def _apply_saved_theme(self):
        """主题持久化：QSettings 记住上次选择（注册表，无额外文件）。"""
        settings = QSettings("NaviRAG", "desktop")
        dark = settings.value("dark_theme", False, type=bool)
        self.theme_switch.setChecked(dark)
        setTheme(Theme.DARK if dark else Theme.LIGHT)
        # 主题应用后，欢迎气泡颜色跟随重建
        for i in range(self.chat_lay.count() - 1):
            w = self.chat_lay.itemAt(i).widget()
            if isinstance(w, Bubble) and w.role == "user":
                bg = "#0078d4" if not dark else "#2b6cb8"
                w.setStyleSheet(f"CardWidget{{background:{bg};border-radius:8px;}}")

    def _welcome(self):
        tip = ("试试问我：\n"
               "· 玛莲妮亚怎么逃课\n"
               "· 玛莲妮亚怎么打（不逃课）\n"
               "· 我 60 级能打大树守卫吗\n"
               "· 玛莲妮亚的追忆能换什么")
        b = Bubble("bot")
        b.set_text(tip)
        self._add_bubble(b)

    # ---------------- 气泡管理

    def _add_bubble(self, w: QWidget, role: str = "bot"):
        align = Qt.AlignRight if role == "user" else Qt.AlignLeft
        self.chat_lay.insertWidget(self.chat_lay.count() - 1, w, 0, align)
        self._scroll_bottom()

    def _scroll_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ---------------- 槽

    def _on_chunk(self, chunk: str):
        # 第一个片段触发创建回答气泡
        if self._answer_bubble is None:
            self._answer_bubble = Bubble("bot")
            self._buf = ""
            self._add_bubble(self._answer_bubble)
        if chunk:
            self._buf += chunk
        self._answer_bubble.set_text(self._buf)
        self._scroll_bottom()

    def _on_done(self, sources: list):
        if sources:
            self._add_bubble(SourcesCard(sources))
        self._answer_bubble = None

    def _set_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.send_btn.setText("回答中…" if busy else "发送")
        self.progress.show() if busy else self.progress.hide()
        if not busy:
            self.input.setFocus()

    def _toggle_theme(self, checked: bool):
        setTheme(Theme.DARK if checked else Theme.LIGHT)
        QSettings("NaviRAG", "desktop").setValue("dark_theme", bool(checked))
        # 换肤后用户气泡底色需要重建
        bg = "#2b6cb8" if checked else "#0078d4"
        for i in range(self.chat_lay.count() - 1):
            w = self.chat_lay.itemAt(i).widget()
            if isinstance(w, Bubble) and w.role == "user":
                w.setStyleSheet(f"CardWidget{{background:{bg};border-radius:8px;}}")
        InfoBar.success("已切换主题", "下次启动会记住这个选择。",
                        duration=2000, parent=self.window())

    # ---------------- 动作

    def send(self):
        text = self.input.text().strip()
        if not text:
            return
        if not self.qa.ready:
            InfoBar.warning("还没就绪", "索引正在加载，稍等几秒再问。",
                            duration=2000, parent=self.window())
            return
        self.input.clear()
        self._set_busy(True)
        b = Bubble("user")
        b.set_text(text)
        self._add_bubble(b, role="user")
        threading.Thread(target=ask_thread, args=(self.qa, self.bridge, text),
                         daemon=True).start()


class DesktopChatWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)   # 单实例靠这个标题找回已有窗口
        self.resize(860, 720)
        self.interface = ChatInterface(self)
        self.addSubInterface(self.interface, FluentIcon.CHAT, "问答")


def run_rebuild():
    """--rebuild 模式：更新攻略库（md→jsonl→重建索引），弹窗报告，不进聊天界面。

    打包成 exe 后更新攻略的方式：双击「更新攻略.bat」→ 它调 NaviRAG.exe --rebuild。
    数据外置在 exe 旁，改完 md 跑这一下就生效，不用重装 Python、不用重新打包。

    打包后没有控制台，进度同时写入 exe 旁的 update_log.txt——
    既是给弹窗攒内容，也是更新失败时唯一能回看的现场。
    """
    from PySide6.QtWidgets import QMessageBox

    from core.retriever import ROOT
    from rebuild import rebuild

    lines: list[str] = []
    logfile = ROOT / "update_log.txt"

    class _LogWriter:
        """把 stdout/stderr 引到日志文件：打包版没有控制台，
        build_index 里的 print 和原生崩溃的 traceback 全靠这里落盘。"""

        def write(self, s):
            if s.strip():
                lines.append(s.rstrip())
                with open(logfile, "a", encoding="utf-8") as f:
                    f.write(s if s.endswith("\n") else s + "\n")

        def flush(self):
            pass

    def log(m: str):
        lines.append(str(m))
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{m}\n")

    try:
        logfile.write_text("", encoding="utf-8")    # 每次更新覆盖旧日志
        _stdout, _stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _LogWriter()      # 捕获 build_index 的进度 print
        rebuild(log)
        sys.stdout, sys.stderr = _stdout, _stderr
        QMessageBox.information(None, "攻略库更新完成", "\n".join(lines[-40:]))
    except Exception as e:
        import traceback
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stdout, sys.stderr = _stdout, _stderr
        tail = "\n".join(lines[-20:]) or "（无更多日志）"
        QMessageBox.critical(None, "攻略库更新失败", f"{e}\n\n{tail}")


def main():
    # 单实例闸门已在模块顶部(_enforce_single_instance)执行过，走到这里说明
    # 本进程持有互斥量，是唯一的那个实例。

    if "--rebuild" in sys.argv:
        _ = QApplication(sys.argv)      # 弹窗必需
        run_rebuild()
        return

    app = QApplication(sys.argv)
    win = DesktopChatWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
