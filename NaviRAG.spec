# -*- mode: python ; coding: utf-8 -*-
"""NaviRAG 打包配置（PyInstaller 6.x，onedir）。

设计决策（为什么这样打）:
    onedir 而非 onefile —— 启动快一个量级，且数据/模型外置在 exe 旁边，
    改攻略不用重新打包（见 rebuild.py 的注释）。
    console=False —— 桌面 GUI 不带黑框；更新攻略走 --rebuild 弹窗展示日志。
    数据不进 datas —— data/、models/、.vectorstore/ 由 package_win.py
    在打包完成后复制到 dist/NaviRAG/ 旁边，属于"外置数据"，不封进包里。
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('qfluentwidgets')   # Fluent 控件内置的字体/样式资源

a = Analysis(
    ['desktop_chat.py'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'rebuild',            # --rebuild 入口（函数内 import，防漏收集）
        'scripts.build_l2',   # rebuild 依赖的 md 解析器
        'scripts.build_l3',
        # chromadb 在 config.py 里用 importlib 按字符串动态加载实现类，
        # 静态分析收不到（实测缺 chromadb.telemetry.product.posthog 直接崩），
        # 必须整包强制收集。
        *collect_submodules('chromadb'),
    ],
    excludes=[
        'tkinter', 'matplotlib', 'IPython', 'pytest',
        'PyQt5', 'PyQt6',     # qfluentwidgets 会探测多绑定，只保留 PySide6
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NaviRAG',
    debug=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='NaviRAG',
)
