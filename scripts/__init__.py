# -*- coding: utf-8 -*-
"""scripts 包标记：让 rebuild 入口可以 `from scripts.build_l2 import convert`。

打包成 exe 时，PyInstaller 依赖静态 import 分析收集模块，
命名空间包（无 __init__.py）可能被漏掉，所以显式声明。
"""
