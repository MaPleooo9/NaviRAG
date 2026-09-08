# -*- coding: utf-8 -*-
"""
app.py 的无头冒烟测试（不需要打开浏览器）。

用 Streamlit 官方的 AppTest 框架真跑一遍脚本，捕获运行时异常——
Streamlit 的错误经常只在运行时才暴露（比如 cache_resource 传了不可 hash 的对象、
session_state 键不存在），肉眼审查代码是看不出来的。

用法:
    python scripts/smoke_app.py
    python scripts/smoke_app.py --ask "玛莲妮亚怎么逃课"   # 顺便模拟一次提问

退出码: 0 = 全部通过；1 = 有异常
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

CASES = [
    "玛莲妮亚怎么逃课？",
    "给我最无脑的打法",
    "我 60 级能打大树守卫吗",
    "女武神的水鸟乱舞怎么躲",
]


def check(label: str, at: AppTest) -> bool:
    if at.exception:
        print(f"  ❌ {label} 抛异常:")
        for e in at.exception:
            print(f"     {type(e).__name__}: {e}")
        return False
    print(f"  ✅ {label}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None, help="额外模拟一次提问")
    args = ap.parse_args()

    print("=" * 60)
    print("app.py 无头冒烟测试")
    print("=" * 60)

    ok = True
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    print(f"首次运行（含索引加载 + 模型预热）: {time.time() - t0:.1f}s")
    ok &= check("脚本首次运行", at)

    if not ok:
        return 1

    md = [m.value for m in at.markdown if m.value]
    print(f"  渲染 Markdown 块: {len(md)} 个")
    print(f"  侧边栏按钮: {len(at.sidebar.button)} 个")

    # 模拟点击示例按钮 → 走一次完整问答
    questions = [args.ask] if args.ask else CASES[:1]
    for q in questions:
        print(f"\n── 模拟提问: {q}")
        at.session_state.pending = q
        t1 = time.time()
        at.run()
        print(f"  耗时 {time.time() - t1:.1f}s")
        ok &= check("问答流程", at)

        texts = [m.value for m in at.markdown if m.value]
        if texts:
            last = texts[-1]
            print(f"  末尾输出预览: {last[:120]}…")
        if at.error:
            print("  ⚠️ 页面上有 error:",
                  [e.value[:100] for e in at.error])

    print("\n" + "=" * 60)
    print("全部通过 ✅" if ok else "存在失败 ❌")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
