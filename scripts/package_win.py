# -*- coding: utf-8 -*-
"""
Windows 一键打包脚本：PyInstaller 构建 → 复制外置数据 → 体积报告。

产出 dist/NaviRAG/ 目录：
    NaviRAG.exe        双击启动桌面问答
    更新攻略.bat        改完 md 后双击，更新检索库（弹窗看进度）
    data/ models/ .vectorstore/   外置数据，改了不用重新打包

用法:
    python scripts/package_win.py            # 构建 + 复制数据
    python scripts/package_win.py --zip      # 额外压成 NaviRAG-dist.zip（挂 Release 用）
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "NaviRAG"

# 外置数据：打包后复制到 exe 旁边（方案 A 的"数据与程序分离"）
EXTERNAL = ["data", "models", ".vectorstore", "更新攻略.bat"]


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="打包后压成 zip")
    ap.add_argument("--skip-build", action="store_true",
                    help="跳过 PyInstaller，只重新复制外置数据")
    args = ap.parse_args()

    if not args.skip_build:
        t0 = time.time()
        print("== PyInstaller 构建（首次约 3-8 分钟）==", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
             str(ROOT / "NaviRAG.spec")],
            cwd=ROOT,
        )
        if r.returncode != 0:
            print("❌ PyInstaller 失败，见上方日志")
            return 1
        print(f"== 构建完成，用时 {time.time() - t0:.0f} 秒 ==")

    if not DIST.exists():
        print(f"❌ 未找到 {DIST}，先构建")
        return 1

    print("== 复制外置数据 ==", flush=True)
    for name in EXTERNAL:
        src = ROOT / name
        dst = DIST / name
        if not src.exists():
            print(f"  跳过（不存在）: {name}")
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  + {name}")

    exe = DIST / "NaviRAG.exe"
    total = du(DIST)
    print(f"\n✅ 完成: {DIST}")
    print(f"   NaviRAG.exe  {human(exe.stat().st_size)}")
    print(f"   目录总计      {human(total)}（不含 zip）")
    print("   分发：整个 NaviRAG/ 目录打 zip 挂 GitHub Release（≤2GB/文件，不占仓库体积）")

    if args.zip:
        import zipfile

        zpath = ROOT / "dist" / "NaviRAG-dist.zip"
        print(f"\n== 压缩 {zpath.name}（几分钟）==", flush=True)
        t0 = time.time()
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in DIST.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(DIST.parent))
        print(f"✅ {zpath}  {human(zpath.stat().st_size)}，用时 {time.time() - t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
