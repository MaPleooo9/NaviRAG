# -*- coding: utf-8 -*-
"""把人工层数据（L2 逃课 / L3 常规）中所有条目的「验证: 否」批量改为「验证: 是」。

用法:
    python scripts/verify_l2.py                          # 预览 L2（只打印会改哪些）
    python scripts/verify_l2.py --apply                  # 真正写入 L2
    python scripts/verify_l2.py --file data/elden_ring/l3_normal.md --apply   # 应用到 L3

只匹配 DATA-START 之后、以 "-" 开头的验证字段行，不会动说明文档和注释。
"""
import io
import re
import sys

DEFAULT_PATH = "data/elden_ring/l2_cheese.md"

# 形如:  - **验证**: 否   /  - 验证：否   /  - **验证**否
PATTERN = re.compile(
    r"(?m)^(?P<head>\s*-\s*(?:\*\*验证\**|\*验证\**|验证)\s*[:：]?\s*\**\s*)否(?P<tail>\s*\**\s*)$"
)


def main() -> None:
    apply = "--apply" in sys.argv
    path = DEFAULT_PATH
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    src = io.open(path, encoding="utf-8").read()

    marker = "<!-- DATA-START"
    idx = src.find(marker)
    head, body = (src[:idx], src[idx:]) if idx >= 0 else ("", src)

    hits = PATTERN.findall(body)
    new_body = PATTERN.sub(lambda m: m.group("head") + "是" + m.group("tail"), body)

    print("待修改条目数:", len(hits))
    for line in PATTERN.finditer(body):
        print("   ", line.group(0).strip())

    remain_no = len(re.findall(r"验证\**\s*[:：]\**\s*否", new_body))
    remain_yes = len(re.findall(r"验证\**\s*[:：]\**\s*是", new_body))
    print("修改后 -> 否: %d, 是: %d" % (remain_no, remain_yes))

    if not apply:
        print("\n(预览模式，未写入。加 --apply 生效)")
        return

    io.open(path, "w", encoding="utf-8", newline="").write(head + new_body)
    print("已写入:", path)


if __name__ == "__main__":
    main()
