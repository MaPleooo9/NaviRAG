"""把 OpenSpec 生成的 skill 桥接成 WorkBuddy 能加载的形式。

背景
----
`openspec init --tools codebuddy` 会把 skill 写到 `.codebuddy/skills/openspec-*/`，
但 WorkBuddy 只从 `~/.workbuddy/skills/`（用户级）和 `<项目>/.workbuddy/skills/`
（项目级）加载 skill，读不到 `.codebuddy/`。所以需要一次转换 + 拷贝。

转换做三件事
------------
1. **frontmatter 规范化**：只保留 `name` / `description`，补 `agent_created: true`
   （本机惯例）。丢掉 `allowed-tools` —— 那是 Claude Code 的工具白名单语法，
   WorkBuddy 不认，留着有被当成工具门禁误解释的风险。
2. **注入「本机调用约定」**：本机没有全局安装 openspec CLI，裸 `openspec xxx` 会
   报 command not found，必须走 `npx -y @fission-ai/openspec@latest xxx`。
3. **注入「使用边界」**：OpenSpec 原生的 planning boundary 要求 propose 之后停下、
   等用户下一轮再 apply。这与本机既定规范「目标清楚后自主推进，不停留在计划阶段」
   冲突，按后者覆盖（与 spec-driven-development skill 的处理方式保持一致）。

为什么用脚本而不是手工拷
------------------------
`openspec update` 会重新生成 `.codebuddy/skills/`，把这三点覆盖掉。有脚本就能一条命令
重新桥接，不用记得手工补。脚本本身进仓库，skill 产物在 `.workbuddy/`（已 gitignore）。

用法
----
    python scripts/sync_openspec_skills.py            # 同步
    python scripts/sync_openspec_skills.py --check    # 只检查是否一致，不写
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".codebuddy" / "skills"
DST = ROOT / ".workbuddy" / "skills"

FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)

# 注入块：调用约定 + 使用边界。放在 frontmatter 之后、正文之前。
INJECTED = """
## 本机适配（由 scripts/sync_openspec_skills.py 注入，勿手工改）

### 1. 调用约定：openspec 未全局安装

本机**没有**把 openspec 装到全局（避免污染环境），裸命令 `openspec xxx` 会报
command not found。所有形如 `openspec <子命令>` 的地方，一律改成：

```bash
npx -y @fission-ai/openspec@latest <子命令>
```

例：`openspec context --json` → `npx -y @fission-ai/openspec@latest context --json`。
首次调用会下载包（约几秒），之后走 npx 缓存。

### 2. 使用边界：默认连续执行到归档（覆盖下文 planning boundary）

下文（以及本 skill 正文里的 Planning boundary / Guardrails 段落）要求
「propose 完就停下，等用户下一轮再 apply」。**本机按用户既定规范覆盖它**：

- 用户已授权「目标清楚后自主推进，不停留在计划阶段，也不反复确认已决定的事项」。
  因此用户明确要求实现某个变更时，**propose → apply → archive 在同一轮内走完**，
  不要在中途停下等确认。
- **例外**：用户说「先出方案给我看」「先别动代码」「我要审一下」时，严格停在
  propose，产出物摆出来等审阅。

产物（proposal / specs / design / tasks）**照常全部生成**——被覆盖的只是「何时开始
写代码」这个门禁，审计链不受影响。

"""


def read_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """返回 (frontmatter 行的字典, 正文)。description 保留原始行以免重新引号出错。"""
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("没有 frontmatter")
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip()] = val  # 保留原始值（含可能的引号）
    return fields, text[m.end():]


def build(name: str, description_raw: str, body: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description:{description_raw}\n"
        "agent_created: true\n"
        "---\n"
        + INJECTED
        + body.lstrip("\n")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="桥接 OpenSpec skills 到 .workbuddy/skills/")
    ap.add_argument("--check", action="store_true", help="只检查是否一致，不写文件")
    args = ap.parse_args()

    if not SRC.is_dir():
        print(f"✗ 找不到源目录 {SRC.relative_to(ROOT)}")
        print("  先跑：npx -y @fission-ai/openspec@latest init --tools codebuddy")
        return 1

    sources = sorted(p for p in SRC.iterdir() if p.is_dir() and p.name.startswith("openspec-"))
    if not sources:
        print(f"✗ {SRC.relative_to(ROOT)} 下没有 openspec-* 目录")
        return 1

    if not args.check:
        DST.mkdir(parents=True, exist_ok=True)

    changed = same = 0
    for src_dir in sources:
        src_file = src_dir / "SKILL.md"
        if not src_file.is_file():
            print(f"  跳过 {src_dir.name}（没有 SKILL.md）")
            continue

        fields, body = read_frontmatter(src_file.read_text(encoding="utf-8"))
        name = fields.get("name", src_dir.name).strip().strip("'\"")
        desc = fields.get("description")
        if desc is None:
            print(f"  跳过 {src_dir.name}（frontmatter 缺 description）")
            continue

        out = build(name, desc, body)
        dst_file = DST / src_dir.name / "SKILL.md"

        if dst_file.is_file() and dst_file.read_text(encoding="utf-8") == out:
            same += 1
            print(f"  = {src_dir.name}（已一致）")
            continue

        if args.check:
            changed += 1
            print(f"  ! {src_dir.name}（内容不一致，需重新同步）")
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_text(out, encoding="utf-8")
        changed += 1
        print(f"  ✓ {src_dir.name} → .workbuddy/skills/{src_dir.name}/SKILL.md")

    print()
    if args.check:
        print(f"检查完毕：{same} 个一致，{changed} 个需同步")
        return 1 if changed else 0
    print(f"同步完毕：{changed} 个已写入，{same} 个无需改动")
    print(f"目标目录：{DST.relative_to(ROOT)}（.workbuddy/ 已 gitignore，不进仓库）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
