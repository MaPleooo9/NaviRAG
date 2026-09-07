# NaviRAG — 本地轻量级多游戏攻略 RAG 问答助手

> 无 GPU、纯 CPU、完全本地离线。问「玛莲妮亚怎么逃课」，它给你真·社区打法，而不是 wiki 流水账。

## 这是什么

一个本地运行的 RAG（检索增强生成）游戏攻略问答助手，当前主打《艾尔登法环》。它和「又一个 RAG 教程项目」的区别在于两点：

1. **分层知识库**：L1 层（wiki 攻略 + 官方中文语料，自动抓取）+ L2 层（逃课/轮椅打法，**人工整理的社区智慧**）。逃课打法天然不在 wiki 上——wiki 记录正规流程，「卡石头」「蹲大树引魔像」只存在于玩家社区。L2 层是手工资产，也是本项目的核心壁垒。
2. **规则路由 + 混合检索**：纯向量检索在「短问句 vs 长文档」场景下会被内容词干扰（实测：问大树守卫，召回的是文本里带「蹲大树」的铃珠猎人打法）。解法是实体命中硬置顶 + 两路召回分层加权，而不是调 embedding。

## 特性

- **零 GPU**：qwen2.5:3b 纯 CPU 实测 80 tok/s（i9-14900HX），250 字回答约 4.7 秒
- **官方术语表**：从游戏文件提取 15,751 条官方中英对照（玛莲妮亚 / Malenia），中文提问自动注入英文名增强跨语言检索，回答中英双语
- **意图路由不花 Token**：常规 / 逃课 / 无脑度排序 三种意图纯规则识别，零 LLM 调用
- **结构化过滤**：「我 60 级能逃吗」→ 直接按 level_req 过滤；「不用召唤的」→ 按 requires 排除——这是纯向量检索做不到的
- **冷启动治理**：Ollama 默认 5 分钟卸载模型，首次调用 31.7s / 热调用 0.18s。启动预热 + keep_alive 常驻解决（见 docs/bench.md）

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. Ollama + 模型（约 2GB）
ollama pull qwen2.5:3b

# 3. 构建数据（依次运行，产物在 data/elden_ring/）
python scripts/build_glossary.py    # 官方术语表（15,751 条）
python scripts/fetch_wiki.py        # wiki 抓取（~950 页）
python scripts/build_l1_zh.py       # 官方中文语料（~5,000 条）
python scripts/clean_l1.py          # 清洗索引页噪声

# 4. embedding 模型（ModelScope 直链，~450MB）
python scripts/download_model.py

# 5. 构建向量库 + 验证检索
python -m core.retriever "玛莲妮亚怎么打"
```

## 目录结构

```
├── app.py                  # Streamlit 前端（开发中）
├── core/
│   └── retriever.py        # 两路召回 + 规则路由 + 混合排序
├── scripts/
│   ├── fetch_wiki.py       # wiki.gg 抓取（断点续传 / UA 伪装 / 限速）
│   ├── build_glossary.py   # 官方中英术语表构建
│   ├── build_l1_zh.py      # 官方中文语料导出
│   ├── build_l2.py         # L2 手工数据解析（markdown → jsonl，带校验）
│   ├── download_model.py   # embedding 模型下载（ModelScope 直链）
│   └── bench_ollama.py     # 本机 LLM benchmark（TTFT / tok/s）
├── data/elden_ring/
│   └── l2_cheese.md        # ★ 逃课打法知识库（人工整理，核心资产）
└── docs/bench.md           # 性能实测数据
```

## 性能实测（笔记本，i9-14900HX，无独显）

| 指标 | 数值 |
|---|---|
| qwen2.5:3b 生成速度（热） | 80.6 tok/s |
| 首字延迟（冷 / 热） | 31.7s / 0.18s |
| 生成 250 字中文 | ~4.7s |
| 知识库规模 | 8,357 条（L1 8,334 + L2 23） |

## Roadmap

- [ ] Streamlit UI + 流式输出 + 异常降级（LLM 挂了直接返回检索原文）
- [ ] L2 扩充至 100+ 条（当前 23）
- [ ] 评估：Hit Rate / MRR，Base vs +Reranker 对比实验
- [ ] pywebview 桌面壳（Windows 走系统 WebView2，零 Chromium 依赖）

## 数据说明

- L1 语料抓自 [eldenring.wiki.gg](https://eldenring.wiki.gg)（CC BY-NC-SA），官方中文译名/描述提取自游戏文本（仅作术语对照与检索用途）
- 派生数据文件不入库（见 .gitignore），clone 后用 `scripts/` 一键重建
