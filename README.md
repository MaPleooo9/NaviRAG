# NaviRAG — 本地轻量级多游戏攻略 RAG 问答助手

> 无 GPU、纯 CPU、完全本地离线。问「玛莲妮亚怎么逃课」，它给你真·社区打法，而不是 wiki 流水账。

## 这是什么

一个本地运行的 RAG（检索增强生成）游戏攻略问答助手，当前主打《艾尔登法环》。它和「又一个 RAG 教程项目」的区别在于两点：

1. **三层知识库**：
   - **L1** wiki 攻略 + 官方中文语料（自动抓取，8,334 条）
   - **L2** 逃课/轮椅打法（**人工整理**，55 条）——逃课打法天然不在 wiki 上，wiki 记录正规流程，「卡石头」「蹲大树引魔像」只存在于玩家社区
   - **L3** 常规打法（**人工整理**，30 条）——不卡 bug、不利用地形，靠操作和机制理解打赢

   L2 和 L3 必须分开：混在一层时，问「怎么逃课」会召回归常规打法，问「怎么打」又会召回归逃课，两边都不对。想学真本事的人和二周目赶路的人要的不是同一种答案。
2. **规则路由 + 混合检索**：纯向量检索在「短问句 vs 长文档」场景下会被内容词干扰（实测：问大树守卫，召回的是文本里带「蹲大树」的铃珠猎人打法）。解法是实体命中硬置顶 + 三路召回分层加权，而不是调 embedding。路由还能识别否定表达——「怎么打**不逃课**」里也含"逃课"二字，纯关键词会误判，这里做了专门的否定词表。

## 特性

- **本地 GPU 推理**：qwen3:8b 在 RTX 4070 Laptop 实测 41 tok/s（直答首字 ~1s），250 字回答约 9 秒；qwen2.5:3b 可达 97 tok/s
- **官方术语表**：从游戏文件提取 15,751 条官方中英对照（玛莲妮亚 / Malenia），中文提问自动注入英文名增强跨语言检索，回答中英双语
- **意图路由不花 Token**：常规 / 逃课 / 无脑度排序 三种意图纯规则识别，零 LLM 调用
- **结构化过滤**：「我 60 级能逃吗」→ 直接按 level_req 过滤；「不用召唤的」→ 按 requires 排除——这是纯向量检索做不到的
- **冷启动治理**：Ollama 默认 5 分钟卸载模型，首次调用 31.7s / 热调用 0.18s。启动预热 + keep_alive 常驻解决（见 docs/bench.md）
- **降级不白屏**：本地大模型掉线时自动切「纯检索模式」，直接把原文片段结构化列出来——检索层才是核心资产，宁可不给模型润色，也不能让它编

## 快速开始

```bash
# 0. 启动界面（数据已就绪时，这一步就够了）
streamlit run app.py

# 1. 依赖
pip install -r requirements.txt

# 2. Ollama + 模型（约 2GB）
ollama pull qwen3:8b      # 也兼容 qwen2.5:3b（更快，质量稍低）

# 3. 构建数据（依次运行，产物在 data/elden_ring/）
python scripts/build_glossary.py    # 官方术语表（15,751 条）
python scripts/fetch_wiki.py        # wiki 抓取（~950 页）
python scripts/build_l1_zh.py       # 官方中文语料（~5,000 条）
python scripts/clean_l1.py          # 清洗索引页噪声
# L2/L3 是人工层，Markdown 写好后再转 JSONL（改完必跑，否则不生效）
python scripts/build_l2.py          # 逃课打法 → l2_cheese.jsonl
python scripts/build_l3.py          # 常规打法 → l3_normal.jsonl

# 4. embedding 模型（ModelScope 直链，~450MB）
python scripts/download_model.py

# 5. 构建向量库 + 验证检索
python -m core.retriever "玛莲妮亚怎么打"

# 6. 界面无头冒烟（不打开浏览器也能验证脚本没崩）
python scripts/smoke_app.py
```

## 界面

`streamlit run app.py` 之后是一个对话界面，左侧控制台可调「返回条数 / 最低无脑度 / 是否显示来源」。

两个刻意的设计：

- **答案下方永远摊开检索来源**。RAG 系统如果不敢把原文亮出来，就等于让用户盲信一个会编造的 3B 模型。每条来源标注层（L1 攻略 / L2 逃课）、无脑度、建议等级、是否本人验证、融合得分，以及本次的路由决策（意图 + 双路召回条数）。
- **Ollama 掉了不白屏**。本地大模型连不上时自动切「纯检索模式」，直接把原文片段结构化列出来——检索层才是核心资产，宁可不给模型润色，也不能让它编。

## 目录结构

```
├── app.py                  # Streamlit 界面（流式输出 + 来源展示 + 降级）
├── core/
│   ├── retriever.py        # 三路召回 + 规则路由 + 混合排序
│   ├── qa.py               # 编排层：检索 → Prompt → 流式生成 / 降级
│   └── llm.py              # Ollama 客户端：流式 + 预热常驻 + 故障可读
├── scripts/
│   ├── smoke_app.py        # 界面无头冒烟测试（AppTest）
│   ├── fetch_wiki.py       # wiki.gg 抓取（断点续传 / UA 伪装 / 限速）
│   ├── build_glossary.py   # 官方中英术语表构建
│   ├── build_l1_zh.py      # 官方中文语料导出
│   ├── build_l2.py         # L2 逃课数据解析（markdown → jsonl，带校验）
│   ├── build_l3.py         # L3 常规打法解析（字段不同，独立解析器）
│   ├── verify_l2.py        # 批量把「验证: 否」改成「是」
│   ├── download_model.py   # embedding 模型下载（ModelScope 直链）
│   └── bench_ollama.py     # 本机 LLM benchmark（TTFT / tok/s）
├── data/elden_ring/
│   ├── l2_cheese.md        # ★ 逃课打法知识库（人工整理，核心资产）
│   └── l3_normal.md        # ★ 常规打法知识库（人工整理）
└── docs/bench.md           # 性能实测数据
```

## 性能实测（笔记本，i9-14900HX，无独显）

| 指标 | 数值 |
|---|---|
| qwen3:8b 生成速度（热，直答） | 41.3 tok/s（qwen2.5:3b 为 96.7） |
| 首字延迟（冷 / 热） | 31.7s / 0.18s |
| 生成 250 字中文 | ~4.7s |
| 知识库规模 | 8,419 条（L1 8,334 + L2 55 + L3 30） |
| 界面冷启动（索引 + 模型预热） | 9.0s |
| 端到端问答（检索 + 生成） | 3.4s |
| 索引全量重建 | 67s |

## Roadmap

- [x] Streamlit UI + 流式输出 + 异常降级（LLM 挂了直接返回检索原文）
- [ ] L2 扩充至 100+ 条（当前 55）
- [ ] L3 扩充至 50 条（当前 30）
- [ ] 评估：Hit Rate / MRR，Base vs +Reranker 对比实验
- [ ] pywebview 桌面壳（Windows 走系统 WebView2，零 Chromium 依赖）

## 数据说明

- L1 语料抓自 [eldenring.wiki.gg](https://eldenring.wiki.gg)（CC BY-NC-SA），官方中文译名/描述提取自游戏文本（仅作术语对照与检索用途）
- 派生数据文件不入库（见 .gitignore），clone 后用 `scripts/` 一键重建
