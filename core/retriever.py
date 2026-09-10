# -*- coding: utf-8 -*-
"""
分层检索器：L1 wiki（英/中）+ L2 逃课层 + L3 常规打法层，三路召回、加权融合。

为什么是"分层"而不是一把梭：
    L1 是通用攻略，L2 是人工整理的逃课技巧，L3 是人工整理的常规打法。
    用户问"怎么打"时 L1 命中，问"怎么逃"时必须命中 L2，想学真本事时要命中 L3——
    但 L2/L3 表述和"怎么打"的主流段落语义相似度并不高，
    纯向量检索会把它们淹没。所以分开召回、按层加权，这也是本项目的核心实验变量。

    L2 和 L3 必须分开：混在一层时，问"怎么逃课"会召回归常规打法，
    问"怎么打"又会召回归逃课——两边都不对。

用法（命令行自测）:
    python -m core.retriever "玛莲妮亚怎么打"
    python -m core.retriever "给我最无脑的打法" --show 8
"""

import json
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

def _app_root() -> Path:
    """应用根目录：开发时=项目目录；PyInstaller 打包后=exe 所在目录。

    打包成 exe 后 `__file__` 指向解包临时目录（_internal），
    数据必须外置在 exe 旁边才能"改攻略不重打包"——这是方案 A 的基石。
    """
    if getattr(sys, "frozen", False):          # PyInstaller 打包运行
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = _app_root()
DATA = ROOT / "data" / "elden_ring"
DB_DIR = ROOT / ".vectorstore"

# 优先用本地模型（scripts/download_model.py 下载），否则从 HF 拉取
LOCAL_MODEL = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = str(LOCAL_MODEL) if LOCAL_MODEL.exists() \
    else "paraphrase-multilingual-MiniLM-L12-v2"

# 默认权重：L2/L3 是人工整理的核心资产，给更高权重。这是评估实验要扫的参数。
W_L1, W_L2, W_L3 = 0.4, 0.4, 0.2

TOPK_L1 = 12      # L1 层召回数（8000+ 条，取多了慢且没用）
TOPK_L2 = 300     # L2 层全量召回（人工条目总共几百条，漏一条就是漏一个打法）
TOPK_L3 = 300     # L3 层同理
FINAL_K = 6       # 最终返回多少条


# 别名档位：俗称/称号/错写命中（"女武神" → 玛莲妮亚）。
# 刻意低于全名精确命中（2.0）和短名精确命中（1.5）——别名是"用户没说正式名"
# 的兜底，不该压过真的说对了名字的条目；但要高于部分匹配（1.0），
# 否则"接肢怎么正常打"这种省略后半段的写法会被别的条目蹭边抢走。
ALIAS_SCORE = 1.45


def layer_of(r) -> str:
    """按字段判断条目属于哪一层。id 前缀不靠谱，字段才是事实来源。"""
    if "difficulty" in r:
        return "l3"
    if "brain_level" in r:
        return "l2"
    return "l1"


# ---------------------------------------------------------------- 索引构建

def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _embed_text(r):
    """送进 embedding 模型的文本。L1 英文条目拼上官方中文名，跨语言匹配更稳"""
    t = r["text"]
    if r.get("title_zh"):
        t = f"{r['title_zh']} {r['title']}。{t}"
    elif r.get("title") and r.get("lang") == "en":
        t = f"{r['title']}. {t}"
    return t[:2000]


# HNSW 默认的 search_ef 很小，近似度高到「同一条 query 两次召回排序不一样」
# ——实测评估脚本连跑六次能出四种结果，指标根本不可复现。
# 数据量只有八千多条，精确搜索也很快（71 条查询 1.1s，几乎无额外开销），
# 没必要为了省几毫秒牺牲可复现性，所以把候选队列拉满。
# 注意：这两个参数在建库时就固化进 collection，改了要 force=True 重建才生效。
SEARCH_EF = 1000
HNSW_METADATA = {"hnsw:space": "cosine", "hnsw:search_ef": SEARCH_EF}


def build_index(model=None, force=False):
    """构建/增量更新 ChromaDB。返回 (client, collection, model)"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(
        "elden_ring", metadata=HNSW_METADATA)

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    if force:
        client.delete_collection("elden_ring")
        col = client.get_or_create_collection(
            "elden_ring", metadata=HNSW_METADATA)

    if (col.metadata or {}).get("hnsw:search_ef") != SEARCH_EF:
        print(f"  ⚠ 当前 collection 的 hnsw:search_ef="
              f"{(col.metadata or {}).get('hnsw:search_ef')}，"
              f"低于 {SEARCH_EF} 会让检索结果不可复现，"
              f"请用 build_index(force=True) 重建索引")

    rows = []
    for name in ("l1_wiki.jsonl", "l1_zh_game.jsonl", "l2_cheese.jsonl",
                 "l3_normal.jsonl"):
        p = DATA / name
        if p.exists():      # L3 刚加，老环境可能没有这个文件
            rows += _load_jsonl(p)

    # 增量：跳过已有 id
    existing = set(col.get()["ids"]) if col.count() else set()
    todo = [r for r in rows if r["id"] not in existing]
    if not todo:
        return client, col, model

    # 按层分组跑 embedding（按 id 分组会导致每组只有 1 条，白白调用几千次 add）
    by_src = {}
    for r in todo:
        by_src.setdefault(layer_of(r), []).append(r)

    # ChromaDB 单次 add 上限 5461 条，L1 有 8000+ 条，必须分批
    BATCH = 4000
    for src, group in by_src.items():
        for start in range(0, len(group), BATCH):
            chunk = group[start:start + BATCH]
            texts = [_embed_text(r) for r in chunk]
            embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                                normalize_embeddings=True)
            col.add(
                ids=[r["id"] for r in chunk],
                documents=[r["text"] for r in chunk],
                embeddings=[e.tolist() for e in embs],
                metadatas=[{
                    "layer": layer_of(r),
                    "lang": r.get("lang", "zh"),
                    "category": r.get("category", ""),
                    "title": (r.get("title") or "")[:120],
                    "title_zh": (r.get("title_zh") or "")[:60],
                    "target": (r.get("target") or "")[:60],
                    "url": r.get("url", ""),
                    "brain_level": int(r.get("brain_level") or 0),
                    "difficulty": int(r.get("difficulty") or 0),
                    "level_req": int(r.get("level_req") or 0),
                    "verified": bool(r.get("verified")),
                } for r in chunk],
            )
            print(f"  + {src} [{start // BATCH + 1}]: {len(chunk)} 条")
    return client, col, model


# ---------------------------------------------------------------- 查询增强

def load_zh2en():
    """官方术语表反向映射：中文名 → 英文名，用于 query 增强

    注意：表里的值常带称号前缀（"「米凯拉的锋刃」玛莲妮亚"），
    用户只会输入"玛莲妮亚"三个字，所以必须同时登记去掉称号的短名。
    """
    flat = json.loads((DATA / "glossary_flat.json").read_text(encoding="utf-8"))

    # 官方术语表只覆盖本体游戏文件，DLC 内容（黄金树之影）不在其中。
    # 手工补充 DLC 名 → 英文名，保证中文提问也能命中英文 wiki 页面。
    alias_file = DATA / "alias_manual.json"
    if alias_file.exists():
        flat.update(json.loads(alias_file.read_text(encoding="utf-8")))

    zh2en = {}
    for en, zh in flat.items():
        zh = zh.strip()
        if not zh:
            continue
        if zh not in zh2en:
            zh2en[zh] = en
        # 去掉称号前缀：兼容 “…” / 「…」 / 『…』三种引号
        m = re.match(r"^[「『“\"][^」』”\"]*[」』”\"](.+)$", zh)
        if m:
            short = m.group(1).strip()
            if short and short not in zh2en:
                zh2en[short] = en
            # 称号本身也要注册：用户可能搜"恶兆王"而不是"蒙葛特"
            title = re.match(r"^[「『“\"]([^」』”\"]*)[」』”\"]", zh)
            if title and len(title.group(1).strip()) >= 2 \
                    and title.group(1).strip() not in zh2en:
                zh2en[title.group(1).strip()] = en
    return zh2en


_ALIAS_CACHE = None


def load_aliases():
    """boss 别名表：{实体名: [俗称...]}，让俗称提问也能命中实体规则。

    为什么单独一张表而不是写进 md 条目里：
      1. 同一 boss 在 L2/L3 可能有两套 target 写法（"“接肢”葛瑞克" /
         "接肢葛瑞克"），别名写进条目要重复维护，写在这里一处即可；
      2. 它是纯查询侧数据——改完不需要重建向量索引，加个俗称立刻生效。
    文件缺失时返回空表，检索退化为「无别名」行为，不会报错。
    """
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        p = DATA / "aliases.json"
        raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        _ALIAS_CACHE = {k: [a for a in v if len(a) >= 2]
                        for k, v in raw.items() if not k.startswith("_")}
    return _ALIAS_CACHE


def enhance_query(q, zh2en):
    """『玛莲妮亚怎么打』→『玛莲妮亚 Malenia 怎么打』"""
    hits = []
    for zh, en in zh2en.items():
        if len(zh) >= 2 and zh in q:
            hits.append((len(zh), zh, en))
    # 最长的先替换，避免短词抢先
    hits.sort(reverse=True)
    injected = set()
    for _, zh, en in hits[:3]:
        # 同一实体的称号与短名会被分别命中（"穿刺者"+"米德拉"），只注入一次
        if en in injected:
            continue
        injected.add(en)
        q = q.replace(zh, f"{zh} {en}")
    return q


# ---------------------------------------------------------------- 规则路由

CHEESE_KEYWORDS = ["逃课", "轮椅", "无脑", "不想努力", "偷懒", "躺赢", "简单打", "轻松"]
BRAIN_KEYWORDS = ["最无脑", "最简单", "最轻松", "手残"]

# 问物品 / 地点 / 掉落 / 剧情，不是问打法。
# 这类问题里常含 boss 名（"玛莲妮亚的追忆能换什么"），实体命中会把打法条目
# 顶到第一——答非所问。识别出来后把 L1 拉满、L2/L3 压到最低。
KNOWLEDGE_KEYWORDS = [
    "在哪", "在哪里", "在哪个", "位置", "怎么去", "怎么走",
    "能换", "换什么", "兑换", "换取",
    "怎么获得", "怎么拿", "哪里拿", "哪里买", "掉落", "掉率",
    "是什么", "剧情", "背景", "来历", "弱点", "抗性", "免疫",
]

# 「不逃课」「正常打」这类表达里也含"逃课"二字，纯关键词匹配会误判成想逃课。
# 否定/常规诉求必须优先于 CHEESE_KEYWORDS 命中。
NORMAL_FORCE = [
    "不逃课", "不想逃课", "不要逃课", "不用逃课", "不靠逃课",
    "正常打", "正常打法", "常规打法", "正经打", "硬打", "实打实",
    "不用轮椅", "不要轮椅", "不卡bug", "不卡 bug", "凭本事", "真本事",
]


def route(query):
    """
    规则路由（纯关键词，不花 LLM 调用）：
        level     → "我 60 级能逃吗"：提取等级，L2 做结构化过滤
        knowledge → 问物品/地点/掉落/剧情，不是问打法：L1 拉满
        brain     → 明确要"最无脑"，L2 按 brain_level 排序
        cheese    → 逃课意图，L2 权重拉满，L3 压到最低
        normal    → 常规攻略问题：L1 + L3 为主，L2 只留一点（万一他其实想逃）
    返回 (intent, w_l1, w_l2, w_l3, max_level)
    """
    m = re.search(r"(\d+)\s*级", query)
    if m and any(k in query for k in ["逃", "打得过", "能打", "可以打", "够"]):
        return "level", 0.20, 0.60, 0.20, int(m.group(1))
    # 「追忆能换什么」「腐败吐息在哪拿」——问的是知识不是打法。
    # 必须排在 cheese/brain 之前吗？不必：这类问句不含逃课词。
    # 但必须排在 normal 之前，否则会落进 normal 的默认权重，
    # 实体命中会把打法条目顶到第一——用户问掉落，拿到的是打法，答非所问。
    if any(k in query for k in KNOWLEDGE_KEYWORDS):
        return "knowledge", 0.85, 0.05, 0.10, None
    # 「怎么打 不逃课」「想学正常打法」→ 常规，L2 压到最低
    if any(k in query for k in NORMAL_FORCE):
        return "normal", 0.45, 0.10, 0.45, None
    if any(k in query for k in BRAIN_KEYWORDS):
        return "brain", 0.15, 0.75, 0.10, None
    if any(k in query for k in CHEESE_KEYWORDS):
        return "cheese", 0.20, 0.65, 0.15, None
    return "normal", 0.45, 0.10, 0.45, None


# ---------------------------------------------------------------- 检索

def target_names(target, title_zh="") -> list:
    """从 target / 标题里拆出实体名片段。

    全角引号也要当分隔符：target 常写成「"米凯拉的锋刃"玛莲妮亚」这种
    称号 + 本名的形式，拆开后"米凯拉的锋刃"和"玛莲妮亚"都能独立命中。

    抽成公共函数是为了让 scripts/check_aliases.py 复用同一套拆解逻辑——
    校验脚本要是自己写一遍正则，迟早和检索器漂移，别名表就会静默失效。
    """
    tgt = (target or "") + " " + (title_zh or "")
    return [t.strip() for t in re.split(r"[：:（）()，,\s「」“”『』]+", tgt)
            if len(t.strip()) >= 2]


def _entity_hit(query, m, alias_map=None) -> float:
    """query 里的 boss 名是否命中该条目的 target。

    这是硬规则而非软加成——实测"大树守卫怎么逃课"与灵马风筝条目的向量相似度
    只有 0.299（模型对"短问句 vs 长文档"匹配弱），而正文里沾边的无关条目
    （"蹲大树"）能到 0.63。调权重救不了，必须按实体直接分组。
    """
    tgt_names = target_names(m.get("target"), m.get("title_zh"))
    # 双向匹配：
    #   正向  target 全名出现在 query 里（"玛莲妮亚怎么打"）
    #   反向  query 里只说了 target 的一部分
    #         - 后缀："石像鬼" ⊂ "英雄石像鬼"（用户省略前缀 英雄/双/老将）
    #         - 前缀："接肢"  ⊂ "接肢葛瑞克"（用户省略后缀 葛瑞克）
    # 只做正向时用户省略前缀就匹配不上；只做后缀时省略后缀同样匹配不上——
    # 后者踩过一次：「接肢怎么正常打」「满月女王怎么打」都匹配不到 L2/L3，
    # 掉进 L1 的《"接肢"贵族后裔》《…的追忆》这类同名词页面。
    # 两档内部都按命中长度分级，原因有两个（各踩过一次）：
    #   exact：引号拆词让复合名拆出短词（"“碎星将军”拉塔恩"→"拉塔恩"），
    #          短词与 7 字全名同档的话，"约定之王拉塔恩"会被一代目抢走 Top1；
    #   part： "弗尔桑克斯"4 字尾部精确命中与"…顿桑克斯"2 字蹭边同档的话，
    #          死龙会被龙王普拉顿桑克斯挤下去。
    exact_lens = [len(t) for t in tgt_names if t in query]
    part_lens = [
        n
        for t in tgt_names if len(t) >= 3        # 太短的 target 拆不出有意义的片段
        for n in (4, 3, 2) if len(t) > n         # 排除整名：整名命中已归 exact 档
        if t[:n] in query or t[-n:] in query
    ]
    # 别名档：用户用的是俗称/称号/常见错写（"女武神""黑剑""接肢"）。
    # 必须排在精确与部分匹配之后——同一条目若同时命中正式名和别名，取高分那个，
    # 但"说对了名字"永远比"说对了外号"更可信。
    alias_lens = [
        len(a)
        for t in tgt_names
        for a in (alias_map or {}).get(t, ())
        if a in query
    ]
    # 全名命中 > 短名/部分命中 > 别名 > 未命中。分层是因为"龙装大树守卫"
    # 与"大树守卫"是两个 boss，部分命中不能享受同等置顶。
    scores = []
    if exact_lens:
        scores.append(2.0 if max(exact_lens) >= 4 else 1.5)
    if part_lens:
        scores.append(1.5 if max(part_lens) >= 4 else 1.0)
    if alias_lens:
        scores.append(ALIAS_SCORE)
    return max(scores) if scores else 0.0


def _recall_structured(col, emb, layer, topk, n_total, query,
                       max_level=None, require=None, exclude=None,
                       entity_boost=True, alias_map=None):
    """L2 / L3 层召回：向量相似度 + 结构化过滤 + 实体命中打分。

    两层的召回逻辑完全一样，区别只在后面排序时用的字段
    （L2 用无脑度，L3 用难度），所以抽成一个函数。

    entity_boost=False 用于评估脚本的消融实验——关掉它才能量化
    "实体命中置顶"到底贡献了多少。线上永远开着。
    """
    res = col.query(query_embeddings=[emb], n_results=min(topk, n_total),
                    where={"layer": layer},
                    include=["documents", "metadatas", "distances"])
    out = []
    for i in range(len(res["ids"][0])):
        m = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        # 结构化过滤（纯向量检索做不到的部分）
        if max_level and m["level_req"] and m["level_req"] > max_level:
            continue
        req_text = "".join(re.findall(r"需要：(.+?)。", doc))
        if exclude and any(x in req_text for x in exclude):
            continue
        if require and not any(x in req_text for x in require):
            continue
        out.append({"id": res["ids"][0][i],
                    "score_raw": 1.0 - res["distances"][0][i],
                    "meta": m, "text": doc,
                    "target_hit": _entity_hit(query, m, alias_map)
                    if entity_boost else 0.0})
    return out


def search(col, model, query, k=FINAL_K, w_l1=None, w_l2=None, w_l3=None,
           zh2en=None, max_level=None, require=None, exclude=None,
           entity_boost=True, use_alias=True):
    """
    三路召回 + 加权融合：L1 通用攻略 / L2 逃课 / L3 常规打法。

    结构化过滤（L2、L3 人工层专属，纯向量检索做不到）:
        max_level: 用户等级，过滤 level_req <= max_level 的打法
        require:   必须包含的道具关键词列表
        exclude:   必须不包含的道具关键词列表

    entity_boost: 关掉可退化为"纯分层加权"，供评估脚本做消融对比
    """
    # 保留用户原话：增强注入的英文名会把原词插断（"黑剑眷属" →
    # "黑剑 black blade眷属 black blade kindred"），实体匹配和路由
    # 都必须基于原话，增强只服务于 embedding。
    orig_query = query
    if zh2en:
        query = enhance_query(query, zh2en)
    emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    alias_map = load_aliases() if use_alias else {}

    n = col.count()
    # 三路召回：每层独立查 top-K。若合成一次查，L1（8000+条）会把
    # L2/L3（人工条目只有几十条）全部挤出榜单——分层加权的意义就在于此
    l1 = []
    res = col.query(query_embeddings=[emb], n_results=min(TOPK_L1, n),
                    where={"layer": "l1"},
                    include=["documents", "metadatas", "distances"])
    for i in range(len(res["ids"][0])):
        l1.append({"id": res["ids"][0][i], "score_raw": 1.0 - res["distances"][0][i],
                   "meta": res["metadatas"][0][i], "text": res["documents"][0][i]})

    l2 = _recall_structured(col, emb, "l2", TOPK_L2, n, orig_query,
                            max_level, require, exclude, entity_boost, alias_map)
    l3 = _recall_structured(col, emb, "l3", TOPK_L3, n, orig_query,
                            max_level, require, exclude, entity_boost, alias_map)

    # 显式传权重时不走路由（评估脚本做权重扫描要用）；否则由 route 决定
    intent, rw1, rw2, rw3, auto_level = route(orig_query)
    w_l1 = rw1 if w_l1 is None else w_l1
    w_l2 = rw2 if w_l2 is None else w_l2
    w_l3 = rw3 if w_l3 is None else w_l3
    if max_level is None:
        max_level = auto_level

    # L1：常规相似度
    for it in l1:
        it["score"] = w_l1 * it["score_raw"]
    # L2：实体命中直接置顶（base=2 保证压过一切未命中条目），
    # brain/level 意图下无脑度主导档位
    for it in l2:
        base = it["target_hit"]
        if intent == "knowledge":
            # 问"追忆能换什么"时，打法条目只是背景信息，不享受实体置顶
            it["score"] = w_l2 * it["score_raw"]
        elif intent in ("brain", "level"):
            # 无脑度只能在"目标正确"的前提下参与排序，绝不能反过来压过目标匹配。
            # 曾经的 bug：无脑度按 1.0/级 直接相加，于是
            #   大树守卫（命中 2.0 + 无脑 3）= 5.0
            #   熔炉骑士（命中 0   + 无脑 5）= 5.0
            # 两者基数打平，最后被向量相似度反超——用户问大树守卫，拿到熔炉骑士的打法。
            # 改成以 3 为中心、幅度只有 ±1，永远小于实体命中的 2.0 档差。
            it["score"] = w_l2 * (base + (it["meta"]["brain_level"] - 3) * 0.5
                                  + it["score_raw"])
        else:
            it["score"] = w_l2 * (base + it["score_raw"])
    # L3：常规打法不需要按难度排序——用户问"怎么打"时想要的是对应 boss 的打法，
    # 不是"最简单那个"（那是 L2 的活）。实体命中 + 相似度即可。
    for it in l3:
        base = 0.0 if intent == "knowledge" else it["target_hit"]
        it["score"] = w_l3 * (base + it["score_raw"])

    merged = l1 + l2 + l3
    merged.sort(key=lambda x: -x["score"])
    return merged[:k], {"query_enhanced": query, "intent": intent,
                        "w_l1": w_l1, "w_l2": w_l2, "w_l3": w_l3,
                        "max_level": max_level,
                        "n_l1": len(l1), "n_l2": len(l2), "n_l3": len(l3)}


# ---------------------------------------------------------------- CLI 自测

def _cli():
    q = sys.argv[1] if len(sys.argv) > 1 else "玛莲妮亚怎么打"
    show = 6
    if "--show" in sys.argv:
        show = int(sys.argv[sys.argv.index("--show") + 1])

    print("=" * 62)
    print(f"查询: {q}")
    print("=" * 62)

    client, col, model = build_index()
    info = col.count()
    print(f"向量库: {info:,} 条\n")

    zh2en = load_zh2en()
    results, diag = search(col, model, q, k=show, zh2en=zh2en)
    print(f"[增强后 query] {diag['query_enhanced']}")
    lv = f" | 等级上限 {diag['max_level']}" if diag.get("max_level") else ""
    print(f"[路由] {diag['intent']}  (L1 {diag['w_l1']} / L2 {diag['w_l2']} "
          f"/ L3 {diag['w_l3']}{lv})")
    print(f"[召回] L1 {diag['n_l1']} 条 | L2 {diag['n_l2']} 条 "
          f"| L3 {diag['n_l3']} 条\n")

    tags = {"l2": "★L2逃课", "l3": "⚔L3常规", "l1": " L1攻略"}
    for rank, r in enumerate(results, 1):
        m = r["meta"]
        tag = tags.get(m["layer"], " ?")
        title = m["title_zh"] or m["title"]
        extra = ""
        if m["layer"] == "l2":
            extra = f" | 无脑度{m['brain_level']}"
        elif m["layer"] == "l3":
            extra = f" | 难度{m['difficulty']}"
        print(f"{rank:>2}. [{tag}] ({r['score']:.3f}) {title}{extra}")
        print(f"    {r['text'][:110]}...")
    print()


if __name__ == "__main__":
    _cli()
