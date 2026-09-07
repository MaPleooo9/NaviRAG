# -*- coding: utf-8 -*-
"""
分层检索器：L1 wiki（英/中）+ L2 逃课层 两路召回、加权融合。

为什么是"分层"而不是一把梭：
    L1 是通用攻略，L2 是人工整理的逃课技巧。用户问"怎么打"时 L1 命中，
    问"怎么逃"时必须命中 L2——但 L2 表述和"怎么打"的主流段落语义相似度并不高，
    纯向量检索会把 L2 淹没。所以分开召回、按层加权，这也是本项目的核心实验变量。

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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "elden_ring"
DB_DIR = ROOT / ".vectorstore"

# 优先用本地模型（scripts/download_model.py 下载），否则从 HF 拉取
LOCAL_MODEL = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = str(LOCAL_MODEL) if LOCAL_MODEL.exists() \
    else "paraphrase-multilingual-MiniLM-L12-v2"

# 默认权重：L2 是核心资产，给更高权重。这是评估实验要扫的参数。
W_L1, W_L2 = 0.4, 0.6

TOPK_L1 = 12      # L1 层召回数（8000+ 条，取多了慢且没用）
TOPK_L2 = 300     # L2 层全量召回（人工条目总共几百条，漏一条就是漏一个打法）
FINAL_K = 6       # 最终返回多少条


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


def build_index(model=None, force=False):
    """构建/增量更新 ChromaDB。返回 (client, collection, model)"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(
        "elden_ring", metadata={"hnsw:space": "cosine"})

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    if force:
        client.delete_collection("elden_ring")
        col = client.get_or_create_collection(
            "elden_ring", metadata={"hnsw:space": "cosine"})

    rows = _load_jsonl(DATA / "l1_wiki.jsonl") + _load_jsonl(DATA / "l1_zh_game.jsonl") \
        + _load_jsonl(DATA / "l2_cheese.jsonl")

    # 增量：跳过已有 id
    existing = set(col.get()["ids"]) if col.count() else set()
    todo = [r for r in rows if r["id"] not in existing]
    if not todo:
        return client, col, model

    layer = "l2" if "brain_level" in todo[0] else "l1"
    # 按来源分组跑（l1 两个文件 + l2）
    by_src = {}
    for r in todo:
        by_src.setdefault(r["id"].split(":")[0], []).append(r)

    import numpy as np
    for src, group in by_src.items():
        texts = [_embed_text(r) for r in group]
        embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True)
        col.add(
            ids=[r["id"] for r in group],
            documents=[r["text"] for r in group],
            embeddings=[e.tolist() for e in embs],
            metadatas=[{
                "layer": "l2" if "brain_level" in r else "l1",
                "lang": r.get("lang", "zh"),
                "category": r.get("category", ""),
                "title": (r.get("title") or "")[:120],
                "title_zh": (r.get("title_zh") or "")[:60],
                "target": (r.get("target") or "")[:60],
                "url": r.get("url", ""),
                "brain_level": int(r.get("brain_level") or 0),
                "level_req": int(r.get("level_req") or 0),
                "verified": bool(r.get("verified")),
            } for r in group],
        )
        print(f"  + {src}: {len(group)} 条")
    return client, col, model


# ---------------------------------------------------------------- 查询增强

def load_zh2en():
    """官方术语表反向映射：中文名 → 英文名，用于 query 增强

    注意：表里的值常带称号前缀（"「米凯拉的锋刃」玛莲妮亚"），
    用户只会输入"玛莲妮亚"三个字，所以必须同时登记去掉称号的短名。
    """
    flat = json.loads((DATA / "glossary_flat.json").read_text(encoding="utf-8"))
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


def enhance_query(q, zh2en):
    """『玛莲妮亚怎么打』→『玛莲妮亚 Malenia 怎么打』"""
    hits = []
    for zh, en in zh2en.items():
        if len(zh) >= 2 and zh in q:
            hits.append((len(zh), zh, en))
    # 最长的先替换，避免短词抢先
    hits.sort(reverse=True)
    for _, zh, en in hits[:3]:
        q = q.replace(zh, f"{zh} {en}")
    return q


# ---------------------------------------------------------------- 规则路由

CHEESE_KEYWORDS = ["逃课", "轮椅", "无脑", "不想努力", "偷懒", "躺赢", "简单打", "轻松"]
BRAIN_KEYWORDS = ["最无脑", "最简单", "最轻松", "手残"]


def route(query):
    """
    规则路由（纯关键词，不花 LLM 调用）：
        level  → "我 60 级能逃吗"：提取等级，L2 做结构化过滤
        brain  → 明确要"最无脑"，L2 按 brain_level 排序
        cheese → 逃课意图，L2 权重拉高
        normal → 常规攻略问题，L1 为主
    返回 (intent, w_l1, w_l2, max_level)
    """
    m = re.search(r"(\d+)\s*级", query)
    if m and any(k in query for k in ["逃", "打得过", "能打", "可以打", "够"]):
        return "level", 0.25, 0.75, int(m.group(1))
    if any(k in query for k in BRAIN_KEYWORDS):
        return "brain", 0.25, 0.75, None
    if any(k in query for k in CHEESE_KEYWORDS):
        return "cheese", 0.3, 0.7, None
    return "normal", 0.7, 0.3, None


# ---------------------------------------------------------------- 检索

def search(col, model, query, k=FINAL_K, w_l1=W_L1, w_l2=W_L2, zh2en=None,
           max_level=None, require=None, exclude=None):
    """
    分层检索 + 加权融合。

    结构化过滤（L2 专属，纯向量检索做不到）:
        max_level: 用户等级，过滤 level_req <= max_level 的逃课打法
        require:   必须包含的道具关键词列表
        exclude:   必须不包含的道具关键词列表
    """
    if zh2en:
        query = enhance_query(query, zh2en)
    emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    n = col.count()
    # 两路召回：每层独立查 top-K。若合成一次查，L1（8000+条）会把 L2（23条）
    # 全部挤出榜单——分层加权的意义就在于此
    l1, l2 = [], []
    res = col.query(query_embeddings=[emb], n_results=min(TOPK_L1, n),
                    where={"layer": "l1"},
                    include=["documents", "metadatas", "distances"])
    for i in range(len(res["ids"][0])):
        l1.append({"id": res["ids"][0][i], "score_raw": 1.0 - res["distances"][0][i],
                   "meta": res["metadatas"][0][i], "text": res["documents"][0][i]})

    res = col.query(query_embeddings=[emb], n_results=min(TOPK_L2, n),
                    where={"layer": "l2"},
                    include=["documents", "metadatas", "distances"])
    for i in range(len(res["ids"][0])):
        m = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        # L2 结构化过滤（纯向量检索做不到的部分）
        if max_level and m["level_req"] and m["level_req"] > max_level:
            continue
        req_text = "".join(re.findall(r"需要：(.+?)。", doc))
        if exclude and any(x in req_text for x in exclude):
            continue
        if require and not any(x in req_text for x in require):
            continue
        # 实体匹配：query 里的 boss 名是否出现在该条目的 target 中。
        # 这是硬规则而非软加成——实测"大树守卫怎么逃课"与灵马风筝条目的
        # 向量相似度只有 0.299（模型对"短问句 vs 长文档"匹配弱），
        # 而 content 词沾边的外部条目（"蹲大树"）能到 0.63。调权重救不了，
        # 必须按实体直接分组。
        tgt = (m.get("target") or "") + " " + (m.get("title_zh") or "")
        tgt_names = [t.strip() for t in re.split(r"[：:（）()，,\s]+", tgt)
                     if len(t.strip()) >= 2]
        hit = any(t in query for t in tgt_names)
        l2.append({"id": res["ids"][0][i], "score_raw": 1.0 - res["distances"][0][i],
                   "meta": m, "text": doc, "target_hit": hit})

    intent, w_l1, w_l2, auto_level = route(query)
    if max_level is None:
        max_level = auto_level

    # L1：常规相似度
    for it in l1:
        it["score"] = w_l1 * it["score_raw"]
    # L2：实体命中直接置顶（base=2 保证压过一切未命中条目），
    # brain/level 意图下无脑度主导档位
    for it in l2:
        base = 2.0 if it["target_hit"] else 0.0
        if intent in ("brain", "level"):
            it["score"] = w_l2 * (base + it["meta"]["brain_level"] + it["score_raw"])
        else:
            it["score"] = w_l2 * (base + it["score_raw"])

    merged = l1 + l2
    merged.sort(key=lambda x: -x["score"])
    return merged[:k], {"query_enhanced": query, "intent": intent,
                        "w_l1": w_l1, "w_l2": w_l2, "max_level": max_level,
                        "n_l1": len(l1), "n_l2": len(l2)}


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
    print(f"[路由] {diag['intent']}  (L1 权重 {diag['w_l1']} / L2 权重 {diag['w_l2']}{lv})")
    print(f"[召回] L1 {diag['n_l1']} 条 | L2 {diag['n_l2']} 条\n")

    for rank, r in enumerate(results, 1):
        m = r["meta"]
        tag = "★L2逃课" if m["layer"] == "l2" else " L1攻略"
        title = m["title_zh"] or m["title"]
        print(f"{rank:>2}. [{tag}] ({r['score']:.3f}) {title}"
              + (f" | 无脑度{m['brain_level']}" if m["layer"] == "l2" else ""))
        print(f"    {r['text'][:110]}...")
    print()


if __name__ == "__main__":
    _cli()
