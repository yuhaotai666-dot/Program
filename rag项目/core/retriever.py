"""
core/retriever.py ── 检索核心：BM25 + FAISS 混合召回、Rerank、技能提取
所有函数均为纯函数（显式传参，不依赖全局 res），便于单元测试和复用。
"""
import re
import numpy as np
from collections import Counter
from rank_bm25 import BM25Okapi

from config import (
    TOP_K, THRESHOLD,
    FAISS_WEIGHT, BM25_WEIGHT,
    RERANK_CANDIDATE_K, FAISS_INDEX_ROWS,
)


# ── BM25 分词：中文按字切，英文按词切 ──────────────────────
def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())


# ── 融合召回（FAISS + BM25） ───────────────────────────────
def recall_with_scores(query: str,
                       query_vec: np.ndarray,
                       index,
                       bm25: BM25Okapi,
                       k: int = RERANK_CANDIDATE_K) -> dict[int, float]:
    """
    返回 {row_idx: 融合分数}，row_idx 直接对应 df_jobs.iloc[i]。
    FAISS 分数和 BM25 分数各自归一化到 [0,1] 后加权求和。
    """
    # FAISS 语义检索
    faiss_scores, faiss_indices = index.search(query_vec, min(k, index.ntotal))
    faiss_max  = max(float(faiss_scores[0].max()), 1e-9)
    faiss_norm = {
        int(idx): float(s) / faiss_max
        for s, idx in zip(faiss_scores[0], faiss_indices[0]) if idx != -1
    }

    # BM25 关键词检索
    query_tokens = tokenize_for_bm25(query)
    bm25_scores  = bm25.get_scores(query_tokens)
    bm25_max     = max(float(bm25_scores.max()), 1e-9)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_norm    = {int(i): float(bm25_scores[i]) / bm25_max for i in top_bm25_idx}

    # 加权融合
    all_idx = set(faiss_norm) | set(bm25_norm)
    return {
        idx: faiss_norm.get(idx, 0.0) * FAISS_WEIGHT + bm25_norm.get(idx, 0.0) * BM25_WEIGHT
        for idx in all_idx
    }


def do_retrieval_hybrid(query_vec: np.ndarray,
                        query_text: str,
                        index,
                        bm25: BM25Okapi,
                        texts: list) -> list[dict]:
    """
    完整混合检索：召回 → 阈值过滤 → 返回 top-K。
    结果格式：[{"text": str, "score": float}, ...]
    """
    candidate_k = min(TOP_K * 4, len(texts))
    fused = recall_with_scores(query_text, query_vec, index, bm25, k=candidate_k)

    results = []
    for idx, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:TOP_K]:
        if score < THRESHOLD:
            continue
        results.append({"text": texts[idx], "score": round(score, 4)})
    return results


# ── Rerank：精确关键词匹配优先 ────────────────────────────
def rerank_by_keyword(query: str, candidates: list[dict]) -> list[dict]:
    """
    输入：[{"job": {...行数据}, "semantic_score": float}, ...]
    输出：按 (keyword_score, final_score) 降序排列的同结构列表

    优先级：
      完全精确匹配(1.0) > 标题包含完整查询串(0.9) >
      标题包含所有查询词(0.8) > 标题包含任意查询词(0.6) >
      纯语义分(0-0.5)
    """
    q       = query.strip().lower()
    q_words = [w for w in q.split() if len(w) > 1]

    for c in candidates:
        title = str(c["job"].get("job_title", "")).lower()
        if q == title:
            kw = 1.0
        elif q in title:
            kw = 0.9
        elif q_words and all(w in title for w in q_words):
            kw = 0.8
        elif q_words and any(w in title for w in q_words):
            kw = 0.6
        else:
            kw = 0.0
        c["keyword_score"] = kw
        c["final_score"]   = kw if kw > 0 else c["semantic_score"] * 0.5

    return sorted(candidates,
                  key=lambda x: (x["keyword_score"], x["final_score"]),
                  reverse=True)


# ── 真实技能提取（防幻觉 grounding 数据） ─────────────────
def extract_required_skills(target_job: str,
                             query_vec: np.ndarray,
                             df_jobs,
                             index,
                             bm25: BM25Okapi,
                             top_n: int = 15) -> list[str]:
    """
    从两个来源提取目标岗位的真实高频技能：
      ① FAISS+BM25 向量召回的岗位（覆盖前 FAISS_INDEX_ROWS 行）
      ② Pandas 关键词精确匹配的岗位（覆盖全量 df_jobs）
    返回按频率排序的技能列表（小写）。
    """
    fused = recall_with_scores(target_job, query_vec, index, bm25, k=RERANK_CANDIDATE_K)
    skill_counter: Counter = Counter()

    # 来源 ①：向量召回
    for idx in fused:
        if idx < FAISS_INDEX_ROWS and idx < len(df_jobs):
            for s in str(df_jobs.iloc[idx]["skills"]).split(","):
                s = s.strip()
                if s and s.lower() not in ("nan", "none", ""):
                    skill_counter[s.lower()] += 1

    # 来源 ②：pandas 精确匹配
    mask = df_jobs["Job Title"].str.contains(target_job, case=False, na=False)
    for skills_str in df_jobs[mask]["skills"].head(80):
        for s in str(skills_str).split(","):
            s = s.strip()
            if s and s.lower() not in ("nan", "none", ""):
                skill_counter[s.lower()] += 1

    return [skill for skill, _ in skill_counter.most_common(top_n)]
