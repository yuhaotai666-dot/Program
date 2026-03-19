"""
api/routes.py ── 所有 FastAPI 路由
业务逻辑调用 core/ 中的纯函数，资源通过 state.res 访问。
"""
import io
import re
import json
from collections import Counter
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File, HTTPException

from state import res, history_store
from config import RERANK_CANDIDATE_K, FAISS_INDEX_ROWS, LLM_MODEL
from core.embedder import encode_query
from core.retriever import (
    recall_with_scores,
    do_retrieval_hybrid,
    rerank_by_keyword,
    extract_required_skills,
)
from core.generator import do_generate, do_generate_chat, parse_resume_structured
from api.models import (
    SearchRequest, SearchResponse,
    ChatRequest,
    JobSearchRequest, SkillMatchRequest,
)

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

router = APIRouter()


# ── 工具函数（仅本文件使用） ──────────────────────────────

def parse_salary_range(salary_str: str) -> str:
    nums = re.findall(r'\$(\d+)K', str(salary_str))
    if not nums:
        return "Not specified"
    avg = sum(int(n) for n in nums) / len(nums)
    if avg < 50:   return "<$50K"
    if avg < 80:   return "$50K-$80K"
    if avg < 120:  return "$80K-$120K"
    if avg < 150:  return "$120K-$150K"
    return ">$150K"


def _compute_stats(matched_df) -> dict:
    """从匹配到的 DataFrame 计算统计数据（技能 Top5 / 地点 / 薪资 / 工作类型）"""
    all_skills = []
    for s in matched_df['skills']:
        all_skills.extend([x.strip() for x in str(s).split(',') if x.strip()])
    top_skills = [[k, v] for k, v in Counter(all_skills).most_common(5)]

    locations  = matched_df['location'].value_counts().head(8).to_dict()
    locations  = {str(k): int(v) for k, v in locations.items() if k}

    salary_buckets = {"<$50K": 0, "$50K-$80K": 0, "$80K-$120K": 0,
                      "$120K-$150K": 0, ">$150K": 0, "Not specified": 0}
    for sal in matched_df['Salary Range']:
        b = parse_salary_range(sal)
        salary_buckets[b] = salary_buckets.get(b, 0) + 1
    salary_ranges = {k: v for k, v in salary_buckets.items() if v > 0}

    work_types = matched_df['Work Type'].value_counts().head(5).to_dict()
    work_types = {str(k): int(v) for k, v in work_types.items() if k}

    return {
        "total_count":   len(matched_df),
        "top_skills":    top_skills,
        "locations":     locations,
        "salary_ranges": salary_ranges,
        "work_types":    work_types,
    }


def _row_to_job(row) -> dict:
    return {
        "job_title":   str(row['Job Title']),
        "company":     str(row['Company']),
        "location":    str(row['location']),
        "country":     str(row['Country']),
        "salary":      str(row['Salary Range']),
        "work_type":   str(row['Work Type']),
        "experience":  str(row['Experience']),
        "skills":      str(row['skills']),
        "description": str(row['Job Description'])[:200],
    }


# ═══════════════════════════════════════════════════════
# 基础路由
# ═══════════════════════════════════════════════════════

@router.get("/")
def root():
    return {"status": "ok", "message": "职位 RAG 系统运行中"}


@router.get("/api/health")
def health():
    return {
        "status":       "ok",
        "vector_count": res.index.ntotal if res.index else 0,
        "text_count":   len(res.texts) if res.texts else 0,
        "bm25_ready":   res.bm25 is not None,
        "job_count":    len(res.df_jobs) if res.df_jobs is not None else 0,
    }


# ═══════════════════════════════════════════════════════
# Function 1：职位分析（RAG + Rerank）
# ═══════════════════════════════════════════════════════

@router.post("/api/search-jobs")
async def search_jobs(req: JobSearchRequest):
    """
    ① FAISS+BM25 向量召回候选
    ② Pandas 关键词召回（全量 50k）
    ③ 合并去重 → rerank_by_keyword
    ④ 统计数据基于 Pandas 全量匹配
    """
    if not req.job_title.strip():
        raise HTTPException(status_code=400, detail="职位名称不能为空")

    df    = res.df_jobs
    query = req.job_title.strip()

    # Step 1：向量召回（async 路由在事件循环主线程执行，避免 macOS SIGSEGV）
    query_vec = encode_query(query, res.tokenizer, res.model)
    fused_scores = recall_with_scores(query, query_vec, res.index, res.bm25,
                                      k=RERANK_CANDIDATE_K)

    # Step 2：Pandas 关键词召回（用于统计 + 补充候选）
    mask_title = df['Job Title'].str.contains(query, case=False, na=False)
    mask_desc  = df['Job Description'].str.contains(query, case=False, na=False)
    matched    = df[mask_title | mask_desc]
    pandas_idx = set(df.index[mask_title].tolist()[:RERANK_CANDIDATE_K])

    # Step 3：合并候选，去重
    all_candidate_idx = set(fused_scores.keys()) | pandas_idx
    candidates = []
    for row_idx in all_candidate_idx:
        if row_idx >= len(df):
            continue
        row = df.iloc[row_idx]
        candidates.append({
            "semantic_score": fused_scores.get(row_idx, 0.0),
            "job": _row_to_job(row),
        })

    # Step 4：Rerank
    reranked = rerank_by_keyword(query, candidates)
    jobs     = [c["job"] for c in reranked[:req.top_n]]

    # Step 5：统计（基于 Pandas 全量，保证数字准确）
    if len(matched) == 0:
        matched = df.iloc[list(fused_scores.keys())]
    stats = _compute_stats(matched)

    return {
        "jobs":      jobs,
        "stats":     stats,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/api/random-jobs")
def random_jobs(n: int = 10):
    """随机样本，用于页面初始加载"""
    df     = res.df_jobs
    sample = df.sample(n=min(n, len(df)), random_state=None)
    jobs   = [_row_to_job(row) for _, row in sample.iterrows()]

    # 全局统计（基于前 5000 条，速度快）
    stats = _compute_stats(df.head(5000))
    stats["total_count"] = len(df)   # 总数显示真实值
    return {"jobs": jobs, "stats": stats}


# ═══════════════════════════════════════════════════════
# Function 2：Skill Match & Learning Path（RAG grounding 防幻觉）
# ═══════════════════════════════════════════════════════

@router.post("/api/skill-match")
async def skill_match(req: SkillMatchRequest):
    """
    ① RAG 召回真实岗位 → 提取高频真实技能（grounding）
    ② 数据驱动计算基准匹配分，约束 LLM 不随意打分
    ③ Prompt 约束：skill_gaps 只能从真实技能列表中选取
    """
    if not req.target_job.strip():
        raise HTTPException(status_code=400, detail="目标职位不能为空")

    query     = f"{req.target_job} {req.current_major}"
    query_vec = encode_query(query, res.tokenizer, res.model)

    # RAG 召回原文（作为 LLM context）
    retrieved = do_retrieval_hybrid(query_vec, query, res.index, res.bm25, res.texts)
    context   = "\n".join([
        f"[职位{i+1}（相关度{r['score']:.2f}）]: {r['text']}"
        for i, r in enumerate(retrieved)
    ])

    # 从真实岗位数据提取高频技能（grounding）
    real_skills = extract_required_skills(
        req.target_job, query_vec, res.df_jobs, res.index, res.bm25, top_n=20
    )

    # 数据驱动：匹配分析
    user_skill_set = {s.strip().lower() for s in req.current_skills.split(',') if s.strip()}
    data_matched   = [s for s in real_skills if s in user_skill_set]
    data_gaps      = [s for s in real_skills if s not in user_skill_set][:10]
    base_score     = int(len(data_matched) / min(len(real_skills), 10) * 100) if real_skills else 50

    grounding_block = f"""【真实招聘数据技能要求（{len(real_skills)}项，按需求频率排序）】
{', '.join(real_skills) if real_skills else '数据不足，请参考职位原文'}

【数据统计结果（不可随意更改）】
- 数据驱动匹配技能：{', '.join(data_matched) if data_matched else '暂无'}
- 数据驱动技能差距：{', '.join(data_gaps) if data_gaps else '暂无'}
- 数据驱动基准匹配度：{base_score}分"""

    prompt = f"""你是一个职业规划专家。请基于以下真实招聘数据分析求职者的匹配程度。

求职者信息：
- 专业背景：{req.current_major}
- 现有技能：{req.current_skills}
目标职位：{req.target_job}

{grounding_block}

【真实职位原文参考】
{context}

⚠️ 约束（必须遵守）：
1. matched_skills 和 skill_gaps 必须从真实技能列表中选取，不可编造
2. match_score 以基准分 {base_score} 为参考，可 ±15 分调整
3. learning_roadmap 中的技能优先来自数据驱动差距列表

请以JSON格式返回，不要输出其他内容：
{{
  "match_score": 匹配度（0-100，参考基准分{base_score}）,
  "match_analysis": "2-3句分析，需引用真实技能数据",
  "matched_skills": ["已具备的真实要求技能"],
  "skill_gaps": ["缺失的真实要求技能"],
  "learning_roadmap": [
    {{
      "phase": "第一阶段", "duration": "1-2个月", "goal": "阶段目标",
      "skills": [{{"name": "技能名", "description": "1句话说明", "bilibili_keyword": "B站搜索词"}}]
    }},
    {{
      "phase": "第二阶段", "duration": "2-3个月", "goal": "阶段目标",
      "skills": [{{"name": "技能名", "description": "1句话说明", "bilibili_keyword": "B站搜索词"}}]
    }},
    {{
      "phase": "第三阶段", "duration": "3-6个月", "goal": "阶段目标",
      "skills": [{{"name": "技能名", "description": "1句话说明", "bilibili_keyword": "B站搜索词"}}]
    }}
  ],
  "advice": "2-3句综合建议，结合真实市场数据"
}}"""

    try:
        resp = res.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = re.sub(r'^```json\s*|\s*```$', '', resp.choices[0].message.content.strip())
        result  = json.loads(content)

        for phase in result.get("learning_roadmap", []):
            for skill in phase.get("skills", []):
                kw = skill.get("bilibili_keyword", skill["name"])
                skill["bilibili_url"] = f"https://search.bilibili.com/all?keyword={quote(kw)}&order=totalrank"

        result["_grounding_info"] = {
            "real_skills_found":  len(real_skills),
            "data_matched_count": len(data_matched),
            "data_gaps_count":    len(data_gaps),
            "base_score":         base_score,
        }
        return {"result": result, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    except Exception as e:
        import traceback
        traceback.print_exc()   # 在后端终端打印完整错误堆栈
        raise HTTPException(status_code=500, detail=f"skill_match 失败：{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════
# RAG 对话接口
# ═══════════════════════════════════════════════════════

@router.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """单轮检索"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    query_vec = encode_query(req.query, res.tokenizer, res.model)
    retrieved = do_retrieval_hybrid(query_vec, req.query, res.index, res.bm25, res.texts)
    answer    = do_generate(req.query, retrieved, res.llm)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history_store.setdefault(req.user_id, []).append({
        "query": req.query, "results": retrieved,
        "answer": answer, "timestamp": timestamp,
    })
    return SearchResponse(query=req.query, results=retrieved, answer=answer, timestamp=timestamp)


@router.post("/api/chat", response_model=SearchResponse)
async def chat(req: ChatRequest):
    """多轮对话"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    query_vec = encode_query(req.query, res.tokenizer, res.model)
    retrieved = do_retrieval_hybrid(query_vec, req.query, res.index, res.bm25, res.texts)
    history   = [{"role": m.role, "content": m.content} for m in req.history]
    answer    = do_generate_chat(req.query, retrieved, history, res.llm)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return SearchResponse(query=req.query, results=retrieved, answer=answer, timestamp=timestamp)


@router.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...), user_id: str = "anonymous"):
    """上传简历：结构化解析 → 混合检索 → AI 推荐"""
    content  = await file.read()
    filename = file.filename.lower()
    text = ""

    if filename.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        if not HAS_PDF:
            raise HTTPException(status_code=400, detail="服务器未安装 PyPDF2，请上传 txt 文件")
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith(".docx"):
        if not HAS_DOCX:
            raise HTTPException(status_code=400, detail="服务器未安装 python-docx，请上传 txt 文件")
        doc  = Document(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise HTTPException(status_code=400, detail="仅支持 PDF / DOCX / TXT 格式")

    if not text.strip():
        raise HTTPException(status_code=400, detail="无法提取文件文字内容")

    parsed  = parse_resume_structured(text, res.llm)
    desired = "、".join(parsed.get("desired_positions", []))
    skills  = "、".join(parsed.get("skills", [])[:5])
    query   = f"{desired} {skills}".strip() or text[:500]

    query_vec = encode_query(query, res.tokenizer, res.model)
    retrieved = do_retrieval_hybrid(query_vec, query, res.index, res.bm25, res.texts)
    answer    = do_generate(f"根据简历内容（{parsed.get('summary', '')}）推荐合适职位", retrieved, res.llm)

    return {
        "parsed_resume":          parsed,
        "extracted_text_preview": text[:200] + "...",
        "results":                retrieved,
        "answer":                 answer,
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════
# 历史记录
# ═══════════════════════════════════════════════════════

@router.get("/api/history/{user_id}")
def get_history(user_id: str, limit: int = 10):
    records = history_store.get(user_id, [])
    return {"user_id": user_id, "total": len(records), "records": records[-limit:][::-1]}


@router.delete("/api/history/{user_id}")
def clear_history(user_id: str):
    history_store.pop(user_id, None)
    return {"message": f"用户 {user_id} 的历史已清空"}
