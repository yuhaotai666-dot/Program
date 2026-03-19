"""
core/generator.py ── LLM 生成：单轮、多轮对话、简历解析
所有函数均为纯函数（llm 显式传参）。
"""
import re
import json
from openai import OpenAI

from config import LLM_MODEL


def do_generate(query: str, retrieved: list[dict], llm: OpenAI) -> str:
    """单轮生成：检索结果 → AI 职业建议"""
    if not retrieved:
        return "未找到相关职位，请尝试换个关键词，例如：数据分析、软件开发、市场营销。"

    context = "\n".join([
        f"[职位{i+1}（相关度{r['score']:.2f}）]: {r['text']}"
        for i, r in enumerate(retrieved)
    ])
    prompt = f"""你是一个专业的职业顾问。根据以下检索到的职位信息，回答用户的问题。

【用户需求】{query}

【相关职位信息】
{context}

请给出专业建议，包括：
1. 最匹配的职位方向
2. 典型工作内容
3. 简短求职建议

回答简洁专业，150字以内。"""

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def do_generate_chat(query: str,
                     retrieved: list[dict],
                     history: list[dict],
                     llm: OpenAI) -> str:
    """多轮对话生成：携带最近 3 轮历史上下文"""
    if not retrieved:
        return "未找到相关职位，请尝试换个关键词，例如：数据分析、软件开发、市场营销。"

    context = "\n".join([
        f"[职位{i+1}（相关度{r['score']:.2f}）]: {r['text']}"
        for i, r in enumerate(retrieved)
    ])
    system_msg = {
        "role": "system",
        "content": "你是一个专业的职业顾问。根据检索到的职位信息，结合对话历史回答用户问题。回答简洁专业，150字以内。"
    }
    messages = [system_msg] + history[-6:]   # 最近 3 轮（6条消息）
    messages.append({
        "role": "user",
        "content": f"【用户需求】{query}\n\n【相关职位信息】\n{context}\n\n请给出职位方向、工作内容、求职建议。"
    })

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def parse_resume_structured(text: str, llm: OpenAI) -> dict:
    """用 LLM 从简历文本中提取结构化信息"""
    prompt = f"""请从以下简历中提取结构化信息，以JSON格式返回，不要输出其他内容：
{{
  "skills": ["技能1", "技能2"],
  "experience_years": 数字或null,
  "desired_positions": ["期望职位1", "期望职位2"],
  "education": "最高学历",
  "summary": "50字以内的简历摘要"
}}

简历内容：
{text[:2000]}"""

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    try:
        content = resp.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content)
        return json.loads(content)
    except Exception:
        return {
            "skills": [], "experience_years": None,
            "desired_positions": [], "education": "", "summary": text[:100]
        }
