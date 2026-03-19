"""
core/chunker.py ── Semantic Chunking：以职位为单位的语义切分
"""
import re
import pandas as pd
from transformers import BertTokenizer

from config import TOKEN_BUDGET


def split_sentences(text: str) -> list[str]:
    """
    按句末标点（. ! ?）+ 空格切分英文文本，不依赖 nltk。
    保留每个句子末尾的标点。
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def make_semantic_chunk(row: pd.Series, tokenizer: BertTokenizer) -> str:
    """
    将一条职位 DataFrame 行转换为一个完整语义块。

    策略：
      ① 技能列表：按逗号边界完整保留技能名，不在名称中间截断
      ② 职位描述：按句子边界贪心填充，直到达到 TOKEN_BUDGET
      ③ fallback：无句末标点时按词截断
    """
    title       = str(row.get("Job Title",      "")).strip()
    work_type   = str(row.get("Work Type",       "")).strip()
    experience  = str(row.get("Experience",      "")).strip()
    raw_skills  = str(row.get("skills",          "")).strip()
    description = str(row.get("Job Description", "")).strip()

    # ── ① 技能：按逗号切分，贪心保留完整技能名 ──
    skill_list = [
        s.strip() for s in raw_skills.split(",")
        if s.strip() and s.strip().lower() not in ("nan", "none", "")
    ]
    skill_budget = TOKEN_BUDGET // 3
    chosen_skills, used = [], 0
    for sk in skill_list:
        t = len(tokenizer.encode(sk, add_special_tokens=False))
        if used + t > skill_budget:
            break
        chosen_skills.append(sk)
        used += t
    skills_str = ", ".join(chosen_skills) if chosen_skills else raw_skills[:150]

    # ── 结构化头部 ──
    meta = (f"Title: {title} | "
            f"Type: {work_type} | "
            f"Experience: {experience} | "
            f"Skills: {skills_str}")

    # ── ② 描述：按句子边界贪心填充 ──
    meta_tokens = len(tokenizer.encode(meta, add_special_tokens=False))
    desc_budget = max(TOKEN_BUDGET - meta_tokens, 50)

    sentences = split_sentences(description)
    collected, used_desc = [], 0
    for sent in sentences:
        t = len(tokenizer.encode(sent, add_special_tokens=False))
        if used_desc + t > desc_budget:
            break
        collected.append(sent)
        used_desc += t

    # ── ③ fallback：无句末标点，按词截断 ──
    if not collected and description:
        words = description.split()
        fb, used_fb = [], 0
        for w in words:
            t = len(tokenizer.encode(w, add_special_tokens=False))
            if used_fb + t > desc_budget:
                break
            fb.append(w)
            used_fb += t
        collected = [" ".join(fb)]

    desc_str = " ".join(collected)
    return f"{meta} | Description: {desc_str}"
