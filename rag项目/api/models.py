"""
api/models.py ── 所有 Pydantic 请求 / 响应模型
"""
from typing import List
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query:   str
    user_id: str = "anonymous"


class SearchResponse(BaseModel):
    query:     str
    results:   list[dict]
    answer:    str
    timestamp: str


class Message(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query:   str
    user_id: str = "anonymous"
    history: List[Message] = []


class JobSearchRequest(BaseModel):
    job_title: str
    top_n:     int = 10


class SkillMatchRequest(BaseModel):
    current_major:  str
    current_skills: str
    target_job:     str
