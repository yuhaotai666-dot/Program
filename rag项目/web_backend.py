"""
web_backend.py ── FastAPI 应用入口
职责：应用初始化 + 资源生命周期管理（lifespan）
业务逻辑 → api/routes.py
核心算法 → core/
配置      → config.py
全局资源  → state.py
"""
import os
import json
from contextlib import asynccontextmanager

# 必须在 import torch 之前设置，禁止 PyTorch OpenMP 多线程
# 防止 macOS 上 uvicorn 线程池 + PyTorch 导致 SIGSEGV
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"   # 允许 PyTorch + FAISS 同时加载各自的 OpenMP

import faiss
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    MODEL_PATH, INDEX_PATH, TEXTS_PATH,
    JOBS_CSV_PATH, JOBS_SAMPLE,
    OPENAI_API_KEY, OPENAI_BASE_URL,
)
from state import res
from core.retriever import tokenize_for_bm25
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载所有资源，关闭时释放"""
    print("正在加载资源...")

    # 禁止 PyTorch 和 FAISS 各自的多线程，防止 macOS SIGSEGV
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)
    res.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    res.model     = BertModel.from_pretrained(MODEL_PATH)
    res.model.eval()

    # FAISS 向量索引 + 语义块文本
    res.index = faiss.read_index(INDEX_PATH)
    res.texts = json.load(open(TEXTS_PATH, "r", encoding="utf8"))

    # BM25 索引（基于已有 texts，秒级构建）
    print("构建 BM25 索引...")
    res.bm25 = BM25Okapi([tokenize_for_bm25(t) for t in res.texts])

    # LLM 客户端
    res.llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

    # 职位 CSV 数据
    print("加载职位数据...")
    cols = ['Job Title', 'location', 'Country', 'Salary Range',
            'skills', 'Work Type', 'Experience', 'Company', 'Job Description']
    res.df_jobs = pd.read_csv(JOBS_CSV_PATH, usecols=cols, nrows=JOBS_SAMPLE)
    res.df_jobs.fillna('', inplace=True)

    print(f"资源加载完成 ✓  向量库 {res.index.ntotal} 条 | 职位数据 {len(res.df_jobs):,} 条")
    yield
    print("服务关闭")


app = FastAPI(title="职位 RAG 推荐系统", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_backend:app", host="0.0.0.0", port=8000, reload=False)
