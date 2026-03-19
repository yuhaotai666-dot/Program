"""
state.py ── 全局运行时资源（单例），由 web_backend.py lifespan 初始化，
            api/routes.py 读取使用。
"""
import threading
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import OpenAI
from transformers import BertTokenizer, BertModel

# BERT 推理锁：防止 macOS 上 PyTorch 在多线程中崩溃（SIGSEGV）
bert_lock = threading.Lock()


class Resources:
    tokenizer: BertTokenizer = None
    model:     BertModel     = None
    index:     faiss.Index   = None
    texts:     list          = None
    bm25:      BM25Okapi     = None
    llm:       OpenAI        = None
    df_jobs:   pd.DataFrame  = None


res: Resources = Resources()

# 内存历史记录（生产环境可换 MySQL）
history_store: dict = {}
