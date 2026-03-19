"""
config.py ── 全局配置，所有文件从这里 import，不再各自硬编码
"""
import os

# ── 路径 ────────────────────────────────────────────────
MODEL_PATH        = "./bert_pretrain"
DATA_PATH         = "./data/job_descriptions.csv"
INDEX_PATH        = "./data/faiss.index"
TEXTS_PATH        = "./data/texts.json"
JOBS_CSV_PATH     = "./data/job_descriptions.csv"

# ── 数据规模 ─────────────────────────────────────────────
SAMPLE_SIZE       = 5000    # build_index.py 建索引时取的行数
JOBS_SAMPLE       = 50000   # 后端运行时加载的行数
FAISS_INDEX_ROWS  = 5000    # 与 SAMPLE_SIZE 一致，用于 row_idx 对齐

# ── 检索参数 ─────────────────────────────────────────────
TOP_K             = 5
THRESHOLD         = 0.3
FAISS_WEIGHT      = 0.7
BM25_WEIGHT       = 0.3
RERANK_CANDIDATE_K = 60

# ── Semantic Chunking ────────────────────────────────────
BATCH_SIZE        = 8
TOKEN_BUDGET      = 500     # 每个 chunk 的内容 token 上限（≤512 给 BERT）

# ── LLM ─────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY1", "sk-0Sg251l8G6qsOblN9jQi4SynVjjDHO4jJcYYBcHcDUVYN3ne")
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
LLM_MODEL         = "gpt-4o-mini"
