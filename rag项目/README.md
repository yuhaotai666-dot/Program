# Job Analysis & Skill Match — RAG 职位推荐系统

> 基于 BERT + FAISS + BM25 混合检索，结合 GPT-4o-mini 生成的智能职位分析与技能匹配平台

---

## 功能概览

| 功能 | 说明 |
|:---|:---|
| **职位搜索** | 混合 RAG 检索（语义 + 关键词），返回最相关职位列表及市场统计图表 |
| **技能匹配** | 基于真实招聘数据 grounding，分析技能差距，输出 0-100 匹配分 |
| **学习路线** | GPT-4o-mini 生成三阶段学习计划，附 B 站视频搜索链接 |
| **简历解析** | 上传 PDF / DOCX / TXT 简历，自动提取技能并推荐职位 |

---

## 界面截图

### Tab 1 — 职位分析 & 市场看板

![Job Analysis](docs/screenshot_job_analysis.png)

搜索职位后，左侧展示 Top-N 职位列表（公司、地点、薪资、技能标签），右侧展示市场总览、技能频率柱状图与地点分布饼图。

### Tab 2 — 技能匹配 & 学习路线

![Skill Match & Learning Path](docs/screenshot_skill_match.png)

输入当前专业、现有技能和目标职位，系统从 50k+ 真实岗位中提取高频技能进行 grounding，由 LLM 生成匹配分析与阶段性学习路线图。

---

## 技术架构

```
用户请求
    │
    ▼
Gradio Frontend (port 7860)
    │  HTTP
    ▼
FastAPI Backend (port 8000)
    │
    ├── Embedder    BERT (本地 bert_pretrain) → 768 维语义向量
    ├── FAISS       内积向量索引，5000 条语义块
    ├── BM25        全量文本关键词索引（50k 条职位）
    ├── Retriever   混合召回（FAISS 70% + BM25 30%）→ Rerank
    └── Generator   GPT-4o-mini，RAG grounding 防幻觉
```

### 检索流程（Hybrid RAG）

1. **Semantic Chunking** — 将职位 CSV 按 BERT token 预算（≤512）切分为结构化语义块（标题 / 技能 / 描述）
2. **FAISS 向量召回** — query 编码后进行内积检索，返回 Top-60 候选
3. **BM25 关键词召回** — 中英文混合分词，补充精确匹配结果
4. **加权融合** — 双路分数各自归一化后加权求和（FAISS 0.7 + BM25 0.3）
5. **Rerank** — 按关键词优先级（精确 > 包含 > 词级 > 语义）重排序
6. **LLM 生成** — 将召回原文作为 context，约束 GPT-4o-mini 基于真实数据回答

---

## 项目结构

```
rag项目/
├── config.py            # 全局配置（路径 / 模型参数 / API Key）
├── state.py             # 全局资源容器（FAISS index / BM25 / LLM client）
├── build_index.py       # 离线建索引脚本
├── web_backend.py       # FastAPI 应用入口（lifespan 资源加载）
├── web_frontend.py      # Gradio 前端（Job Analysis + Skill Match 两个 Tab）
│
├── core/
│   ├── embedder.py      # BERT query 编码
│   ├── chunker.py       # Semantic Chunking（按 token 预算切分）
│   ├── retriever.py     # 混合召回 / Rerank / 技能提取
│   └── generator.py     # LLM 生成（单轮 / 多轮 / 简历解析）
│
├── api/
│   ├── models.py        # Pydantic 请求/响应模型
│   └── routes.py        # 所有 FastAPI 路由
│
├── bert_pretrain/       # 本地 BERT 模型权重（sentence-bert）
│
├── data/
│   ├── job_descriptions.csv   # 职位数据集（50k+ 条）
│   ├── faiss.index            # 预构建向量索引
│   └── texts.json             # 语义块文本（与 FAISS 对齐）
│
└── docs/
    ├── screenshot_job_analysis.png
    └── screenshot_skill_match.png
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn gradio openai faiss-cpu \
            transformers sentence-transformers rank-bm25 \
            pandas numpy PyPDF2 python-docx
```

### 2. 配置

编辑 `config.py`，设置 OpenAI API Key 和代理地址（或通过环境变量）：

```bash
export OPENAI_API_KEY1="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 或第三方代理
```

### 3. 构建向量索引（首次运行）

```bash
python build_index.py
```

生成 `data/faiss.index` 和 `data/texts.json`，默认取前 5000 条数据建索引。

### 4. 启动后端

```bash
python web_backend.py
# FastAPI 运行在 http://localhost:8000
```

### 5. 启动前端

```bash
python web_frontend.py
# Gradio 运行在 http://localhost:7860
```

---

## API 接口

| Method | Path | 说明 |
|:---|:---|:---|
| `GET` | `/api/health` | 后端健康检查 |
| `GET` | `/api/random-jobs` | 随机返回 N 条职位 |
| `POST` | `/api/search-jobs` | 职位搜索（混合 RAG + Rerank） |
| `POST` | `/api/skill-match` | 技能匹配 & 学习路线生成 |
| `POST` | `/api/search` | 单轮 RAG 对话检索 |
| `POST` | `/api/chat` | 多轮 RAG 对话 |
| `POST` | `/api/upload-resume` | 简历上传解析 & 职位推荐 |
| `GET` | `/api/history/{user_id}` | 查询历史记录 |

---

## 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `SAMPLE_SIZE` | 5000 | 建索引时取的职位行数 |
| `JOBS_SAMPLE` | 50000 | 后端运行时加载的职位行数 |
| `TOP_K` | 5 | 最终返回结果数 |
| `FAISS_WEIGHT` | 0.7 | 语义检索权重 |
| `BM25_WEIGHT` | 0.3 | 关键词检索权重 |
| `RERANK_CANDIDATE_K` | 60 | Rerank 候选数量 |
| `TOKEN_BUDGET` | 500 | 每个语义块的 token 上限 |
| `LLM_MODEL` | gpt-4o-mini | LLM 模型 |

---

## 技术栈

- **Embedding**: `bert-base` / sentence-transformers（本地推理）
- **向量检索**: FAISS（内积索引）
- **关键词检索**: BM25Okapi（rank-bm25）
- **LLM**: GPT-4o-mini（OpenAI API）
- **后端框架**: FastAPI + uvicorn
- **前端框架**: Gradio
- **数据处理**: Pandas / NumPy
