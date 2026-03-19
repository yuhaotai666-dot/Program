"""
build_index.py ── 离线脚本：构建 FAISS 向量索引
运行：python build_index.py
依赖 core/embedder.py 和 core/chunker.py，不再重复定义 BERT 函数。
"""
import json
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from config import (
    MODEL_PATH, DATA_PATH, INDEX_PATH, TEXTS_PATH,
    BATCH_SIZE, SAMPLE_SIZE,
)
from core.embedder import encode_sentences   # ← 不再重复定义
from core.chunker  import make_semantic_chunk  # ← 不再重复定义


def build_index(texts: list[str], tokenizer: BertTokenizer, model: BertModel) -> faiss.Index:
    """批量编码所有语义块，构建 FAISS IndexFlatIP 索引"""
    dim   = 768
    index = faiss.IndexFlatIP(dim)

    all_vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="语义编码"):
        batch = texts[i: i + BATCH_SIZE]
        vecs  = encode_sentences(batch, tokenizer, model)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs).astype("float32")
    index.add(all_vecs)
    print(f"索引构建完成，共 {index.ntotal} 条向量")
    return index


if __name__ == "__main__":
    # 1. 读取 CSV
    print(f"读取数据（前 {SAMPLE_SIZE} 条）...")
    cols = ["Job Title", "Work Type", "Experience", "skills", "Job Description"]
    df   = pd.read_csv(DATA_PATH, usecols=cols, nrows=SAMPLE_SIZE)
    df.fillna("", inplace=True)
    print(f"共读取 {len(df)} 条职位数据")

    # 2. 加载 BERT
    print("加载 BERT 模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model     = BertModel.from_pretrained(MODEL_PATH)
    model.eval()

    # 3. Semantic Chunking：每个职位 → 一个语义块
    print("语义切分中...")
    texts      = []
    token_lens = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="切分职位"):
        chunk = make_semantic_chunk(row, tokenizer)
        texts.append(chunk)
        token_lens.append(len(tokenizer.encode(chunk, add_special_tokens=True)))

    tl = np.array(token_lens)
    print(f"\n[语义切分统计]")
    print(f"  平均 token 数：{tl.mean():.1f}")
    print(f"  最大 token 数：{tl.max()}")
    print(f"  超过512的块数：{(tl > 512).sum()}（应为0）")
    print(f"  示例（第0条）:\n  {texts[0][:200]}...\n")

    # 4. 构建索引
    index = build_index(texts, tokenizer, model)

    # 5. 保存
    faiss.write_index(index, INDEX_PATH)
    json.dump(texts, open(TEXTS_PATH, "w", encoding="utf8"), ensure_ascii=False)
    print(f"索引已保存至 {INDEX_PATH}")
    print(f"文本已保存至 {TEXTS_PATH}（共 {len(texts)} 条语义块）")
