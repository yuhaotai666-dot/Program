"""
core/embedder.py ── BERT 编码（唯一来源，其他文件 import 这里，不再重复定义）
"""
import torch
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


def mean_pooling(model_output, attention_mask) -> torch.Tensor:
    """对所有 token 向量做 attention 加权平均，得到句子向量"""
    token_embeddings    = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_sentences(sentences: list[str],
                     tokenizer: BertTokenizer,
                     model: BertModel) -> np.ndarray:
    """
    批量编码（build_index.py 建索引时使用）。
    返回 L2 归一化向量矩阵，shape = (len(sentences), 768)。
    """
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1).numpy()


def encode_query(text: str,
                 tokenizer: BertTokenizer,
                 model: BertModel) -> np.ndarray:
    """
    单条查询编码（API 检索时使用）。
    返回 float32 行向量，shape = (1, 768)。
    """
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)
    embedding = mean_pooling(output, encoded["attention_mask"])
    return F.normalize(embedding, p=2, dim=1).numpy().astype("float32")
