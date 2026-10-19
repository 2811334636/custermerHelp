"""VectorStore BM25 混合检索测试。"""

import time

import numpy as np
from pymilvus import DataType, MilvusClient

from app.knowledge.vector_store import VectorStore


def test_vector_store_create_with_bm25(test_collection_bm25):
    """VectorStore 创建包含 sparse_vector + BM25 Function 的 schema。"""
    store = VectorStore(support_bm25=True, collection=test_collection_bm25)
    store.ensure_collection()

    coll = store.collection_name
    client = MilvusClient(uri=store.uri)

    # 验证 schema 包含 sparse_vector 字段
    desc = client.describe_collection(coll)
    field_names = [f["name"] for f in desc["fields"]]
    assert "sparse_vector" in field_names

    # 验证 sparse_vector 类型
    sparse_field = next(f for f in desc["fields"] if f["name"] == "sparse_vector")
    assert sparse_field["type"] == DataType.SPARSE_FLOAT_VECTOR

    # 验证 Function 存在
    functions = desc.get("functions", [])
    assert len(functions) == 1
    assert functions[0]["name"] == "bm25_fn"
    assert functions[0]["input_field_names"] == ["content"]
    assert functions[0]["output_field_names"] == ["sparse_vector"]


def test_hybrid_search(test_collection_bm25):
    """hybrid_search 返回 RRF 融合后的结果。"""
    store = VectorStore(support_bm25=True, collection=test_collection_bm25)
    store.ensure_collection()

    # 插入测试数据
    data = [
        {
            "chunk_id": "test1",
            "content": "运费说明：满99元包邮",
            "vector": np.random.rand(1024).tolist(),
        },
        {
            "chunk_id": "test2",
            "content": "退货政策：7天无理由",
            "vector": np.random.rand(1024).tolist(),
        },
    ]
    store.upsert(data)

    # 等待索引构建
    time.sleep(0.5)

    # hybrid_search
    query_vec = np.random.rand(1024).tolist()
    ids = store.hybrid_search(
        query_vector=query_vec,
        query_text="运费",
        dense_limit=2,
        bm25_limit=2,
        final_limit=2,
    )

    assert len(ids) > 0
    assert "test1" in ids  # BM25 应优先召回"运费"相关
