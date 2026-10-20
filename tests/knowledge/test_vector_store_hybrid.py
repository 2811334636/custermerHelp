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

    # spec §3.3：元数据字段必须在 schema 里，才能用 filter= 下推过滤。
    assert "category" in field_names
    assert "section_path" in field_names


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

    # 随机 dense vector + RRF 融合不保证固定排序，只验证返回有效结果
    assert len(ids) > 0 and all(id in ["test1", "test2"] for id in ids)


def test_bm25_search_applies_category_filter_on_milvus(test_collection_bm25):
    """真实 Milvus：filter_expr 在检索层排除其他分类，不是回表后过滤。"""
    store = VectorStore(support_bm25=True, collection=test_collection_bm25)
    store.ensure_collection()

    vector = np.zeros(1024).tolist()
    store.upsert([
        {
            "chunk_id": "cat-a",
            "content": "退款政策说明",
            "vector": vector,
            "category": "A",
            "section_path": "A > 退款",
        },
        {
            "chunk_id": "cat-b",
            "content": "退款政策说明",
            "vector": vector,
            "category": "B",
            "section_path": "B > 退款",
        },
    ])
    time.sleep(0.5)

    all_ids = store.bm25_search("退款", top_k=10)
    filtered = store.bm25_search(
        "退款", top_k=10, filter_expr='category == "A"'
    )

    assert {"cat-a", "cat-b"} <= set(all_ids), "不过滤时应两路都能命中"
    assert "cat-a" in filtered
    assert "cat-b" not in filtered, "filter 必须排除 B 分类"


def test_upsert_backward_compat_with_bm25(test_collection_bm25):
    """验证旧 upsert(ids, vectors) 签名在 support_bm25=True 时仍可用。"""
    store = VectorStore(support_bm25=True, collection=test_collection_bm25)
    store.ensure_collection()

    ids = ["old1", "old2"]
    vectors = [np.random.rand(1024).tolist() for _ in range(2)]

    # 旧签名不传 content，依赖 nullable=True
    store.upsert(ids, vectors)

    # 等待数据可见
    time.sleep(0.1)

    assert store.count() == 2

