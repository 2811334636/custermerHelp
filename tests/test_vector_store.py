from unittest.mock import MagicMock

import pytest

from app.knowledge.vector_store import VectorStore, VectorStoreUnavailable


def make_store(client):
    return VectorStore(client=client, collection="knowledge", dim=1024)


def test_search_returns_ids_in_score_order():
    client = MagicMock()
    client.has_collection.return_value = True
    client.search.return_value = [[{"id": "a"}, {"id": "b"}]]
    ids = make_store(client).search([0.0] * 1024, top_k=2)
    assert ids == ["a", "b"]


def test_empty_collection_returns_empty_list():
    client = MagicMock()
    client.has_collection.return_value = True
    client.search.return_value = [[]]
    assert make_store(client).search([0.0] * 1024, top_k=3) == []


def test_milvus_failure_is_wrapped():
    client = MagicMock()
    client.has_collection.return_value = True
    client.search.side_effect = RuntimeError("connection refused")
    with pytest.raises(VectorStoreUnavailable):
        make_store(client).search([0.0] * 1024, top_k=3)


def test_search_passes_top_k_and_explicit_retrieval_params():
    client = MagicMock()
    client.search.return_value = []
    assert make_store(client).search([0.0] * 1024, top_k=7) == []
    client.search.assert_called_once_with(
        collection_name="knowledge",
        data=[[0.0] * 1024],
        anns_field="vector",
        search_params={"metric_type": "COSINE"},
        limit=7,
        output_fields=["id"],
    )
    # 裸 metric_type= 会被 pymilvus 3.0.2 接受但静默忽略，不能出现。
    assert "metric_type" not in client.search.call_args.kwargs


def test_ensure_collection_creates_schema_and_cosine_index():
    """集合不存在时：VARCHAR 主键 + dim 维向量 + AUTOINDEX/COSINE。"""
    from pymilvus import DataType

    client = MagicMock()
    client.has_collection.return_value = False
    schema = MagicMock()
    client.create_schema.return_value = schema
    index_params = MagicMock()
    client.prepare_index_params.return_value = index_params

    make_store(client).ensure_collection()

    client.has_collection.assert_called_once_with("knowledge")
    client.create_schema.assert_called_once_with(auto_id=False)
    schema.add_field.assert_any_call(
        "id", DataType.VARCHAR, is_primary=True, max_length=512
    )
    schema.add_field.assert_any_call("vector", DataType.FLOAT_VECTOR, dim=1024)
    # Task 0 实测：pymilvus 3.0.2 顶层没有 IndexParams，必须用公开工厂。
    client.prepare_index_params.assert_called_once_with()
    index_params.add_index.assert_called_once_with(
        "vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    client.create_collection.assert_called_once_with(
        "knowledge", schema=schema, index_params=index_params
    )


def test_ensure_collection_is_idempotent_when_collection_exists():
    client = MagicMock()
    client.has_collection.return_value = True

    make_store(client).ensure_collection()

    client.create_collection.assert_not_called()


def test_upsert_sends_rows():
    client = MagicMock()
    make_store(client).upsert(["a", "b"], [[0.0] * 1024, [1.0] * 1024])
    client.upsert.assert_called_once_with(
        "knowledge",
        [{"id": "a", "vector": [0.0] * 1024}, {"id": "b", "vector": [1.0] * 1024}],
    )


def test_upsert_rejects_mismatched_lengths():
    """zip 会静默截断；ids 与 vectors 数量不符必须响亮失败。"""
    client = MagicMock()
    with pytest.raises(ValueError) as exc:
        make_store(client).upsert(["a", "b"], [[0.0] * 1024])
    assert "2" in str(exc.value) and "1" in str(exc.value)
    client.upsert.assert_not_called()


def test_empty_upsert_is_noop():
    client = MagicMock()
    make_store(client).upsert([], [])
    client.upsert.assert_not_called()


def test_upsert_failure_is_wrapped():
    client = MagicMock()
    client.upsert.side_effect = RuntimeError("connection refused")
    with pytest.raises(VectorStoreUnavailable):
        make_store(client).upsert(["a"], [[0.0] * 1024])


def test_bm25_search_queries_sparse_field():
    """纯 BM25 走 sparse_vector + BM25 metric，不能退回 dense。"""
    client = MagicMock()
    client.search.return_value = [[{"id": "a"}, {"id": "b"}]]
    ids = make_store(client).bm25_search("运费", top_k=2)

    assert ids == ["a", "b"]
    client.search.assert_called_once_with(
        collection_name="knowledge",
        data=["运费"],
        anns_field="sparse_vector",
        search_params={"metric_type": "BM25"},
        limit=2,
        output_fields=["id"],
    )


def test_bm25_search_empty_result_is_empty_list():
    client = MagicMock()
    client.search.return_value = [[]]
    assert make_store(client).bm25_search("运费", top_k=3) == []


def test_bm25_search_failure_is_wrapped():
    client = MagicMock()
    client.search.side_effect = RuntimeError("connection refused")
    with pytest.raises(VectorStoreUnavailable):
        make_store(client).bm25_search("运费", top_k=3)


def test_count_uses_count_star():
    client = MagicMock()
    client.query.return_value = [{"count(*)": 42}]
    assert make_store(client).count() == 42
    client.query.assert_called_once_with(
        "knowledge", filter="", output_fields=["count(*)"]
    )


def test_count_empty_result_is_zero():
    client = MagicMock()
    client.query.return_value = []
    assert make_store(client).count() == 0


def test_count_failure_is_wrapped():
    client = MagicMock()
    client.query.side_effect = RuntimeError("connection refused")
    with pytest.raises(VectorStoreUnavailable):
        make_store(client).count()
