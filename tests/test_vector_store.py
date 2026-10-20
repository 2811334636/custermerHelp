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
        filter="",
    )
    # 裸 metric_type= 会被 pymilvus 3.0.2 接受但静默忽略，不能出现。
    assert "metric_type" not in client.search.call_args.kwargs


def test_search_passes_filter_expr_to_milvus():
    """过滤必须下推到 Milvus（spec §3.3），不是 Python 侧过滤。"""
    client = MagicMock()
    client.search.return_value = []
    make_store(client).search(
        [0.0] * 1024, top_k=3, filter_expr='category == "售后政策"'
    )
    assert client.search.call_args.kwargs["filter"] == 'category == "售后政策"'


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
    # spec §3.3：元数据字段必须进 Milvus schema，供服务端 filter 使用。
    schema.add_field.assert_any_call(
        "category", DataType.VARCHAR, max_length=255, nullable=True
    )
    schema.add_field.assert_any_call(
        "section_path", DataType.VARCHAR, max_length=512, nullable=True
    )
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


def test_ensure_collection_rejects_dense_only_collection_when_bm25_requested():
    """旧 dense-only 集合不能被静默复用：提示操作员走 --rebuild。"""
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": "id"}, {"name": "vector"}]
    }
    store = VectorStore(
        client=client, collection="knowledge", dim=1024, support_bm25=True
    )

    with pytest.raises(VectorStoreUnavailable, match="--rebuild"):
        store.ensure_collection()


def test_ensure_collection_accepts_existing_bm25_collection():
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id"},
            {"name": "vector"},
            {"name": "sparse_vector"},
            {"name": "category"},
        ]
    }
    store = VectorStore(
        client=client, collection="knowledge", dim=1024, support_bm25=True
    )

    store.ensure_collection()  # 不抛异常


def test_ensure_collection_rejects_bm25_collection_without_category():
    """旧 BM25 集合缺 category：过滤无法下推，必须 --rebuild。"""
    client = MagicMock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": [{"name": "id"}, {"name": "vector"}, {"name": "sparse_vector"}]
    }
    store = VectorStore(
        client=client, collection="knowledge", dim=1024, support_bm25=True
    )

    with pytest.raises(VectorStoreUnavailable, match="category") as exc:
        store.ensure_collection()
    assert "--rebuild" in str(exc.value)


def test_drop_collection_removes_existing_collection():
    client = MagicMock()
    client.has_collection.return_value = True

    make_store(client).drop_collection()

    client.drop_collection.assert_called_once_with("knowledge")


def test_upsert_sends_rows():
    client = MagicMock()
    make_store(client).upsert(["a", "b"], [[0.0] * 1024, [1.0] * 1024])
    client.upsert.assert_called_once_with(
        "knowledge",
        [{"id": "a", "vector": [0.0] * 1024}, {"id": "b", "vector": [1.0] * 1024}],
    )


def test_upsert_dict_mode_includes_metadata_without_bm25():
    """非 BM25 模式也走字典行，category/section_path 必须写入。"""
    client = MagicMock()
    make_store(client).upsert([
        {
            "chunk_id": "a",
            "vector": [0.0] * 1024,
            "category": "售后政策",
            "section_path": "售后政策 > 退货",
        }
    ])
    client.upsert.assert_called_once_with(
        "knowledge",
        [
            {
                "id": "a",
                "vector": [0.0] * 1024,
                "category": "售后政策",
                "section_path": "售后政策 > 退货",
            }
        ],
    )


def test_upsert_dict_mode_includes_content_and_metadata_for_bm25():
    client = MagicMock()
    store = VectorStore(
        client=client, collection="knowledge", dim=1024, support_bm25=True
    )
    store.upsert([
        {
            "chunk_id": "a",
            "content": "退货文本",
            "vector": [0.0] * 1024,
            "category": "售后政策",
            "section_path": "售后政策 > 退货",
        }
    ])
    client.upsert.assert_called_once_with(
        "knowledge",
        [
            {
                "id": "a",
                "vector": [0.0] * 1024,
                "content": "退货文本",
                "category": "售后政策",
                "section_path": "售后政策 > 退货",
            }
        ],
    )


def test_legacy_upsert_keeps_working_and_leaves_metadata_null():
    """旧签名仍可用；没有元数据的行由 nullable 字段留空。"""
    client = MagicMock()
    store = VectorStore(
        client=client, collection="knowledge", dim=1024, support_bm25=True
    )
    store.upsert(["a"], [[0.0] * 1024])
    client.upsert.assert_called_once_with(
        "knowledge", [{"id": "a", "vector": [0.0] * 1024, "content": None}]
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
        filter="",
    )


def test_bm25_search_passes_filter_expr_to_milvus():
    client = MagicMock()
    client.search.return_value = []
    make_store(client).bm25_search(
        "运费", top_k=2, filter_expr='category == "A"'
    )
    assert client.search.call_args.kwargs["filter"] == 'category == "A"'


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
