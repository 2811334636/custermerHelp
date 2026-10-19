from typing import Any

from pymilvus import DataType, MilvusClient

from app.config import get_settings

# chunk_id 目前最长的是 "<文件名>:<序号>"，512 足够且留有充分余量。
_ID_MAX_LENGTH = 512


class VectorStoreUnavailable(RuntimeError):
    """Milvus 读写失败（连接、超时、服务端错误等）。"""


class VectorStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        collection: str | None = None,
        dim: int | None = None,
    ):
        s = get_settings()
        self._client = client if client is not None else MilvusClient(uri=s.milvus_uri)
        self._collection = collection if collection is not None else s.milvus_collection
        self._dim = dim if dim is not None else s.embedding_dim

    def ensure_collection(self) -> None:
        """集合不存在时创建：VARCHAR 主键 + 指定维度的向量列 + COSINE 索引。"""
        if self._client.has_collection(self._collection):
            return
        schema = self._client.create_schema(auto_id=False)
        schema.add_field(
            "id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        # pymilvus 3.0.2 顶层没有 IndexParams，公开工厂方法才是可用入口。
        index_params = self._client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            self._collection, schema=schema, index_params=index_params
        )

    def upsert(self, ids: list[str], vectors: list[list[float]]) -> None:
        """按主键幂等写入；重复 upsert 同一 id 不会产生重复行。"""
        rows = [{"id": i, "vector": v} for i, v in zip(ids, vectors)]
        try:
            self._client.upsert(self._collection, rows)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc

    def search(self, vector: list[float], top_k: int) -> list[str]:
        """返回相似度从高到低的 chunk id，最多 top_k 条。"""
        try:
            res = self._client.search(
                self._collection, data=[vector], limit=top_k, output_fields=["id"]
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc
        if not res:
            return []
        return [hit["id"] for hit in res[0]]

    def count(self) -> int:
        try:
            rows = self._client.query(
                self._collection, filter="", output_fields=["count(*)"]
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc
        return int(rows[0]["count(*)"]) if rows else 0
