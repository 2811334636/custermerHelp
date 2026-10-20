from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config import get_settings
from app.knowledge.errors import KnowledgeBackendUnavailable

# chunk_id 目前最长的是 "<文件名>:<序号>"，512 足够且留有充分余量。
_ID_MAX_LENGTH = 512


class VectorStoreUnavailable(KnowledgeBackendUnavailable):
    """Milvus 读写失败（连接、超时、服务端错误等）。"""


class VectorStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        collection: str | None = None,
        dim: int | None = None,
        support_bm25: bool = False,
    ):
        s = get_settings()
        if client is not None:
            self._client = client
        else:
            # MilvusClient 构造时会等待 channel ready（默认约 10s），连不上
            # 抛的是 pymilvus/gRPC 原生异常。必须在构造处就转成我们自己的
            # 类型，否则 query_faq 的降级分支永远走不到（final review C1）。
            try:
                self._client = MilvusClient(uri=s.milvus_uri)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreUnavailable(str(exc)) from exc
        self._collection = collection if collection is not None else s.milvus_collection
        self._dim = dim if dim is not None else s.embedding_dim
        self._support_bm25 = support_bm25

    @property
    def collection_name(self) -> str:
        """返回 collection 名称。"""
        return self._collection

    @property
    def support_bm25(self) -> bool:
        """当前 store 是否为 BM25 schema（builder 据此决定 upsert 形态）。"""
        return self._support_bm25

    @property
    def uri(self) -> str:
        """返回 Milvus URI。"""
        return get_settings().milvus_uri

    def _field_names(self) -> set[str]:
        """已存在 collection 的字段名集合（用于 schema 陈旧检查）。"""
        try:
            desc = self._client.describe_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc
        fields = desc.get("fields", []) if isinstance(desc, dict) else []
        return {field.get("name") for field in fields}

    def ensure_collection(self) -> None:
        """集合不存在时创建：VARCHAR 主键 + 指定维度的向量列 + COSINE 索引。

        已有的 collection 若与当前模式不匹配（要 BM25 但集合是旧的
        dense-only schema），直接报错并指出重建命令——静默复用只会让
        BM25/hybrid 永远召回为空，属于最难排查的一类故障。
        """
        if self._client.has_collection(self._collection):
            if self._support_bm25:
                # ch04 过滤依赖 category；旧集合没有它就无法在 Milvus 侧
                # 按分类过滤，必须重建而不是静默降级成 Python 侧过滤。
                missing = {"sparse_vector", "category"} - self._field_names()
                if missing:
                    raise VectorStoreUnavailable(
                        f"collection「{self._collection}」是旧的 schema，"
                        f"缺少字段：{', '.join(sorted(missing))}；请先运行 "
                        "`uv run python -m jobs.build_knowledge --rebuild` "
                        "重建集合。"
                    )
            return
        schema = self._client.create_schema(auto_id=False)
        schema.add_field(
            "id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)

        # 元数据过滤（spec §3.3）：category / section_path 存在 Milvus 的
        # scalar 字段里，检索时用 filter= 在服务端过滤，而不是回表后再筛。
        # nullable=True 兼容旧 upsert(ids, vectors) 路径（不传元数据）。
        schema.add_field(
            "category", DataType.VARCHAR, max_length=255, nullable=True
        )
        schema.add_field(
            "section_path", DataType.VARCHAR, max_length=512, nullable=True
        )

        # BM25 支持：sparse_vector 字段 + content 字段 + BM25 Function
        if self._support_bm25:
            # content 字段设为非必需，允许旧代码路径不传 content
            schema.add_field(
                "content",
                DataType.VARCHAR,
                max_length=2000,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
                nullable=True,
            )
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            bm25_fn = Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(bm25_fn)

        # pymilvus 3.0.2 顶层没有 IndexParams，公开工厂方法才是可用入口。
        index_params = self._client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")

        if self._support_bm25:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )

        self._client.create_collection(
            self._collection, schema=schema, index_params=index_params
        )

    def drop_collection(self) -> None:
        """删除整个 collection（仅 --rebuild 使用）。"""
        try:
            if self._client.has_collection(self._collection):
                self._client.drop_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc

    def upsert(
        self, ids_or_data: list[str] | list[dict], vectors: list[list[float]] | None = None
    ) -> None:
        """按主键幂等写入；重复 upsert 同一 id 不会产生重复行。

        支持两种调用方式：
        1. upsert(data) - 字典列表，包含 chunk_id/vector，以及可选的
           content（BM25 模式必需）、category / section_path。builder 与
           BM25 路径都走这一形态，元数据因此在两种模式下都会写入 Milvus。
        2. upsert(ids, vectors) - 旧接口，仅保留给外部调用方；元数据字段
           留空（nullable=True）。
        """
        # 字典形态：以空列表进入时也按字典路径处理（幂等 no-op）。
        if isinstance(ids_or_data, list) and (
            not ids_or_data or isinstance(ids_or_data[0], dict)
        ):
            if ids_or_data and vectors is not None:
                raise ValueError("字典形态下不应传入 vectors 参数")
            if not ids_or_data:
                return
            rows = []
            for d in ids_or_data:
                row: dict[str, Any] = {
                    "id": d["chunk_id"],
                    "vector": d["vector"],
                    "category": d.get("category"),
                    "section_path": d.get("section_path"),
                }
                # 非 BM25 collection 没有 content 字段，不能传这个 key。
                if self._support_bm25:
                    row["content"] = d.get("content")
                rows.append(row)
        else:
            # 原有模式：ids + vectors
            ids = ids_or_data
            if vectors is None:
                raise ValueError("非 BM25 模式下必须传入 vectors 参数")
            # zip 会按较短序列静默截断，导致部分 id/向量被悄悄丢弃。
            if len(ids) != len(vectors):
                raise ValueError(
                    f"ids 与 vectors 数量不一致：{len(ids)} != {len(vectors)}"
                )
            if not ids:
                return
            # 如果启用了 BM25，旧接口路径下 content 为空（nullable=True）
            if self._support_bm25:
                rows = [{"id": i, "vector": v, "content": None} for i, v in zip(ids, vectors)]
            else:
                rows = [{"id": i, "vector": v} for i, v in zip(ids, vectors)]

        try:
            self._client.upsert(self._collection, rows)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc

    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        filter_expr: str | None = None,
    ) -> list[str]:
        """返回相似度从高到低的 chunk id，最多 top_k 条。

        ``filter_expr`` 交由 Milvus 在检索层应用（spec §3.3），不是回表后过滤。
        """
        try:
            res = self._client.search(
                collection_name=self._collection,
                data=[vector],
                anns_field="vector",
                # metric_type 必须放进 search_params：pymilvus 3.0.2 接受裸
                # metric_type= 但会静默忽略，从而退回索引/服务端默认。
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                output_fields=["id"],
                filter=filter_expr or "",
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc
        if not res:
            return []
        return [hit["id"] for hit in res[0]]

    def bm25_search(
        self,
        query_text: str,
        top_k: int,
        *,
        filter_expr: str | None = None,
    ) -> list[str]:
        """纯 BM25 全文检索：返回匹配度从高到低的 chunk id。

        BM25 是 hybrid_search 的稀疏分支；单独抽出来供 `bm25` 策略与
        四策略评估使用（评估必须能单独观测 BM25 这一路）。
        ``filter_expr`` 同样在 Milvus 侧生效。
        """
        try:
            res = self._client.search(
                collection_name=self._collection,
                data=[query_text],
                anns_field="sparse_vector",
                search_params={"metric_type": "BM25"},
                limit=top_k,
                output_fields=["id"],
                filter=filter_expr or "",
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

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        dense_limit: int,
        bm25_limit: int,
        final_limit: int,
        *,
        filter_expr: str | None = None,
    ) -> list[str]:
        """混合检索：dense + BM25 通过 RRF 融合。

        ``filter_expr`` 必须同时挂在两条召回腿上，否则过滤只作用于其中
        一路，另一路仍会把被排除的 chunk 带进融合结果。
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        # pymilvus 3.0.2 支持 filter=；不能同时传 expr 与 filter。
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=dense_limit,
            filter=filter_expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=bm25_limit,
            filter=filter_expr,
        )

        try:
            results = self._client.hybrid_search(
                self._collection,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=final_limit,
                output_fields=["id"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc

        if not results or not results[0]:
            return []
        return [hit["entity"]["id"] for hit in results[0]]
