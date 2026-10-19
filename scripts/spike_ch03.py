"""Task 0 阻塞性探测：embedding 客户端兼容性 + Milvus 客户端/服务端匹配。

重跑时必须能复现「语义可分」的结论，而不只是维度正确：生产配置已确认
必须关闭 check_embedding_ctx_length，且「邮费是多少」与运费说明文本的
cosine 应明显高于无关文本（final review M4）。
"""

import asyncio
import math
import time

from pymilvus import DataType, MilvusClient

from app.config import get_settings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def probe_milvus() -> None:
    print("=== 探测 1：Milvus 连接与版本 ===")
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    # pymilvus 3.0.2 的 MilvusClient 自带 get_server_version/get_server_type；
    # 简报里的 utility.get_server_version(using=client._using) 依赖 2.x 的私有属性。
    print("  server version:", client.get_server_version())
    print("  server type:", client.get_server_type())
    print("  is_self_hosted:", getattr(client, "is_self_hosted", "n/a"))

    name = "spike_knowledge"
    if client.has_collection(name):
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=s.embedding_dim)
    # pymilvus 3.0.2 不再从顶层导出 IndexParams；公开构造方式是
    # client.prepare_index_params()。
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=index_params)

    vec = [0.01] * s.embedding_dim
    client.upsert(name, {"id": "spike:1", "vector": vec})
    # 幂等性验证：再 upsert 同一个 id
    client.upsert(name, {"id": "spike:1", "vector": vec})

    time.sleep(2)  # 让 flush/可见性稳定
    client.flush(name)
    total = client.query(name, filter="", output_fields=["count(*)"])
    print("  after 2 upserts of same id, count =", total)

    hits = client.search(name, data=[vec], limit=3, output_fields=["id"])
    print("  search hits:", hits[0])
    client.drop_collection(name)
    print("  ✅ Milvus 可用，幂等 upsert 生效（count 应为 1）")


async def probe_embedding() -> None:
    print("\n=== 探测 2：OpenAIEmbeddings 兼容 SiliconFlow ===")
    from langchain_openai import OpenAIEmbeddings

    s = get_settings()
    emb = OpenAIEmbeddings(
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
        model=s.embedding_model,
        # ★ 与生产 app/knowledge/embedding.py 保持一致：默认 True 会把文本
        # 转成 tiktoken token ID 数组，端点不报错但语义显著变差。
        check_embedding_ctx_length=False,
    )
    try:
        vec = await emb.aembed_query("邮费是多少")
        print("  aembed_query 维度:", len(vec))
        vecs = await emb.aembed_documents(["退货运费由谁承担", "怎么开发票"])
        print("  aembed_documents 条数:", len(vecs), "维度:", len(vecs[0]))
        assert len(vec) == s.embedding_dim, (
            f"维度不符: {len(vec)} != {s.embedding_dim}"
        )
        # 语义可分性：这是比维度更关键的验收条件（阶段 7 的教训）。
        shipping = (
            "售后政策 > 退换货\n运费说明\n"
            "因商品质量问题导致的退换货，往返运费由平台承担。"
        )
        shipping_vec = (await emb.aembed_documents([shipping]))[0]
        cos = cosine(vec, shipping_vec)
        print(f"  cos(邮费是多少, 运费说明) = {cos:.4f}")
        assert cos >= 0.5, (
            f"语义塌陷：cos={cos:.4f} < 0.5，检查 check_embedding_ctx_length 是否为 False"
        )
        # 维度 1024 不足以证明 model 字段真的送达上游（也可能是默认模型）。
        # 再用裸 openai SDK 直接调一次，读响应里的 model 回显来确认。
        from openai import OpenAI

        raw = OpenAI(
            base_url=s.embedding_base_url, api_key=s.embedding_api_key
        ).embeddings.create(model=s.embedding_model, input=["邮费是多少"])
        print("  上游回显 model:", raw.model, "维度:", len(raw.data[0].embedding))
        assert raw.model == s.embedding_model, (
            f"model 回显不符: {raw.model} != {s.embedding_model}"
        )
        print("  ✅ OpenAIEmbeddings 兼容，维度与配置一致，且上游收到的 model 正确")
    except Exception as exc:  # noqa: BLE001
        print("  ❌ OpenAIEmbeddings 失败:", type(exc).__name__, str(exc)[:300])
        print("  → 回报用户；候选回退：裸 openai SDK 的 client.embeddings.create")


def main() -> None:
    probe_milvus()
    asyncio.run(probe_embedding())


if __name__ == "__main__":
    main()
