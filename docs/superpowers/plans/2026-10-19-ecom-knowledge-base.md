# ch03 知识库与向量检索 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 的关键词查表升级为 Milvus dense 向量语义检索，工具签名不变；交付离线建库（Markdown 结构感知切分 + 存量 FAQ + 对话挖掘）与双写幂等状态机。

**Architecture:** 新增 `app/knowledge/` 子系统，六层边界：`splitter`（纯函数切分）→ `sources`（三源 → Chunk）→ `builder`（MySQL 权威源 + Milvus 双写状态机）→ `vector_store` / `embedding`（外部依赖封装）→ `retrieval`（在线检索）→ `app/tools/knowledge.py`（只调 retrieval）。离线任务放 `jobs/` 下，独立 CLI，不占应用进程。

**Tech Stack:** Python 3.12.14 / uv / FastAPI 0.141.1 / SQLAlchemy 2.0.54 async / MySQL 8.4（宿主 3307）/ Milvus Standalone（Docker）/ pymilvus 3.0.2 / langchain-openai 1.6.2（`OpenAIEmbeddings`）/ SiliconFlow `BAAI/bge-m3` 1024 维

**Spec:** `docs/superpowers/specs/2026-10-19-ecom-knowledge-base-design.md`

## Global Constraints

- **Python 3.12.14**，依赖管理用 `uv`（无 `pip3`）。所有命令在仓库根 `/home/zzx/custermerHelp` 执行。
- **数据库**：MySQL 在宿主 **3307**（3306 被占用），库 `customer_service` / 测试库 `customer_service_test`。
- **Milvus**：Standalone（Docker），URI `http://127.0.0.1:19530`，集合 `knowledge`。
- **Embedding**：`https://api.siliconflow.cn/v1`，模型 `BAAI/bge-m3`，**1024 维**，度量 **COSINE**。key 在 `.env` 的 `APP_EMBEDDING_API_KEY`，**绝不进 git**。
- **`query_faq` 的签名一个字不改**：`query_faq(keyword: str) -> str`。只改 docstring。
- **`chunk_id` 必须确定性**（`source:ordinal`），否则验收 2 的幂等不成立。
- **切点不许落在句子中间**（找不到句末标点时才允许硬切）。
- **Milvus 不可用时 `query_faq` 返回降级文案，不抛异常**。
- 脚本一律 `uv run python -m 包.模块`，不用 `uv run python 路径/文件.py`。
- 提交信息末尾必须带：`Co-Authored-By: Claude Code <noreply@anthropic.com>`
- 每完成一个任务，向 `dev-notes/ch03.md` 追记一段（由控制者写）。

## Review Focus

以下五类输入/失效模式 spec 隐含要求但容易被实现漏掉，各自在标注的任务里加测试：

1. **超长无标点文本**（一整段没有句号）—— 用户期望是**切出内容且不死循环**，实际常见故障是无限循环或整段被丢。→ Task 3
2. **代码块内含 `#` 或 `|`** —— 用户期望代码块被原样保留、不被误判成标题或表格。→ Task 3
3. **Milvus 可达但集合为空** —— 用户期望 `query_faq` 返回「没有找到」，而不是崩在空结果解包上。→ Task 9
4. **同一文档连续建库两次** —— 用户期望 Milvus 实体数不增长（幂等），而不是翻倍。→ Task 8
5. **Embedding 返回维度与集合维度不一致** —— 用户期望在建库时就报出明确错误，而不是让 Milvus 在写入时抛一个难懂的 gRPC 错误。→ Task 5

---

### Task 0: 阻塞性探测（spike）

**这一任务不产出生产代码。** 产出是**结论**，确认 spec §15 的两项阻塞性未决项。任一不通 → **停下来回报用户**，不自行换方案。

**Files:**
- Create: `scripts/spike_ch03.py`（一次性探针，提交以便复现）

**Interfaces:**
- Consumes: `app.config.get_settings()`
- Produces: 记录进 `dev-notes/ch03.md` 的结论；Milvus 镜像 tag 与 embedding 客户端形态

- [ ] **Step 1: 起 Milvus Standalone**

先确定镜像 tag。pymilvus 3.0.2 的文档提到 "Milvus 3.0 and later"，所以**优先试 3.0.x**：

```bash
docker pull milvusdb/milvus:v3.0.0 2>&1 | tail -2
# 若该 tag 不存在，用下面这条列出可用 tag 再选
# curl -s https://registry.hub.docker.com/v2/repositories/milvusdb/milvus/tags?page_size=100 | python3 -c 'import json,sys;print("\n".join(t["name"] for t in json.load(sys.stdin)["results"] if t["name"].startswith("v3")))'
```

把选定的 tag 写进 `docker-compose.yml`（Task 2 会正式落盘，本步先手工起容器验证）：

```bash
docker network create milvus-net 2>/dev/null || true

docker run -d --name milvus-etcd --network milvus-net \
  -e ETCD_AUTO_COMPACTION_MODE=revision \
  -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
  -e ETCD_QUOTA_BACKEND_BYTES=4294967296 \
  quay.io/coreos/etcd:v3.5.18 \
  etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

docker run -d --name milvus-minio --network milvus-net \
  -e MINIO_ACCESS_KEY=minioadmin -e MINIO_SECRET_KEY=minioadmin \
  minio/minio:RELEASE.2024-05-28T17-19-04Z \
  minio server /minio_data

docker run -d --name milvus-standalone --network milvus-net \
  -e ETCD_ENDPOINTS=etcd:2379 -e MINIO_ADDRESS=minio:9000 \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:<选定tag> milvus run standalone

# 等到 healthy
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:9091/healthz >/dev/null 2>&1 && break
  sleep 2
done
curl -s http://127.0.0.1:9091/healthz; echo
```

- [ ] **Step 2: 写探针 `scripts/spike_ch03.py`**

```python
"""Task 0 阻塞性探测：embedding 客户端兼容性 + Milvus 客户端/服务端匹配。"""

import asyncio

from pymilvus import DataType, IndexParams, MilvusClient, utility

from app.config import get_settings


def probe_milvus() -> None:
    print("=== 探测 1：Milvus 连接与版本 ===")
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    print("  server version:", utility.get_server_version(
        using=client._using) if hasattr(client, "_using") else "n/a")
    print("  is_self_hosted:", getattr(client, "is_self_hosted", "n/a"))

    name = "spike_knowledge"
    if client.has_collection(name):
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=s.embedding_dim)
    index_params = IndexParams()
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
    )
    try:
        vec = await emb.aembed_query("邮费是多少")
        print("  aembed_query 维度:", len(vec))
        vecs = await emb.aembed_documents(["退货运费由谁承担", "怎么开发票"])
        print("  aembed_documents 条数:", len(vecs), "维度:", len(vecs[0]))
        assert len(vec) == s.embedding_dim, f"维度不符: {len(vec)} != {s.embedding_dim}"
        print("  ✅ OpenAIEmbeddings 兼容，维度与配置一致")
    except Exception as exc:  # noqa: BLE001
        print("  ❌ OpenAIEmbeddings 失败:", type(exc).__name__, str(exc)[:300])
        print("  → 回报用户；候选回退：裸 openai SDK 的 client.embeddings.create")


def main() -> None:
    probe_milvus()
    asyncio.run(probe_embedding())


if __name__ == "__main__":
    import time

    main()
```

> 实现时把 `import time` 移到文件顶部；上面的写法只是为了让阅读顺序清楚。

先在 `.env` 加好 §10 的配置项，再跑：

```bash
uv run python -m scripts.spike_ch03
```

Expected:
- Milvus 连接成功，`count = 1`（两次 upsert 同一 id 只留一条）
- embedding 维度 **1024**，与配置一致

- [ ] **Step 3: 按结论处置**

- **Milvus 不通** → 检查镜像 tag 与 pymilvus 版本；仍不通 → **停下来回报用户**
- **`OpenAIEmbeddings` 不通** → 回报用户，候选回退是裸 `openai` SDK；**不要自行改架构**
- **全通** → 把镜像 tag 记进 `dev-notes/ch03.md` 与 Task 2 的 compose 文件

- [ ] **Step 4: 清理探针容器**

```bash
docker rm -f milvus-standalone milvus-etcd milvus-minio
docker network rm milvus-net
```

（Task 2 会用 compose 正式起，不保留手工容器。）

- [ ] **Step 5: 提交**

```bash
git add scripts/spike_ch03.py .env.example
git commit -m "chore: Task 0 阻塞性探测（embedding 兼容性 + Milvus 版本匹配）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 1: 配置与依赖

**Files:**
- Modify: `pyproject.toml`, `app/config.py`, `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `Settings` 新增字段 `embedding_base_url: str`、`embedding_api_key: str`、`embedding_model: str`、`embedding_dim: int`、`milvus_uri: str`、`milvus_collection: str`、`retrieval_top_k: int`、`chunk_max_chars: int`、`chunk_overlap_chars: int`、`embed_batch_size: int`

- [ ] **Step 1: 加依赖**

在 `pyproject.toml` 的 `dependencies` 追加：

```toml
    "pymilvus>=3.0.2",
```

```bash
uv sync
```

- [ ] **Step 2: 写失败的测试（追加到 `tests/test_config.py`）**

```python
def test_knowledge_defaults():
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.milvus_uri == "http://127.0.0.1:19530"
    assert s.milvus_collection == "knowledge"
    assert s.retrieval_top_k == 3
    assert s.chunk_max_chars == 400
    assert s.chunk_overlap_chars == 80
    assert s.embed_batch_size == 16


def test_chunk_params_must_be_sane():
    import pytest
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_max_chars=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_overlap_chars=-1)
```

- [ ] **Step 3: 跑测试确认失败**

```bash
uv run pytest tests/test_config.py -v
```
Expected: FAIL — `AttributeError` / 字段不存在

- [ ] **Step 4: 加配置字段**

`app/config.py` 的 `Settings` 追加（沿用现有 `Field` 风格）：

```python
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    retrieval_top_k: int = Field(default=3, gt=0)
    chunk_max_chars: int = Field(default=400, gt=0)
    chunk_overlap_chars: int = Field(default=80, ge=0)
    embed_batch_size: int = Field(default=16, gt=0)
```

若 `Field` 未导入，补 `from pydantic import Field`。

`.env.example` 追加：

```ini
APP_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
APP_EMBEDDING_API_KEY=sk-your-key-here
APP_EMBEDDING_MODEL=BAAI/bge-m3
APP_EMBEDDING_DIM=1024
APP_MILVUS_URI=http://127.0.0.1:19530
APP_MILVUS_COLLECTION=knowledge
APP_RETRIEVAL_TOP_K=3
APP_CHUNK_MAX_CHARS=400
APP_CHUNK_OVERLAP_CHARS=80
APP_EMBED_BATCH_SIZE=16
```

真实 `.env` 同步追加（key 用用户提供的值）。

- [ ] **Step 5: 跑测试确认通过**

```bash
uv run pytest tests/test_config.py -v
```
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml uv.lock app/config.py tests/test_config.py .env.example
git commit -m "feat: 知识库配置项与 pymilvus 依赖

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: 基础设施（Milvus compose + 两张 MySQL 表）

**Files:**
- Modify: `docker-compose.yml`, `scripts/schema.sql`, `app/db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Consumes: Task 1 的配置
- Produces: Milvus 服务可达；`KnowledgeChunk` / `MinedQaStaging` 两个 ORM 模型

- [ ] **Step 1: 扩 `docker-compose.yml`**

在 `services:` 下追加三个服务，tag 用 Task 0 实测确定的版本：

```yaml
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    container_name: custermerhelp-etcd
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - ./volumes/etcd:/etcd

  minio:
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    container_name: custermerhelp-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    volumes:
      - ./volumes/minio:/minio_data

  milvus:
    image: milvusdb/milvus:<Task0 确定的 tag>
    container_name: custermerhelp-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 10s
      timeout: 5s
      retries: 30
```

`.gitignore` 追加 `volumes/`。

```bash
docker compose up -d
until curl -sf http://127.0.0.1:9091/healthz >/dev/null; do sleep 2; done
echo "milvus healthy"
```

- [ ] **Step 2: 扩 `scripts/schema.sql`**

追加 §6.1 的两张表（`knowledge_chunks` 与 `mined_qa_staging`），逐字使用 spec 的 DDL。

```bash
docker compose down -v && docker compose up -d   # 重建使 initdb 生效
uv run python -m scripts.seed                     # 同时重建测试库
```

- [ ] **Step 3: 写失败的测试（追加到 `tests/test_db_models.py`）**

```python
def test_knowledge_tables_registered():
    t = Base.metadata.tables
    assert "knowledge_chunks" in t
    assert "mined_qa_staging" in t


def test_knowledge_chunk_columns():
    cols = set(Base.metadata.tables["knowledge_chunks"].columns.keys())
    assert cols == {
        "chunk_id", "category", "questions", "answer",
        "section_path", "content_type", "is_key_clause",
        "prev_chunk_id", "next_chunk_id",
        "status", "vector_id", "content_hash",
        "created_at", "updated_at",
    }


def test_knowledge_chunk_pk_is_chunk_id():
    t = Base.metadata.tables["knowledge_chunks"]
    assert [c.name for c in t.primary_key.columns] == ["chunk_id"]
```

- [ ] **Step 4: 跑测试确认失败**

```bash
uv run pytest tests/test_db_models.py -v
```

- [ ] **Step 5: 加 ORM 模型**

在 `app/db/models.py` 追加（沿用现有 `Mapped` / `mapped_column` 风格）：

```python
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(
        Enum("policy", "faq", "mined", name="knowledge_content_type")
    )
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("待向量化", "已向量化", name="knowledge_status"), default="待向量化"
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class MinedQaStaging(Base):
    __tablename__ = "mined_qa_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("待入库", "已入库", "已丢弃", name="mined_status"), default="待入库"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
```

需补 `from sqlalchemy import Boolean`。

- [ ] **Step 6: 跑测试确认通过**

```bash
uv run pytest tests/test_db_models.py -v
uv run pytest -q
```

- [ ] **Step 7: 提交**

```bash
git add docker-compose.yml scripts/schema.sql app/db/models.py tests/test_db_models.py .gitignore
git commit -m "feat: Milvus Standalone 容器与知识库两张表

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: 切分器（本章核心之一）

**Files:**
- Create: `app/knowledge/__init__.py`, `app/knowledge/chunk.py`, `app/knowledge/splitter.py`
- Test: `tests/test_splitter.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `Chunk`（frozen dataclass，字段见下）
  - `split_markdown(text: str, *, source_key: str, content_type: str = "policy", max_chars: int = 400, overlap_chars: int = 80) -> list[Chunk]`

- [ ] **Step 1: 写 `app/knowledge/chunk.py`**

```python
import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: Literal["policy", "faq", "mined"]
    is_key_clause: bool = False
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    @property
    def embedding_text(self) -> str:
        """送去向量化的文本：三字段用换行拼接。"""
        return f"{self.category}\n{self.questions}\n{self.answer}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()
```

- [ ] **Step 2: 写失败的测试 `tests/test_splitter.py`**

```python
from app.knowledge.splitter import split_markdown


BASIC = """# 售后政策

## 退换货

### 退货

签收后 7 天内可无理由退货。定制类商品不支持。

### 运费

因商品质量问题导致的退换货，往返运费由平台承担。因个人原因发起的退换货，寄回运费需您自行承担。
"""


def test_heading_stack_builds_section_path():
    chunks = split_markdown(BASIC, source_key="policy.md")
    paths = [c.section_path for c in chunks]
    assert "售后政策 > 退换货 > 退货" in paths
    assert "售后政策 > 退换货 > 运费" in paths


def test_questions_is_leaf_heading_and_category_is_parent_path():
    chunks = split_markdown(BASIC, source_key="policy.md")
    c = next(c for c in chunks if c.section_path.endswith("运费"))
    assert c.questions == "运费"
    assert c.category == "售后政策 > 退换货"


def test_chunk_ids_are_deterministic_and_unique():
    a = [c.chunk_id for c in split_markdown(BASIC, source_key="policy.md")]
    b = [c.chunk_id for c in split_markdown(BASIC, source_key="policy.md")]
    assert a == b
    assert len(set(a)) == len(a)
    assert all(x.startswith("policy.md:") for x in a)


def test_prev_next_pointers_are_linked():
    chunks = split_markdown(BASIC, source_key="policy.md")
    assert chunks[0].prev_chunk_id is None
    assert chunks[-1].next_chunk_id is None
    for i in range(len(chunks) - 1):
        assert chunks[i].next_chunk_id == chunks[i + 1].chunk_id
        assert chunks[i + 1].prev_chunk_id == chunks[i].chunk_id


def test_long_paragraph_split_ends_on_sentence_boundary():
    text = "# 政策\n\n" + "这是一句话。" * 200
    chunks = split_markdown(text, source_key="long.md", max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.endswith("。"), f"chunk 未以句末标点结尾: ...{c.answer[-20:]!r}"
        assert len(c.answer) <= 100


def test_overlap_is_a_whole_sentence():
    text = "# 政策\n\n" + "第一句话。" * 100
    chunks = split_markdown(text, source_key="over.md", max_chars=60, overlap_chars=30)
    for i in range(1, len(chunks)):
        prev = chunks[i - 1].answer
        cur = chunks[i].answer
        assert cur.startswith("第一句话。"), "重叠部分应从完整句子开始"


def test_table_split_by_row_and_repeats_header():
    text = """# 商品

## 库存

| 商品 | 尺码 | 库存 |
|---|---|---|
| 跑步鞋 | 42 | 12 |
| 运动袜 | L | 88 |
"""
    chunks = split_markdown(text, source_key="goods.md")
    table_chunks = [c for c in chunks if "跑步鞋" in c.answer or "运动袜" in c.answer]
    assert len(table_chunks) == 2
    for c in table_chunks:
        assert "| 商品 | 尺码 | 库存 |" in c.answer
        assert "|---|" in c.answer


def test_code_fence_is_never_split():
    body = "```python\n" + "# 这不是标题\n" * 100 + "```"
    text = f"# 规则\n\n{body}\n"
    chunks = split_markdown(text, source_key="code.md", max_chars=80)
    code_chunks = [c for c in chunks if "这不是标题" in c.answer]
    assert len(code_chunks) == 1
    assert code_chunks[0].answer.startswith("```python")


def test_important_blockquote_sets_key_clause():
    text = "# 规则\n\n> [!IMPORTANT]\n> 每人限购 2 件。\n"
    chunks = split_markdown(text, source_key="rule.md")
    assert any(c.is_key_clause for c in chunks)


def test_text_without_punctuation_terminates():
    text = "# 规则\n\n" + "啊" * 1000
    chunks = split_markdown(text, source_key="nopunct.md", max_chars=100)
    assert sum(len(c.answer) for c in chunks) >= 1000
    assert all(len(c.answer) <= 100 for c in chunks)
```

- [ ] **Step 3: 跑测试确认失败**

```bash
uv run pytest tests/test_splitter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.knowledge'`

- [ ] **Step 4: 写 `app/knowledge/splitter.py`**

```python
"""Markdown 结构感知切分器。

硬规则（spec §5）：
- 超长递归切：段落 → 句子 → 才硬切
- 切点必须落在句末标点（。！？；）之后
- 表格按行切，每块复述表头
- 代码块整体保留，绝不切
- chunk_id 确定性：f"{source_key}:{ordinal}"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge.chunk import Chunk

_SENTENCE_DELIMS = "。！？；"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class _Block:
    kind: str                     # paragraph | list | table | code | quote
    lines: list[str]
    section_path: list[str]
    is_key_clause: bool = False


def _split_sentences(text: str) -> list[str]:
    """按句末标点切，保留标点。无标点时整体返回。"""
    parts: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_DELIMS:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def _pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """把句子组装成不超过 max_chars 的块，相邻块重叠整句。"""
    out: list[str] = []
    cur = ""
    for sent in sentences:
        if len(cur) + len(sent) <= max_chars:
            cur += sent
            continue
        if cur:
            out.append(cur)
            # 用尾部若干完整句子做重叠
            tail: list[str] = []
            acc = ""
            for s in reversed(_split_sentences(cur)):
                if len(s) + acc > overlap_chars:
                    break
                tail.insert(0, s)
                acc = s + acc
            cur = acc + sent
        else:
            # 单句已超长：硬切（spec 规则 ③）
            while len(sent) > max_chars:
                out.append(sent[:max_chars])
                sent = sent[max_chars:]
            cur = sent
    if cur.strip():
        out.append(cur)
    return out


def _parse_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    stack: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            fence_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence_lines.append(lines[i])
                i += 1
            if i < len(lines):
                fence_lines.append(lines[i])
                i += 1
            blocks.append(_Block("code", fence_lines, [s for s in stack if s]))
            continue

        # 表格
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_Block("table", table_lines, [s for s in stack if s]))
            continue

        # 引用块
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i])
                i += 1
            blocks.append(
                _Block(
                    "quote",
                    quote_lines,
                    [s for s in stack if s],
                    is_key_clause="[!IMPORTANT]" in quote_lines[0],
                )
            )
            continue

        # 列表
        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            blocks.append(_Block("list", list_lines, [s for s in stack if s]))
            continue

        # 段落：直到空行或块起始
        para = []
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt) or nxt.strip().startswith(("```", "|", ">")):
                break
            if re.match(r"^(\s*)([-*+]|\d+\.)\s+", nxt):
                break
            para.append(nxt)
            i += 1
        blocks.append(_Block("paragraph", para, [s for s in stack if s]))

    return blocks


def _table_row_chunks(lines: list[str], max_chars: int) -> list[str]:
    """表格按行切，每块复述表头（首行 + 分隔行）。"""
    if len(lines) < 2:
        return ["\n".join(lines)] if lines else []
    header = lines[:2]
    rows = [ln for ln in lines[2:] if ln.strip()]
    if not rows:
        return ["\n".join(header)]
    return ["\n".join(header + [row]) for row in rows]


def split_markdown(
    text: str,
    *,
    source_key: str,
    content_type: str = "policy",
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0

    for block in _parse_blocks(text):
        path = " > ".join(block.section_path)
        category = " > ".join(block.section_path[:-1]) or path
        questions = block.section_path[-1] if block.section_path else source_key

        if block.kind == "table":
            pieces = _table_row_chunks(block.lines, max_chars)
        elif block.kind == "code":
            pieces = ["\n".join(block.lines)]
        else:
            raw = "\n".join(block.lines)
            if len(raw) <= max_chars:
                pieces = [raw]
            else:
                pieces = _pack_sentences(_split_sentences(raw), max_chars, overlap_chars)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source_key}:{ordinal}",
                    category=category,
                    questions=questions,
                    answer=piece,
                    section_path=path,
                    content_type=content_type,  # type: ignore[arg-type]
                    is_key_clause=block.is_key_clause,
                )
            )
            ordinal += 1

    # 回填前后指针
    linked: list[Chunk] = []
    for idx, c in enumerate(chunks):
        linked.append(
            Chunk(
                chunk_id=c.chunk_id,
                category=c.category,
                questions=c.questions,
                answer=c.answer,
                section_path=c.section_path,
                content_type=c.content_type,
                is_key_clause=c.is_key_clause,
                prev_chunk_id=chunks[idx - 1].chunk_id if idx > 0 else None,
                next_chunk_id=chunks[idx + 1].chunk_id if idx < len(chunks) - 1 else None,
            )
        )
    return linked
```

`app/knowledge/__init__.py` 留空。

- [ ] **Step 5: 跑测试确认通过**

```bash
uv run pytest tests/test_splitter.py -v
```

若 `test_long_paragraph_split_ends_on_sentence_boundary` 因硬切边界失败，**先检查是切分逻辑错还是断言过严**——`max_chars=100` 且每句 5 字时应能整除而无硬切。修正实现前先把原因写清楚。

- [ ] **Step 6: 提交**

```bash
git add app/knowledge/ tests/test_splitter.py
git commit -m "feat: Markdown 结构感知切分器（句子边界、表格表头、确定性 chunk_id）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: 三源加载器

**Files:**
- Create: `app/knowledge/sources/__init__.py`, `app/knowledge/sources/markdown.py`, `app/knowledge/sources/faq_table.py`, `app/knowledge/sources/mined.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `Chunk`、`split_markdown`、`app.db.models.Faq` / `MinedQaStaging`
- Produces:
  - `load_markdown_dir(path: Path, *, content_type: str = "policy", max_chars: int, overlap_chars: int) -> list[Chunk]`
  - `load_faq_chunks(session, *, content_type: str = "faq") -> list[Chunk]`
  - `load_mined_chunks(session) -> list[Chunk]`

- [ ] **Step 1: 写失败的测试 `tests/test_sources.py`**

```python
import pytest

from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.sources.mined import load_mined_chunks


def test_markdown_dir_scans_md_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n内容甲。\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n内容乙。\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("忽略", encoding="utf-8")

    chunks = load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)
    answers = {c.answer for c in chunks}
    assert answers == {"内容甲。", "内容乙。"}
    assert {c.content_type for c in chunks} == {"policy"}


def test_markdown_source_key_is_filename(tmp_path):
    (tmp_path / "售后政策.md").write_text("# 政策\n\n七天无理由退货。\n", encoding="utf-8")
    chunks = load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)
    assert chunks[0].chunk_id.startswith("售后政策.md:")


async def test_faq_source_maps_fields(db_session):
    chunks = await load_faq_chunks(db_session)
    assert len(chunks) == 12
    c = next(c for c in chunks if "退货政策" in c.questions)
    assert c.content_type == "faq"
    assert c.category == "售后"
    assert c.chunk_id.startswith("faq:")


async def test_mined_source_only_reads_pending(db_session):
    from app.db.models import MinedQaStaging

    db_session.add(MinedQaStaging(question="能开专票吗", answer="可以。", category="咨询",
                                  status="待入库"))
    db_session.add(MinedQaStaging(question="已入库的", answer="x", category="咨询",
                                  status="已入库"))
    await db_session.flush()

    chunks = await load_mined_chunks(db_session)
    assert len(chunks) == 1
    assert chunks[0].questions == "能开专票吗"
    assert chunks[0].content_type == "mined"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_sources.py -v
```

- [ ] **Step 3: 实现三个加载器**

`app/knowledge/sources/markdown.py`：

```python
from pathlib import Path

from app.knowledge.chunk import Chunk
from app.knowledge.splitter import split_markdown


def load_markdown_dir(
    path: Path,
    *,
    content_type: str = "policy",
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(path.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        out.extend(
            split_markdown(
                text,
                source_key=md.name,
                content_type=content_type,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return out
```

`app/knowledge/sources/faq_table.py`：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Faq
from app.knowledge.chunk import Chunk


async def load_faq_chunks(session: AsyncSession, *, content_type: str = "faq") -> list[Chunk]:
    rows = list((await session.execute(select(Faq).order_by(Faq.id))).scalars())
    return [
        Chunk(
            chunk_id=f"faq:{r.id}",
            category=r.category,
            questions=r.question,
            answer=r.answer,
            section_path=r.category,
            content_type=content_type,  # type: ignore[arg-type]
        )
        for r in rows
    ]
```

`app/knowledge/sources/mined.py`：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MinedQaStaging
from app.knowledge.chunk import Chunk


async def load_mined_chunks(session: AsyncSession) -> list[Chunk]:
    stmt = select(MinedQaStaging).where(MinedQaStaging.status == "待入库").order_by(
        MinedQaStaging.id
    )
    rows = list((await session.execute(stmt)).scalars())
    return [
        Chunk(
            chunk_id=f"mined:{r.id}",
            category=r.category,
            questions=r.question,
            answer=r.answer,
            section_path=r.category,
            content_type="mined",
        )
        for r in rows
    ]
```

`app/knowledge/sources/__init__.py` 留空。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_sources.py -v
```

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/sources/ tests/test_sources.py
git commit -m "feat: 三源知识加载器（Markdown / 存量 FAQ / 挖掘暂存）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Embedding 与 Milvus 封装

**Files:**
- Create: `app/knowledge/embedding.py`, `app/knowledge/vector_store.py`
- Test: `tests/test_vector_store.py`（单元，用假 client）、`tests/test_embedding_dim.py`

**Interfaces:**
- Consumes: Task 1 配置、Task 0 结论
- Produces:
  - `async def embed_texts(texts: list[str]) -> list[list[float]]`
  - `class VectorStore`：`ensure_collection() -> None`、`upsert(ids: list[str], vectors: list[list[float]]) -> None`、`search(vector: list[float], top_k: int) -> list[str]`、`count() -> int`

- [ ] **Step 1: 写失败的测试**

`tests/test_embedding_dim.py`：

```python
import pytest

from app.knowledge.embedding import EmbeddingDimensionError, embed_texts


async def test_embed_texts_returns_configured_dim(monkeypatch):
    async def fake(*a, **kw):
        return [[0.0] * 1024]

    monkeypatch.setattr("app.knowledge.embedding._embed_batch", fake)
    out = await embed_texts(["测试"])
    assert len(out[0]) == 1024


async def test_embed_texts_rejects_wrong_dim(monkeypatch):
    async def fake(*a, **kw):
        return [[0.0] * 512]

    monkeypatch.setattr("app.knowledge.embedding._embed_batch", fake)
    with pytest.raises(EmbeddingDimensionError) as exc:
        await embed_texts(["测试"])
    assert "512" in str(exc.value) and "1024" in str(exc.value)
```

`tests/test_vector_store.py`：

```python
from unittest.mock import MagicMock

from app.knowledge.vector_store import VectorStore


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
    from app.knowledge.vector_store import VectorStoreUnavailable
    import pytest

    client = MagicMock()
    client.has_collection.return_value = True
    client.search.side_effect = RuntimeError("connection refused")
    with pytest.raises(VectorStoreUnavailable):
        make_store(client).search([0.0] * 1024, top_k=3)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_embedding_dim.py tests/test_vector_store.py -v
```

- [ ] **Step 3: 实现**

`app/knowledge/embedding.py`：

```python
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


class EmbeddingDimensionError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
        model=s.embedding_model,
    )


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    return await _client().aembed_documents(texts)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """分批向量化，并校验维度与配置一致。"""
    s = get_settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), s.embed_batch_size):
        batch = texts[i : i + s.embed_batch_size]
        vectors = await _embed_batch(batch)
        for v in vectors:
            if len(v) != s.embedding_dim:
                raise EmbeddingDimensionError(
                    f"embedding 维度不符：期望 {s.embedding_dim}，实际 {len(v)}"
                )
        out.extend(vectors)
    return out
```

`app/knowledge/vector_store.py`：

```python
from pymilvus import DataType, IndexParams, MilvusClient

from app.config import get_settings


class VectorStoreUnavailable(RuntimeError):
    pass


class VectorStore:
    def __init__(self, *, client=None, collection: str | None = None, dim: int | None = None):
        s = get_settings()
        self._client = client or MilvusClient(uri=s.milvus_uri)
        self._collection = collection or s.milvus_collection
        self._dim = dim or s.embedding_dim

    def ensure_collection(self) -> None:
        if self._client.has_collection(self._collection):
            return
        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        index_params = IndexParams()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(self._collection, schema=schema, index_params=index_params)

    def upsert(self, ids: list[str], vectors: list[list[float]]) -> None:
        rows = [{"id": i, "vector": v} for i, v in zip(ids, vectors)]
        try:
            self._client.upsert(self._collection, rows)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc

    def search(self, vector: list[float], top_k: int) -> list[str]:
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
            rows = self._client.query(self._collection, filter="", output_fields=["count(*)"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(str(exc)) from exc
        return int(rows[0]["count(*)"]) if rows else 0
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_embedding_dim.py tests/test_vector_store.py -v
```

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/embedding.py app/knowledge/vector_store.py tests/test_embedding_dim.py tests/test_vector_store.py
git commit -m "feat: embedding 客户端（维度校验）与 Milvus 集合封装

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: 知识块仓储

**Files:**
- Create: `app/repositories/knowledge_chunk.py`
- Test: `tests/test_knowledge_repo.py`

**Interfaces:**
- Consumes: `KnowledgeChunk` 模型
- Produces:
  - `async def upsert_chunks(session, chunks: list[Chunk]) -> None`（按 `chunk_id` upsert；`content_hash` 变化才重置状态）
  - `async def list_pending(session) -> list[KnowledgeChunk]`
  - `async def mark_vectorized(session, chunk_ids: list[str], vector_ids: list[str]) -> None`
  - `async def get_by_ids(session, chunk_ids: list[str]) -> list[KnowledgeChunk]`
  - `async def count_all(session) -> int` / `count_vectorized(session) -> int`

- [ ] **Step 1: 写失败的测试 `tests/test_knowledge_repo.py`**

```python
from app.knowledge.chunk import Chunk
from app.repositories import knowledge_chunk as repo


def make_chunk(cid="policy.md:0", answer="内容。", content_type="policy"):
    return Chunk(
        chunk_id=cid, category="分类", questions="标题",
        answer=answer, section_path="分类 > 标题", content_type=content_type,
    )


async def test_new_chunks_are_pending(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    pending = await repo.list_pending(db_session)
    assert [p.chunk_id for p in pending] == ["policy.md:0"]


async def test_same_hash_keeps_vectorized_status(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    await repo.upsert_chunks(db_session, [make_chunk()])   # 内容未变
    assert await repo.list_pending(db_session) == []


async def test_changed_hash_resets_to_pending(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    await repo.upsert_chunks(db_session, [make_chunk(answer="改过的内容。")])
    assert [p.chunk_id for p in await repo.list_pending(db_session)] == ["policy.md:0"]


async def test_counts(db_session):
    # 用增量断言，不依赖库里的绝对条数（其他测试可能 commit 过数据）
    before_all = await repo.count_all(db_session)
    before_vec = await repo.count_vectorized(db_session)
    await repo.upsert_chunks(db_session, [make_chunk("a:0"), make_chunk("a:1")])
    await repo.mark_vectorized(db_session, ["a:0"], ["a:0"])
    assert await repo.count_all(db_session) == before_all + 2
    assert await repo.count_vectorized(db_session) == before_vec + 1


async def test_get_by_ids_preserves_caller_order(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0"), make_chunk("a:1")])
    rows = await repo.get_by_ids(db_session, ["a:1", "a:0"])
    assert [r.chunk_id for r in rows] == ["a:0", "a:1"]  # 仓储按 id 查，排序由调用方做
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_knowledge_repo.py -v
```

- [ ] **Step 3: 实现 `app/repositories/knowledge_chunk.py`**

```python
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk
from app.knowledge.chunk import Chunk


async def upsert_chunks(session: AsyncSession, chunks: list[Chunk]) -> None:
    """按 chunk_id upsert。content_hash 未变则保留原 status；变了则重置为待向量化。"""
    for c in chunks:
        existing = await session.get(KnowledgeChunk, c.chunk_id)
        if existing is None:
            session.add(
                KnowledgeChunk(
                    chunk_id=c.chunk_id,
                    category=c.category,
                    questions=c.questions,
                    answer=c.answer,
                    section_path=c.section_path,
                    content_type=c.content_type,
                    is_key_clause=c.is_key_clause,
                    prev_chunk_id=c.prev_chunk_id,
                    next_chunk_id=c.next_chunk_id,
                    status="待向量化",
                    content_hash=c.content_hash,
                )
            )
        elif existing.content_hash != c.content_hash:
            existing.category = c.category
            existing.questions = c.questions
            existing.answer = c.answer
            existing.section_path = c.section_path
            existing.content_type = c.content_type
            existing.is_key_clause = c.is_key_clause
            existing.prev_chunk_id = c.prev_chunk_id
            existing.next_chunk_id = c.next_chunk_id
            existing.content_hash = c.content_hash
            existing.status = "待向量化"
            existing.vector_id = None
        # hash 相同：不动 status，也不改任何字段
    await session.flush()


async def list_pending(session: AsyncSession) -> list[KnowledgeChunk]:
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.status == "待向量化")
        .order_by(KnowledgeChunk.chunk_id)
    )
    return list((await session.execute(stmt)).scalars())


async def mark_vectorized(
    session: AsyncSession, chunk_ids: list[str], vector_ids: list[str]
) -> None:
    for cid, vid in zip(chunk_ids, vector_ids):
        row = await session.get(KnowledgeChunk, cid)
        if row is not None:
            row.status = "已向量化"
            row.vector_id = vid
    await session.flush()


async def get_by_ids(session: AsyncSession, chunk_ids: list[str]) -> list[KnowledgeChunk]:
    if not chunk_ids:
        return []
    stmt = select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(chunk_ids))
    return list((await session.execute(stmt)).scalars())


async def count_all(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(KnowledgeChunk))).scalar_one()


async def count_vectorized(session: AsyncSession) -> int:
    stmt = (
        select(func.count())
        .select_from(KnowledgeChunk)
        .where(KnowledgeChunk.status == "已向量化")
    )
    return (await session.execute(stmt)).scalar_one()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_knowledge_repo.py -v
```

- [ ] **Step 5: 提交**

```bash
git add app/repositories/knowledge_chunk.py tests/test_knowledge_repo.py
git commit -m "feat: knowledge_chunks 仓储（hash 驱动的状态机）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 双写 Builder

**Files:**
- Create: `app/knowledge/builder.py`
- Test: `tests/test_builder.py`

**Interfaces:**
- Consumes: `upsert_chunks` / `list_pending` / `mark_vectorized`、`embed_texts`、`VectorStore`
- Produces: `async def build(session, chunks, *, store=None, embed=embed_texts, fail_after: int | None = None) -> BuildReport`
  - `BuildReport(chunks_total: int, chunks_pending: int, vectors_written: int)`
  - `fail_after` 仅测试用：处理 N 条后抛异常，用于中断重跑测试

- [ ] **Step 1: 写失败的测试 `tests/test_builder.py`**

```python
import pytest
from sqlalchemy import delete, func, select

from app.db.base import AsyncSessionLocal
from app.db.models import KnowledgeChunk
from app.knowledge.builder import build
from app.knowledge.chunk import Chunk
from app.repositories import knowledge_chunk as repo

PREFIX = "testbuilder:"


class FakeStore:
    def __init__(self):
        self.data: dict[str, list[float]] = {}

    def ensure_collection(self):
        pass

    def upsert(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.data[i] = v

    def count(self):
        return len(self.data)


async def fake_embed(texts):
    return [[float(len(t))] * 1024 for t in texts]


def chunk(cid, answer="内容。"):
    return Chunk(cid, "分类", "标题", answer, "分类 > 标题", "policy")


@pytest.fixture
async def builder_session():
    """builder 会 commit（为了真实的重跑语义），所以不能用会回滚的 db_session。

    用独立 session + 唯一 id 前缀，结束后只清自己造的行，不碰其他测试的数据。
    """
    async with AsyncSessionLocal() as session:
        yield session
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.chunk_id.like(f"{PREFIX}%"))
        )
        await session.commit()


async def _vectorized(session) -> int:
    stmt = (
        select(func.count())
        .select_from(KnowledgeChunk)
        .where(
            KnowledgeChunk.status == "已向量化",
            KnowledgeChunk.chunk_id.like(f"{PREFIX}%"),
        )
    )
    return (await session.execute(stmt)).scalar_one()


async def test_build_writes_mysql_then_vectors(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    report = await build(builder_session, chunks, store=store, embed=fake_embed)
    assert report.chunks_total == 2
    assert report.vectors_written == 2
    assert store.count() == 2
    assert await _vectorized(builder_session) == 2


async def test_rerun_is_idempotent(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    await build(builder_session, chunks, store=store, embed=fake_embed)
    await build(builder_session, chunks, store=store, embed=fake_embed)   # 同样的输入再跑
    assert store.count() == 2          # 没有变 4
    assert await _vectorized(builder_session) == 2


async def test_interrupted_build_resumes(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}{i}") for i in range(5)]

    with pytest.raises(RuntimeError):
        await build(builder_session, chunks, store=store, embed=fake_embed, fail_after=2)

    assert store.count() == 2
    assert await _vectorized(builder_session) == 2

    report = await build(builder_session, chunks, store=store, embed=fake_embed)
    assert report.vectors_written == 5      # 补了剩下 3 条
    assert store.count() == 5               # 没有重复
    assert await _vectorized(builder_session) == 5
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_builder.py -v
```

- [ ] **Step 3: 实现 `app/knowledge/builder.py`**

```python
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.knowledge.chunk import Chunk
from app.knowledge.embedding import embed_texts
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo


@dataclass(frozen=True)
class BuildReport:
    chunks_total: int
    chunks_pending: int
    vectors_written: int


async def build(
    session: AsyncSession,
    chunks: list[Chunk],
    *,
    store=None,
    embed=embed_texts,
    fail_after: int | None = None,
) -> BuildReport:
    """双写：MySQL 记【待向量化】→ Milvus upsert → 回填状态。"""
    s = get_settings()
    store = store or VectorStore()
    store.ensure_collection()

    # ① 写 MySQL：新块待向量化；hash 未变保持原状态；hash 变了重置
    await repo.upsert_chunks(session, chunks)
    await session.commit()

    # ② 取待办集
    pending = await repo.list_pending(session)
    if not pending:
        return BuildReport(len(chunks), 0, 0)

    # ③④⑤ 分批向量化并写 Milvus，写完一批回填一批
    written = 0
    batch_size = s.embed_batch_size
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        texts = [c.embedding_text for c in batch]
        vectors = await embed(texts)

        ids = [c.chunk_id for c in batch]
        store.upsert(ids, vectors)
        await repo.mark_vectorized(session, ids, ids)
        await session.commit()
        written += len(batch)

        if fail_after is not None and written >= fail_after:
            raise RuntimeError(f"模拟中断：已写入 {written} 条")

    return BuildReport(len(chunks), len(pending), written)
```

> **注意**：每批写完就 `commit()`。这样中断时已完成批次的状态是持久的，重跑只处理剩余待办——这正是验收 2 的行为。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_builder.py -v
```

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/builder.py tests/test_builder.py
git commit -m "feat: 双写 Builder（分批提交、中断可续跑）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 样例文档与建库 CLI

**Files:**
- Create: `knowledge/售后政策.md`, `knowledge/商品FAQ.md`, `knowledge/平台规则.md`, `jobs/__init__.py`, `jobs/build_knowledge.py`
- Test: 手工验证 + 集成断言

**Interfaces:**
- Consumes: 全部前置模块
- Produces: `uv run python -m jobs.build_knowledge` 可用

- [ ] **Step 1: 写三份样例文档**

**`knowledge/售后政策.md`** —— 必须包含一节「运费」，正文用"运费"而非"邮费"（这样才能验证语义召回）：

```markdown
# 售后政策

## 退换货

### 退货政策

签收后 7 天内可无理由退货。商品需保持吊牌完整、不影响二次销售。
定制类商品不支持无理由退货。

### 运费说明

因商品质量问题导致的退换货，往返运费由平台承担。
因个人原因（尺码不合适、不喜欢等）发起的退换货，寄回运费需您自行承担。
订单实付满 99 元包邮，未满收取 8 元配送费。

### 换货流程

在订单页点「申请售后」选换货，填写要更换的规格。
换货商品寄回验收后我们会重新发货。

## 配送

### 发货时效

现货商品在付款后 48 小时内发出。预售商品以商品页标注的发货时间为准。

### 发什么快递

默认发中通或圆通，偏远地区可能转邮政。下单时无法指定快递公司。

## 发票

### 怎么开发票

订单完成后在订单详情页点「申请开票」，填写抬头和税号。
电子发票会在 3 个工作日内发送到您的邮箱。

### 发票能改抬头吗

电子发票未开具前可以修改抬头。已开具的需要先冲红再重开，请联系人工客服。
```

**`knowledge/商品FAQ.md`**：

```markdown
# 商品常见问题

## 尺码

### 鞋子尺码怎么选

商品详情页有尺码对照表。鞋类建议按脚长选，介于两个码之间时建议选大半码。

### 衣服尺码怎么选

按胸围和身高对照表选择。不确定可以联系客服报身高体重，我们帮您推荐。

## 材质

### 鞋子是什么材质

跑步鞋鞋面为透气网布，鞋底为 EVA 缓震中底。具体以商品页材质说明为准。

## 库存

| 商品 | 尺码 | 库存 |
|---|---|---|
| 轻量跑步鞋 | 42 | 12 |
| 轻量跑步鞋 | 43 | 0 |
| 纯棉圆领T恤 | L | 88 |
```

**`knowledge/平台规则.md`**：

```markdown
# 平台规则

## 价格保护

### 价保规则

支持 15 天价保：下单后 15 天内同商品官方渠道降价，可申请退还差价。

> [!IMPORTANT]
> 价保申请必须在降价发生后 15 天内提交，逾期不再受理。

## 限购

> [!IMPORTANT]
> 秒杀商品每人限购 2 件。超出部分平台有权取消订单。
```

- [ ] **Step 2: 写 `jobs/build_knowledge.py`**

```python
"""离线建库 CLI。

用法：uv run python -m jobs.build_knowledge
"""

import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.base import AsyncSessionLocal
from app.knowledge.builder import build
from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.sources.mined import load_mined_chunks

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


async def main() -> None:
    s = get_settings()
    async with AsyncSessionLocal() as session:
        md = load_markdown_dir(
            KNOWLEDGE_DIR,
            content_type="policy",
            max_chars=s.chunk_max_chars,
            overlap_chars=s.chunk_overlap_chars,
        )
        faq = await load_faq_chunks(session)
        mined = await load_mined_chunks(session)

        chunks = md + faq + mined
        print(f"切分得到 {len(chunks)} 块（Markdown {len(md)} / FAQ {len(faq)} / 挖掘 {len(mined)}）")

        report = await build(session, chunks)
        print(
            f"完成：总数 {report.chunks_total}，本次待办 {report.chunks_pending}，"
            f"本次写入 {report.vectors_written}"
        )


if __name__ == "__main__":
    asyncio.run(main())
```

`jobs/__init__.py` 留空。

- [ ] **Step 3: 跑建库**

```bash
docker compose up -d
uv run python -m jobs.build_knowledge
```
Expected: 打印切分块数与完成统计；无异常

- [ ] **Step 4: 验证双写一致性**

```bash
uv run python - <<'PY'
import asyncio
from sqlalchemy import text
from app.db.base import AsyncSessionLocal
from app.knowledge.vector_store import VectorStore

async def main():
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(text(
            "SELECT status, COUNT(*) FROM knowledge_chunks GROUP BY status"))).all()
        print("MySQL:", dict(rows))
    print("Milvus 实体数:", VectorStore().count())

asyncio.run(main())
PY
```
Expected: `已向量化` 的条数 == Milvus 实体数

- [ ] **Step 5: 再跑一次，确认幂等**

```bash
uv run python -m jobs.build_knowledge
```
Expected: `本次待办 0`，Milvus 实体数不变

- [ ] **Step 6: 提交**

```bash
git add knowledge/ jobs/
git commit -m "feat: 样例知识文档与建库 CLI

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: 在线检索与 `query_faq` 改造

**Files:**
- Create: `app/knowledge/retrieval.py`
- Modify: `app/tools/knowledge.py`
- Test: `tests/test_retrieval.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: `embed_texts`、`VectorStore`、`knowledge_chunk.get_by_ids`
- Produces:
  - `async def search(session, query: str, top_k: int | None = None, *, store=None, embed=None) -> list[KnowledgeChunk]`
  - 返回顺序严格跟随 Milvus 分数

- [ ] **Step 1: 写失败的测试 `tests/test_retrieval.py`**

```python
import pytest

from app.knowledge.retrieval import search
from app.repositories import knowledge_chunk as repo
from app.knowledge.chunk import Chunk


class FakeStore:
    def __init__(self, ids):
        self.ids = ids

    def search(self, vector, top_k):
        return self.ids[:top_k]


async def fake_embed(texts):
    return [[0.0] * 1024 for _ in texts]


def c(cid, answer="内容。"):
    return Chunk(cid, "分类", "标题", answer, "分类 > 标题", "policy")


async def test_search_returns_chunks_in_milvus_order(db_session):
    await repo.upsert_chunks(db_session, [c("a:0"), c("a:1")])
    await db_session.commit()

    out = await search(db_session, "问题", store=FakeStore(["a:1", "a:0"]), embed=fake_embed)
    assert [x.chunk_id for x in out] == ["a:1", "a:0"]


async def test_empty_hits_returns_empty_list(db_session):
    out = await search(db_session, "问题", store=FakeStore([]), embed=fake_embed)
    assert out == []


async def test_vector_store_unavailable_propagates(db_session):
    from app.knowledge.vector_store import VectorStoreUnavailable

    class Boom:
        def search(self, v, k):
            raise VectorStoreUnavailable("down")

    with pytest.raises(VectorStoreUnavailable):
        await search(db_session, "问题", store=Boom(), embed=fake_embed)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_retrieval.py -v
```

- [ ] **Step 3: 实现 `app/knowledge/retrieval.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge.embedding import embed_texts
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo


async def search(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    *,
    store=None,
    embed=None,
) -> list[KnowledgeChunk]:
    s = get_settings()
    store = store or VectorStore()
    embed = embed or embed_texts

    qv = (await embed([query]))[0]
    ids = store.search(qv, top_k or s.retrieval_top_k)
    if not ids:
        return []

    rows = await repo.get_by_ids(session, ids)
    by_id = {r.chunk_id: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]   # 严格保持 Milvus 顺序
```

- [ ] **Step 4: 改 `app/tools/knowledge.py`**

```python
"""FAQ 检索工具。ch03 起改为 Milvus dense 向量语义检索。

工具签名保持不变：query_faq(keyword: str) -> str。
"""

from langchain_core.tools import tool

from app.knowledge import retrieval
from app.knowledge.vector_store import VectorStoreUnavailable
from app.tools.context import current_context


@tool
async def query_faq(keyword: str) -> str:
    """按语义检索常见问题与政策手册（退货、运费、发票、尺码、价保等）。

    keyword 请传**用户的完整问题或其中的核心说法**——
    例如「邮费是多少」「退货运费谁承担」「怎么开发票」。
    """
    session = current_context().session
    if session is None:
        return "FAQ 查询暂时不可用（数据库未绑定）。"

    try:
        hits = await retrieval.search(session, keyword)
    except VectorStoreUnavailable:
        return "知识库暂时不可用，请稍后再试或转人工客服。"

    if not hits:
        return f"没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"

    return "\n\n".join(f"问：{h.questions}\n答：{h.answer}" for h in hits)
```

- [ ] **Step 5: 更新 `tests/test_tools.py` 中受影响的断言**

现有 `test_query_faq_hit_via_registry` / `test_query_faq_miss_via_registry` 走真实 `faq` 表；改为打桩 `retrieval.search`：

```python
async def test_query_faq_hit_via_registry(db_session, monkeypatch):
    from app.db.models import KnowledgeChunk
    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        row = KnowledgeChunk(
            chunk_id="policy.md:9", category="售后政策 > 退换货", questions="运费说明",
            answer="因商品质量问题导致的退换货，往返运费由平台承担。",
            section_path="售后政策 > 退换货 > 运费", content_type="policy",
            status="已向量化", content_hash="x",
        )
        return [row]

    monkeypatch.setattr(retrieval, "search", fake_search)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费是多少"}, "c1", conversation_id=1)
    assert out.ok is True
    assert "往返运费由平台承担" in out.content


async def test_query_faq_miss_via_registry(db_session, monkeypatch):
    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        return []

    monkeypatch.setattr(retrieval, "search", fake_search)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c2", conversation_id=1)
    assert out.ok is True
    assert "没有找到" in out.content


async def test_query_faq_degrades_when_milvus_down(db_session, monkeypatch):
    from app.knowledge import retrieval
    from app.knowledge.vector_store import VectorStoreUnavailable

    async def boom(session, query, top_k=None, **kw):
        raise VectorStoreUnavailable("down")

    monkeypatch.setattr(retrieval, "search", boom)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c3", conversation_id=1)
    assert out.ok is True
    assert "暂时不可用" in out.content
```

- [ ] **Step 6: 跑测试**

```bash
uv run pytest tests/test_retrieval.py tests/test_tools.py -v
uv run pytest -q
```

- [ ] **Step 7: 提交**

```bash
git add app/knowledge/retrieval.py app/tools/knowledge.py tests/test_retrieval.py tests/test_tools.py
git commit -m "feat: 在线向量检索并替换 query_faq 的关键词查表

签名不变，只改 docstring；Milvus 故障降级为文案不抛异常。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: 对话挖知识 CLI

**Files:**
- Create: `jobs/mine_knowledge.py`
- Test: `tests/test_mining.py`

**Interfaces:**
- Consumes: `MinedQaStaging`、`KnowledgeChunk`
- Produces:
  - `normalize(text: str) -> str`
  - `async def dedupe_and_import(session) -> tuple[int, int]`（返回 (入库数, 丢弃数)）
  - CLI `uv run python -m jobs.mine_knowledge`

- [ ] **Step 1: 写失败的测试 `tests/test_mining.py`**

```python
from app.db.models import MinedQaStaging
from app.knowledge.dedupe import normalize
from jobs.mine_knowledge import dedupe_and_import


def test_normalize_strips_punct_and_space():
    assert normalize(" 邮费 是多少？ ") == "邮费是多少"


async def test_dedupe_within_batch(db_session):
    db_session.add_all([
        MinedQaStaging(question="邮费是多少？", answer="A", category="咨询"),
        MinedQaStaging(question=" 邮费是多少 ", answer="A", category="咨询"),
    ])
    await db_session.flush()
    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (1, 1)


async def test_dedupe_against_existing_chunks(db_session):
    from app.knowledge.chunk import Chunk
    from app.repositories import knowledge_chunk as repo

    await repo.upsert_chunks(db_session, [
        Chunk("faq:1", "咨询", "邮费是多少", "已有的答案。", "咨询", "faq"),
    ])
    db_session.add(MinedQaStaging(question="邮费是多少？", answer="新答案", category="咨询"))
    await db_session.flush()

    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (0, 1)


async def test_imported_rows_become_pending_chunks(db_session):
    db_session.add(MinedQaStaging(question="能开专票吗", answer="可以。", category="咨询"))
    await db_session.flush()
    await dedupe_and_import(db_session)

    from sqlalchemy import select
    from app.db.models import KnowledgeChunk

    rows = list((await db_session.execute(select(KnowledgeChunk))).scalars())
    assert len(rows) == 1
    assert rows[0].content_type == "mined"
    assert rows[0].status == "待向量化"

    staging = list((await db_session.execute(select(MinedQaStaging))).scalars())
    assert staging[0].status == "已入库"
```

> `normalize` 在 `app/knowledge/dedupe.py`，测试直接从那导入。

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_mining.py -v
```

- [ ] **Step 3: 实现**

`app/knowledge/dedupe.py`：

```python
import re

_PUNCT = re.compile(r"[\s，。！？；：、,.!?;:\"'“”‘’()（）\[\]【】]")


def normalize(text: str) -> str:
    """去空白与标点，转小写，用于精确去重。"""
    return _PUNCT.sub("", text).strip().lower()
```

`jobs/mine_knowledge.py`：

```python
"""从历史客服对话挖掘问答对。

用法：uv run python -m jobs.mine_knowledge
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.config import get_settings
from app.db.base import AsyncSessionLocal
from app.db.models import KnowledgeChunk, Message, MinedQaStaging
from app.knowledge.chunk import Chunk
from app.knowledge.dedupe import normalize
from app.providers import get_chat_model

EXTRACT_PROMPT = """你从客服对话里抽取可复用的问答知识。

只抽取**通用、可复用**的知识（政策、流程、规则）。
不要抽取一次性的个别订单信息（具体订单号、个人地址、单次物流）。
没有可抽取的内容就返回空数组。

输出 JSON 数组，每项形如：
{"question": "用户可能怎么问", "answer": "平台的标准答复", "category": "售后/售前/配送/发票"}

只输出 JSON，不要任何解释。"""


async def extract_from_conversation(text: str) -> list[dict]:
    model = get_chat_model(temperature=0)
    resp = await model.ainvoke([
        SystemMessage(EXTRACT_PROMPT),
        HumanMessage(text),
    ])
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        d for d in data
        if isinstance(d, dict) and d.get("question") and d.get("answer")
    ]


async def dedupe_and_import(session) -> tuple[int, int]:
    """把 status='待入库' 的暂存问答去重后写入 knowledge_chunks。"""
    staged = list((await session.execute(
        select(MinedQaStaging).where(MinedQaStaging.status == "待入库").order_by(MinedQaStaging.id)
    )).scalars())

    existing = list((await session.execute(select(KnowledgeChunk.questions))).scalars())
    seen = {normalize(q) for q in existing}

    imported = dropped = 0
    for row in staged:
        key = normalize(row.question)
        if key in seen:
            row.status = "已丢弃"
            dropped += 1
            continue
        seen.add(key)
        # ★ content_hash 必须在这里就算出来。不能留空等 build_knowledge 补：
        # dedupe 会把 staging 行标为「已入库」，而 build 的 load_mined_chunks 只读
        # 「待入库」——那个补 hash 的链路根本走不到，留空会永久是空串。
        chunk = Chunk(
            chunk_id=f"mined:{row.id}",
            category=row.category,
            questions=row.question,
            answer=row.answer,
            section_path=row.category,
            content_type="mined",
        )
        session.add(
            KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                category=chunk.category,
                questions=chunk.questions,
                answer=chunk.answer,
                section_path=chunk.section_path,
                content_type=chunk.content_type,
                status="待向量化",
                content_hash=chunk.content_hash,
            )
        )
        row.status = "已入库"
        imported += 1

    await session.flush()
    return imported, dropped


async def main() -> None:
    async with AsyncSessionLocal() as session:
        already = set((await session.execute(
            select(MinedQaStaging.conversation_id)
        )).scalars())
        rows = list((await session.execute(
            select(Message.conversation_id, Message.role, Message.content)
            .where(Message.role.in_(["user", "assistant"]))
            .order_by(Message.conversation_id, Message.id)
        )).all())

        by_conv: dict[int, list[str]] = {}
        for cid, role, content in rows:
            if cid in already or not content:
                continue
            by_conv.setdefault(cid, []).append(f"{role}: {content}")

        batch_size = 10
        conv_ids = list(by_conv)
        for i in range(0, len(conv_ids), batch_size):
            batch = conv_ids[i : i + batch_size]
            text = "\n\n---\n\n".join("\n".join(by_conv[c]) for c in batch)
            pairs = await extract_from_conversation(text)
            for p in pairs:
                session.add(MinedQaStaging(
                    question=p["question"][:512],
                    answer=p["answer"],
                    category=p.get("category", "其他")[:64],
                    conversation_id=batch[0],
                ))
            await session.commit()

        imported, dropped = await dedupe_and_import(session)
        await session.commit()
        print(f"挖掘完成：入库 {imported}，去重丢弃 {dropped}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **注意**：`content_hash=""` 会让 Task 6 的 `upsert_chunks` 在下次建库时发现 hash 变化并重置为待向量化——这正是我们想要的。但 `build_knowledge` 的 `load_mined_chunks` 会构造带正确 hash 的 `Chunk`，从而覆盖它。**实现时确认这条链路成立**。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_mining.py -v
```

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/dedupe.py jobs/mine_knowledge.py tests/test_mining.py
git commit -m "feat: 对话挖知识 CLI（暂存表 + 规范化精确去重）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: 检索评估集

**Files:**
- Create: `evals/kb_cases.yaml`, `evals/run_kb_evals.py`, `README.md`

**Interfaces:**
- Consumes: `retrieval.search`
- Produces: 控制台通过率表格

- [ ] **Step 1: 写 `evals/kb_cases.yaml`（20 条）**

```yaml
- {query: 邮费是多少, expect_section: 运费说明}
- {query: 退货运费谁承担, expect_section: 运费说明}
- {query: 运费怎么算, expect_section: 运费说明}
- {query: 满多少包邮, expect_section: 运费说明}
- {query: 退货政策是什么, expect_section: 退货政策}
- {query: 七天无理由怎么算, expect_section: 退货政策}
- {query: 定制商品能退吗, expect_section: 退货政策}
- {query: 怎么换货, expect_section: 换货流程}
- {query: 换货要多久, expect_section: 换货流程}
- {query: 多久发货, expect_section: 发货时效}
- {query: 什么时候能发, expect_section: 发货时效}
- {query: 发什么快递, expect_section: 发什么快递}
- {query: 能指定快递吗, expect_section: 发什么快递}
- {query: 怎么开发票, expect_section: 怎么开发票}
- {query: 发票多久能开出来, expect_section: 怎么开发票}
- {query: 发票抬头写错了, expect_section: 发票能改抬头吗}
- {query: 鞋子尺码怎么选, expect_section: 鞋子尺码怎么选}
- {query: 衣服买大了怎么办, expect_section: 衣服尺码怎么选}
- {query: 价保怎么申请, expect_section: 价保规则}
- {query: 限购几件, expect_section: 限购}
```

- [ ] **Step 2: 写 `evals/run_kb_evals.py`**

```python
"""ch03 检索质量评估集。

用法：uv run python -m evals.run_kb_evals
"""

import asyncio
import sys
from pathlib import Path

import yaml

from app.config import get_settings
from app.db.base import AsyncSessionLocal
from app.knowledge.retrieval import search

ROOT = Path(__file__).parent


async def main() -> int:
    cases = yaml.safe_load((ROOT / "kb_cases.yaml").read_text(encoding="utf-8"))
    top_k = get_settings().retrieval_top_k
    ok = 0

    async with AsyncSessionLocal() as session:
        for c in cases:
            hits = await search(session, c["query"], top_k=top_k)
            sections = [h.questions for h in hits]
            hit = c["expect_section"] in sections
            ok += hit
            print(f"  [{'PASS' if hit else 'FAIL'}] {c['query']:<14} → {sections}")

    total = len(cases)
    print(f"\nTop-{top_k} 命中率：{ok}/{total} = {ok / total:.1%}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 3: 跑评估**

```bash
uv run python -m evals.run_kb_evals
```
Expected: 20/20。**首次不 100% 是正常的**——若失败，先看是"样例文档内容不够丰富"还是"切分把某节切碎了"，**改文档或切分参数，不要改用例**。

- [ ] **Step 4: README 加命令**

```markdown
uv run python -m evals.run_kb_evals     # ch03 检索质量评估（需先建库）
```

- [ ] **Step 5: 提交**

```bash
git add evals/kb_cases.yaml evals/run_kb_evals.py README.md
git commit -m "test: ch03 检索质量评估集（20 条标注查询）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 12: 验收、README 与收尾

**Files:**
- Modify: `README.md`, `dev-notes/ch03.md`

- [ ] **Step 1: 验收 1 —— 「邮费是多少」**

```bash
docker compose up -d
uv run python -m jobs.build_knowledge
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
sleep 6

curl -sN -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"邮费是多少"}' | head -40
```

Expected:
- 出现 `event: tool` 帧，`name` 为 `query_faq`
- 最终回答里包含「运费」相关的内容（如"往返运费由平台承担"或"寄回运费需您自行承担"）
- 把原始输出贴进 `dev-notes/ch03.md`

- [ ] **Step 2: 验收 2 —— 中断后重跑**

```bash
uv run python - <<'PY'
import asyncio
from app.db.base import AsyncSessionLocal
from app.knowledge.builder import build
from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.sources.mined import load_mined_chunks
from pathlib import Path

async def main():
    async with AsyncSessionLocal() as s:
        chunks = (load_markdown_dir(Path("knowledge"), max_chars=400, overlap_chars=80)
                  + await load_faq_chunks(s) + await load_mined_chunks(s))
        try:
            await build(s, chunks, fail_after=5)
        except RuntimeError as e:
            print("已模拟中断：", e)

asyncio.run(main())
PY

# 重跑正常建库
uv run python -m jobs.build_knowledge

# 断言：MySQL 全部已向量化且与 Milvus 条数一致
uv run python - <<'PY'
import asyncio
from sqlalchemy import text
from app.db.base import AsyncSessionLocal
from app.knowledge.vector_store import VectorStore

async def main():
    async with AsyncSessionLocal() as s:
        total = (await s.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))).scalar_one()
        done = (await s.execute(text(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE status='已向量化'"))).scalar_one()
    milvus = VectorStore().count()
    print(f"MySQL 总数={total} 已向量化={done} Milvus={milvus}")
    assert done == total, "有块未补齐"
    assert milvus == total, f"Milvus 条数不一致（可能重复）：{milvus} != {total}"
    print("✅ 验收 2 通过：补齐且无重复")

asyncio.run(main())
PY
```

- [ ] **Step 3: 全量测试**

```bash
uv run pytest -q
```
把数字记进 dev-notes。

- [ ] **Step 4: README 补 ch03 段落**

```markdown
## ch03 · 知识库与向量检索

### 启动

```bash
docker compose up -d                  # MySQL + Milvus Standalone
uv run python -m scripts.seed         # 灌 FAQ 种子 + 建测试库
uv run python -m jobs.build_knowledge # 切分 + 向量化
uv run python -m evals.run_kb_evals   # 检索质量评估
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 知识来源

| 来源 | 内容 | content_type |
|---|---|---|
| `knowledge/*.md` | 政策手册、商品 FAQ、平台规则 | `policy` |
| `faq` 表 | 存量 12 条 FAQ | `faq` |
| `mined_qa_staging` | 从历史对话挖掘的问答对 | `mined` |

### 定时任务

```bash
# 每天凌晨 3 点挖一次历史对话
0 3 * * * cd /home/zzx/custermerHelp && uv run python -m jobs.mine_knowledge && uv run python -m jobs.build_knowledge
```

### 已知限制

- 只用 dense 向量单路，无关键词/混合/重排
- 文档编辑后旧 chunk 的向量不自动清理（孤儿向量）
- 检索质量依赖知识内容本身是否写完整（见 spec §3.2 的实测对比）
```

- [ ] **Step 5: 停服务、提交**

```bash
# 停掉本任务启动的 uvicorn（按 PID）
kill $(pgrep -f "bin/uvicorn app.main:app") 2>/dev/null || true

git add README.md dev-notes/ch03.md
git commit -m "docs: ch03 验收记录与 README

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

- [ ] **Step 6: 请求 code review**

```
使用 superpowers:requesting-code-review 对 ch03 全量改动做评审
```

把评审结论（含被驳回的建议）记进 `dev-notes/ch03.md`。

---

## 自查记录

**1. Spec 覆盖**

| Spec 节 | 覆盖任务 |
|---|---|
| §3 前置条件与实测结论 | Task 0（验证）、Global Constraints |
| §4 架构与模块划分 | Task 1–10 |
| §5 切分规则 | Task 3 |
| §5.5 三源字段填充 | Task 4 |
| §6 数据模型（MySQL + Milvus） | Task 2、Task 5 |
| §7 双写状态机与断点续跑 | Task 6、Task 7 |
| §8 对话挖知识 | Task 10 |
| §9 在线检索与 query_faq | Task 9 |
| §10 配置 | Task 1 |
| §11 错误处理 | Task 5（Milvus 包装）、Task 9（降级） |
| §12 测试与评估 | Task 3–11 |
| §13 验收对应 | Task 12 |
| §14 已知限制 | Task 12（README） |
| §15 未决项 | Task 0 |

**2. 占位符扫描**：无 TBD / TODO / "类似 Task N"。

**3. 类型一致性**

- `Chunk` 字段在 Task 3 定义，Task 4/6/7/9 消费，一致
- `split_markdown(text, *, source_key, content_type, max_chars, overlap_chars)` 在 Task 3 定义、Task 4 调用，一致
- `build(session, chunks, *, store, embed, fail_after)` 在 Task 7 定义、Task 8/12 调用，一致
- `search(session, query, top_k, *, store, embed)` 在 Task 9 定义、Task 9 的 `query_faq` 与 Task 11 调用，一致
- `mark_vectorized(session, chunk_ids, vector_ids)` 在 Task 6 定义、Task 7 调用，一致
- `VectorStore.ensure_collection / upsert / search / count` 在 Task 5 定义，Task 7/8/12 使用，一致

**4. Review Focus 的测试归属**

| 风险 | 归属任务 |
|---|---|
| 超长无标点文本 | Task 3 `test_text_without_punctuation_terminates` |
| 代码块含 `#` / `\|` | Task 3 `test_code_fence_is_never_split` |
| Milvus 集合为空 | Task 5 `test_empty_collection_returns_empty_list` + Task 9 `test_empty_hits_returns_empty_list` |
| 同一文档连续建库两次 | Task 7 `test_rerun_is_idempotent` |
| Embedding 维度不符 | Task 5 `test_embed_texts_rejects_wrong_dim` |

**5. 写计划时自查出的三处返工（已就地修正）**

- **Task 10 的测试导入写错了模块名**（`app.jobs_helpers`）——那是笔误，改成 `app.knowledge.dedupe`。
- **`dedupe_and_import` 原打算把 `content_hash` 留空、等 `build_knowledge` 补**——但 dedupe 会把 staging 标为「已入库」，而 `load_mined_chunks` 只读「待入库」，**那条补 hash 的链路根本走不到**，hash 会永久是空串，之后内容变化检测失效。改为在 dedupe 时就构造成 `Chunk` 并取 `chunk.content_hash`。
- **Task 7 的 builder 测试原本用 `db_session`**——但 `build` 为了真实的重跑语义必须 `commit()`，而 `db_session` 是"跑完回滚"的；commit 会让回滚失效并把测试数据留在库里，进而干扰 `test_knowledge_repo` 的计数断言。改为**独立 session + 唯一 id 前缀 + 事后只删自己造的行**；`test_counts` 同时改为**增量断言**，不再依赖库里的绝对条数。
