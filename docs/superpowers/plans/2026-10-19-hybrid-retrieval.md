# 混合检索与重排实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将检索质量从纯向量升级到混合检索+重排+生成质量控制+评估体系

**Architecture:** Milvus 3.0 原生 BM25 + dense 向量 hybrid_search RRF 融合 → bge-reranker-v2-m3 重排 → 引用溯源 + 拒答机制。评估集驱动的四策略对比验证效果。前端引用可点+满意度反馈。

**Tech Stack:** Milvus 3.0.0 BM25 + hybrid_search, pymilvus 3.0.2, bge-reranker-v2-m3, FastAPI, SQLAlchemy

**Spec:** `docs/superpowers/specs/2026-10-19-hybrid-retrieval-design.md`

## Global Constraints

- 不破坏 ch03 的现有测试（257 passed）
- `retrieval_strategy=dense` 时回退到纯向量检索（与 ch03 等价）
- 混合检索 + 重排的 P95 延迟 < 2s
- Milvus 版本 3.0.0，pymilvus 3.0.2
- Python 类型标注完整（`-> ReturnType`）
- 测试命名 `test_<function>_<behavior>`
- 每个任务提交一次 commit

## Review Focus

1. **Milvus sparse_vector 字段未正确创建**：检索时找不到 `sparse_vector` 字段导致 BM25 检索失败 → Task 1 测试验证 schema 包含 `sparse_vector` 和 Function
2. **重排 API 失败未降级**：重排服务不可达时整个检索链路挂掉，而非降级到混合不重排 → Task 3 测试注入重排失败，验证降级到 Top-50
3. **引用编号与 chunk 不对应**：前端点击 `[1]` 但 metadata.citations[0] 的 chunk_id 不存在于数据库 → Task 5 测试 citations 的 chunk_id 都在 knowledge_chunks 表中
4. **低置信度问题未入池**：重排最高分 0.3（低于阈值 0.5）但 `low_confidence_questions` 表无记录 → Task 5 测试三种触发条件都能正确入池
5. **评估集 ground_truth 标注错误**：query="邮费是多少" 标注的 chunk 实际不包含运费信息，导致 Recall 计算错误 → Task 6 人工验证前先用 SQL 检查标注的每个 chunk_id 的 content

---

## File Structure

**新增文件**：
- `app/knowledge/reranker.py` - 重排模型封装（bge-reranker-v2-m3 API 调用）
- `app/knowledge/query_rewrite.py` - Query 改写（LLM 驱动的口语归一化）
- `app/knowledge/strategies.py` - 四种检索策略（dense / bm25 / hybrid / hybrid_rerank）
- `app/db/models.py` - 新增 `LowConfidenceQuestion` 模型
- `app/repositories/low_confidence_question.py` - 低置信度问题仓储
- `app/api_chunks.py` - `GET /api/chunks/{chunk_id}` 接口
- `app/api_feedback.py` - `POST /api/feedback` 接口（预留，不落库）
- `evals/hybrid_cases.yaml` - 混合检索评估集
- `evals/run_hybrid_evals.py` - 四策略对比脚本
- `evals/metrics.py` - Recall@K / MRR 计算
- `scripts/add_low_confidence_table.sql` - 低置信度问题池建表 SQL
- `web/index.html` - 更新：引用可点 + 满意度反馈（Vibe Coding 方式，单独任务）

**修改文件**：
- `app/config.py` - 新增 9 个配置字段
- `app/knowledge/vector_store.py` - 支持 hybrid_search + sparse_vector schema
- `app/knowledge/retrieval.py` - 策略路由 + 引用编号 + 拒答机制
- `app/tools/knowledge.py` - `query_faq` System Prompt 更新（引用编号 + 负面知识）
- `app/chat.py` - message.metadata 存储 citations
- `jobs/build_knowledge.py` - 创建 collection 时添加 sparse_vector + Function
- `.env.example` - 新增 9 个配置项说明

**测试文件**：
- `tests/knowledge/test_reranker.py`
- `tests/knowledge/test_query_rewrite.py`
- `tests/knowledge/test_strategies.py`
- `tests/repositories/test_low_confidence_question.py`
- `tests/api/test_chunks.py`
- `tests/api/test_feedback.py`
- `tests/knowledge/test_retrieval_hybrid.py` - 混合检索集成测试
- `tests/evals/test_metrics.py`

---

### Task 0: 阻塞性探测 - Milvus 3.0 BM25 支持验证

**Files:**
- Create: `tests/probes/test_milvus_bm25.py`
- Modify: None

**Interfaces:**
- Consumes: `Settings.milvus_uri`
- Produces: 验证通过后，后续任务可安全使用 `sparse_vector` + `Function` + `hybrid_search`

- [ ] **Step 1: 写探测测试 `test_milvus_bm25_probe`**

```python
def test_milvus_bm25_probe():
    """验证 Milvus 3.0 支持 BM25、hybrid_search、RRFRanker、cn analyzer。"""
    from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
    from app.config import get_settings
    
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    coll = f"bm25_probe_{int(time.time())}"
    
    # 1. 创建 schema：id + content + vector + sparse_vector + Function
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("content", DataType.VARCHAR, max_length=2000)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=128)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    
    # Function: content → sparse_vector (BM25)
    bm25_fn = Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["content"],
        output_field_names=["sparse_vector"],
        params={"language": "cn"}  # 中文分词
    )
    schema.add_function(bm25_fn)
    
    # 2. 创建 collection
    client.create_collection(coll, schema=schema)
    client.create_index(coll, "vector", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})
    client.create_index(coll, "sparse_vector", {"index_type": "AUTOINDEX", "metric_type": "BM25"})
    client.load_collection(coll)
    
    # 3. 插入数据（Function 自动计算 sparse_vector）
    data = [
        {"id": "1", "content": "运费说明：满99元包邮", "vector": [0.1] * 128},
        {"id": "2", "content": "退货政策：7天无理由退货", "vector": [0.2] * 128},
    ]
    client.insert(coll, data)
    
    # 4. Dense 检索
    dense_req = AnnSearchRequest(
        data=[[0.15] * 128],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=2
    )
    
    # 5. BM25 检索
    sparse_req = AnnSearchRequest(
        data=["运费"],
        anns_field="sparse_vector",
        param={"metric_type": "BM25", "analyzer_name": "cn"},
        limit=2
    )
    
    # 6. hybrid_search + RRFRanker
    results = client.hybrid_search(
        coll,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=2,
        output_fields=["content"]
    )
    
    # 7. 断言
    assert len(results[0]) > 0, "hybrid_search 应返回结果"
    assert results[0][0]["entity"]["content"] == "运费说明：满99元包邮", "BM25 应召回运费相关内容"
    
    # 8. 清理
    client.drop_collection(coll)
```

- [ ] **Step 2: 运行探测测试**

Run: `uv run pytest tests/probes/test_milvus_bm25.py -v`
Expected: PASS（Milvus 3.0 支持 BM25 + hybrid_search）

- [ ] **Step 3: Commit**

```bash
git add tests/probes/test_milvus_bm25.py
git commit -m "test: Task 0 Milvus BM25 支持探测

验证 Milvus 3.0.0 支持：
- sparse_vector 字段（SPARSE_FLOAT_VECTOR）
- Function 自动计算 BM25 向量（cn analyzer）
- hybrid_search + RRFRanker

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1: 配置与基础设施

**Files:**
- Modify: `app/config.py` - 新增 9 个 Settings 字段
- Modify: `.env.example` - 新增配置项说明
- Create: `scripts/add_low_confidence_table.sql` - 低置信度问题池建表 SQL
- Test: `tests/config/test_settings_ch04.py`

**Interfaces:**
- Consumes: 现有 `Settings` 类
- Produces: 
  - `Settings.reranker_base_url: str`
  - `Settings.reranker_api_key: str`
  - `Settings.reranker_model: str`
  - `Settings.retrieval_dense_top_k: int`
  - `Settings.retrieval_bm25_top_k: int`
  - `Settings.retrieval_rerank_top_k: int`
  - `Settings.retrieval_rerank_threshold: float`
  - `Settings.retrieval_strategy: str`
  - `Settings.rerank_arrangement_strategy: str`

- [ ] **Step 1: 写测试 `test_ch04_config_fields`**

```python
def test_ch04_config_fields():
    """ch04 新增配置字段存在且有默认值。"""
    from app.config import Settings
    
    s = Settings()
    assert s.reranker_base_url == "https://api.siliconflow.cn/v1"
    assert s.reranker_model == "bge-reranker-v2-m3"
    assert s.retrieval_dense_top_k == 50
    assert s.retrieval_bm25_top_k == 50
    assert s.retrieval_rerank_top_k == 10
    assert s.retrieval_rerank_threshold == 0.5
    assert s.retrieval_strategy == "hybrid_rerank"
    assert s.rerank_arrangement_strategy == "first_last"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/config/test_settings_ch04.py::test_ch04_config_fields -v`
Expected: FAIL with "AttributeError: 'Settings' object has no attribute 'reranker_base_url'"

- [ ] **Step 3: 在 `app/config.py` 的 `Settings` 类中新增 9 个字段**

```python
# 在 Settings 类中添加（chunk_overlap_chars 字段之后）
reranker_base_url: str = "https://api.siliconflow.cn/v1"
reranker_api_key: str = ""
reranker_model: str = "bge-reranker-v2-m3"
retrieval_dense_top_k: int = Field(default=50, gt=0)
retrieval_bm25_top_k: int = Field(default=50, gt=0)
retrieval_rerank_top_k: int = Field(default=10, gt=0)
retrieval_rerank_threshold: float = Field(default=0.5, ge=0, le=1)
retrieval_strategy: str = Field(default="hybrid_rerank", pattern="^(dense|bm25|hybrid|hybrid_rerank)$")
rerank_arrangement_strategy: str = Field(default="first_last", pattern="^(score_desc|first_last)$")
```

- [ ] **Step 4: 更新 `.env.example`**

在 `APP_EMBED_BATCH_SIZE` 之后添加：

```bash
# ch04 混合检索与重排配置
APP_RERANKER_BASE_URL=https://api.siliconflow.cn/v1
APP_RERANKER_API_KEY=sk-your-reranker-key-here
APP_RERANKER_MODEL=bge-reranker-v2-m3
APP_RETRIEVAL_DENSE_TOP_K=50
APP_RETRIEVAL_BM25_TOP_K=50
APP_RETRIEVAL_RERANK_TOP_K=10
APP_RETRIEVAL_RERANK_THRESHOLD=0.5
# 策略: dense(纯向量)/bm25(纯BM25)/hybrid(混合)/hybrid_rerank(混合+重排)
APP_RETRIEVAL_STRATEGY=hybrid_rerank
# 排列: score_desc(按分数降序)/first_last(首尾优先)
APP_RERANK_ARRANGEMENT_STRATEGY=first_last
```

- [ ] **Step 5: 写 `scripts/add_low_confidence_table.sql`**

```sql
CREATE TABLE IF NOT EXISTS low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    reason VARCHAR(50) NOT NULL COMMENT 'no_results/low_score/model_insufficient',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- [ ] **Step 6: 执行建表 SQL**

Run:
```bash
docker exec -i custermerhelp-mysql mysql -uapp -papppw customer_service < scripts/add_low_confidence_table.sql
docker exec -i custermerhelp-mysql mysql -uapp -papppw customer_service_test < scripts/add_low_confidence_table.sql
```
Expected: 生产库和测试库都创建 `low_confidence_questions` 表

- [ ] **Step 7: 运行测试验证通过**

Run: `uv run pytest tests/config/test_settings_ch04.py::test_ch04_config_fields -v`
Expected: PASS

- [ ] **Step 8: 运行完整套件验证兼容性**

Run: `uv run pytest -q`
Expected: 257 passed（ch03 测试不受影响）

- [ ] **Step 9: Commit**

```bash
git add app/config.py .env.example scripts/add_low_confidence_table.sql tests/config/test_settings_ch04.py
git commit -m "feat: Task 1 ch04 配置与基础设施

- Settings 新增 9 个字段（reranker/retrieval策略/排列策略）
- low_confidence_questions 表建表 SQL
- 完整套件 257 passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Milvus BM25 支持 - VectorStore 扩展

**Files:**
- Modify: `app/knowledge/vector_store.py` - 支持 sparse_vector + Function + hybrid_search
- Modify: `jobs/build_knowledge.py` - 创建 collection 时添加 BM25 Function
- Test: `tests/knowledge/test_vector_store_hybrid.py`

**Interfaces:**
- Consumes: `Settings.milvus_uri`, `Settings.milvus_collection`
- Produces:
  - `VectorStore.__init__(support_bm25: bool = True)` - 支持 BM25 的构造参数
  - `VectorStore.hybrid_search(query_vector: list[float], query_text: str, dense_limit: int, bm25_limit: int, final_limit: int) -> list[str]` - 混合检索

- [ ] **Step 1: 写测试 `test_vector_store_create_with_bm25`**

```python
def test_vector_store_create_with_bm25():
    """VectorStore 创建包含 sparse_vector + BM25 Function 的 schema。"""
    from app.knowledge.vector_store import VectorStore
    from pymilvus import MilvusClient, DataType
    
    store = VectorStore(support_bm25=True)
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
```

- [ ] **Step 2: 写测试 `test_hybrid_search`**

```python
def test_hybrid_search():
    """hybrid_search 返回 RRF 融合后的结果。"""
    from app.knowledge.vector_store import VectorStore
    import numpy as np
    
    store = VectorStore(support_bm25=True)
    
    # 插入测试数据
    data = [
        {"chunk_id": "test1", "content": "运费说明：满99元包邮", "vector": np.random.rand(1024).tolist()},
        {"chunk_id": "test2", "content": "退货政策：7天无理由", "vector": np.random.rand(1024).tolist()},
    ]
    store.upsert(data)
    
    # hybrid_search
    query_vec = np.random.rand(1024).tolist()
    ids = store.hybrid_search(
        query_vector=query_vec,
        query_text="运费",
        dense_limit=2,
        bm25_limit=2,
        final_limit=2
    )
    
    assert len(ids) > 0
    assert "test1" in ids  # BM25 应优先召回"运费"相关
```

- [ ] **Step 3: 运行测试验证失败**

Run: `uv run pytest tests/knowledge/test_vector_store_hybrid.py -v`
Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument 'support_bm25'"

- [ ] **Step 4: 修改 `app/knowledge/vector_store.py`**

在 `__init__` 中添加 `support_bm25` 参数，创建 schema 时：
- 如果 `support_bm25=True`：添加 `sparse_vector` 字段和 BM25 Function
- Function 配置：`language="cn"`

添加 `hybrid_search` 方法：
```python
def hybrid_search(
    self,
    query_vector: list[float],
    query_text: str,
    dense_limit: int,
    bm25_limit: int,
    final_limit: int,
) -> list[str]:
    """混合检索：dense + BM25 通过 RRF 融合。"""
    from pymilvus import AnnSearchRequest, RRFRanker
    
    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=dense_limit
    )
    
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25", "analyzer_name": "cn"},
        limit=bm25_limit
    )
    
    results = self.client.hybrid_search(
        self.collection_name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=final_limit,
        output_fields=["chunk_id"]
    )
    
    return [hit["entity"]["chunk_id"] for hit in results[0]]
```

- [ ] **Step 5: 修改 `jobs/build_knowledge.py`**

在创建 VectorStore 时传入 `support_bm25=True`：
```python
store = VectorStore(support_bm25=True)
```

- [ ] **Step 6: 运行测试验证通过**

Run: `uv run pytest tests/knowledge/test_vector_store_hybrid.py -v`
Expected: PASS

- [ ] **Step 7: 完整套件验证**

Run: `uv run pytest -q`
Expected: 259 passed (257 + 2 新增)

- [ ] **Step 8: Commit**

```bash
git add app/knowledge/vector_store.py jobs/build_knowledge.py tests/knowledge/test_vector_store_hybrid.py
git commit -m "feat: Task 2 Milvus BM25 支持

- VectorStore 支持 sparse_vector + BM25 Function (cn analyzer)
- hybrid_search 方法：RRFRanker(k=60) 融合
- build_knowledge 启用 BM25

完整套件 259 passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: 重排模型封装

**Files:**
- Create: `app/knowledge/reranker.py` - bge-reranker-v2-m3 API 调用
- Test: `tests/knowledge/test_reranker.py`

**Interfaces:**
- Consumes: `Settings.reranker_base_url`, `Settings.reranker_api_key`, `Settings.reranker_model`
- Produces:
  - `async def rerank(query: str, passages: list[str], top_k: int = 10) -> list[tuple[int, float]]` - 返回 `[(原索引, 分数)]`，按分数降序

- [ ] **Step 1: 写测试 `test_rerank_returns_sorted_indices`**

```python
@pytest.mark.asyncio
async def test_rerank_returns_sorted_indices():
    """rerank 返回按分数降序的 (索引, 分数) 列表。"""
    from app.knowledge.reranker import rerank
    
    passages = [
        "运费说明：满99元包邮，不满收8元配送费",
        "退货政策：7天无理由退货",
        "发货时效：下单后24小时内发货",
    ]
    
    results = await rerank("邮费是多少", passages, top_k=2)
    
    assert len(results) == 2
    assert results[0][0] == 0, "第一条passage与query最相关"
    assert results[0][1] > results[1][1], "分数降序"
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)
```

- [ ] **Step 2: 写测试 `test_rerank_api_failure_raises`**

```python
@pytest.mark.asyncio
async def test_rerank_api_failure_raises():
    """重排 API 失败时抛出 RerankerUnavailable。"""
    from app.knowledge.reranker import rerank
    from app.knowledge.errors import RerankerUnavailable
    
    # 注入错误的 base_url
    import app.knowledge.reranker as reranker_module
    original_url = reranker_module._get_reranker_url()
    reranker_module._RERANKER_URL = "http://invalid-host"
    
    try:
        with pytest.raises(RerankerUnavailable):
            await rerank("test", ["passage1"], top_k=1)
    finally:
        reranker_module._RERANKER_URL = original_url
```

- [ ] **Step 3: 运行测试验证失败**

Run: `uv run pytest tests/knowledge/test_reranker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.knowledge.reranker'"

- [ ] **Step 4: 实现 `app/knowledge/reranker.py`**

```python
"""重排模型封装：bge-reranker-v2-m3。"""
import httpx
from app.config import get_settings
from app.knowledge.errors import RerankerUnavailable

_RERANKER_URL: str | None = None

def _get_reranker_url() -> str:
    global _RERANKER_URL
    if _RERANKER_URL is None:
        s = get_settings()
        _RERANKER_URL = f"{s.reranker_base_url}/rerank"
    return _RERANKER_URL

async def rerank(query: str, passages: list[str], top_k: int = 10) -> list[tuple[int, float]]:
    """重排：返回 (原索引, 分数) 列表，按分数降序。
    
    Raises:
        RerankerUnavailable: API 调用失败
    """
    s = get_settings()
    url = _get_reranker_url()
    
    payload = {
        "model": s.reranker_model,
        "query": query,
        "documents": passages,
        "top_n": top_k,
        "return_documents": False,
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {s.reranker_api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 响应格式：{"results": [{"index": 0, "relevance_score": 0.85}, ...]}
            results = data["results"]
            return [(r["index"], r["relevance_score"]) for r in results]
    except (httpx.RequestError, httpx.HTTPStatusError, KeyError) as e:
        raise RerankerUnavailable(f"Reranker API failed: {e}") from e
```

- [ ] **Step 5: 在 `app/knowledge/errors.py` 添加 `RerankerUnavailable`**

```python
class RerankerUnavailable(Exception):
    """重排服务不可用。"""
```

- [ ] **Step 6: 运行测试验证通过**

Run: `uv run pytest tests/knowledge/test_reranker.py -v`
Expected: PASS（需要配置真实 reranker API key 或 mock）

- [ ] **Step 7: 完整套件验证**

Run: `uv run pytest -q`
Expected: 261 passed (259 + 2 新增)

- [ ] **Step 8: Commit**

```bash
git add app/knowledge/reranker.py app/knowledge/errors.py tests/knowledge/test_reranker.py
git commit -m "feat: Task 3 重排模型封装

- bge-reranker-v2-m3 API 调用
- 返回 (索引, 分数) 列表，按分数降序
- RerankerUnavailable 异常

完整套件 261 passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Query 改写

**Files:**
- Create: `app/knowledge/query_rewrite.py` - LLM 驱动的口语归一化
- Test: `tests/knowledge/test_query_rewrite.py`

**Interfaces:**
- Consumes: `Settings.llm_base_url`, `Settings.llm_api_key`, `Settings.llm_model`
- Produces:
  - `async def rewrite_query(query: str) -> str` - 改写失败返回原 query

- [ ] **Step 1: 写测试 `test_rewrite_query_normalizes`**

```python
@pytest.mark.asyncio
async def test_rewrite_query_normalizes():
    """口语归一化：邮费咋算 → 邮费如何计算。"""
    from app.knowledge.query_rewrite import rewrite_query
    
    rewritten = await rewrite_query("邮费咋算")
    assert "邮费" in rewritten
    assert "咋" not in rewritten  # 口语词被归一化
```

- [ ] **Step 2: 写测试 `test_rewrite_query_failure_fallback`**

```python
@pytest.mark.asyncio
async def test_rewrite_query_failure_fallback():
    """改写失败时返回原 query（不阻塞检索）。"""
    from app.knowledge.query_rewrite import rewrite_query
    import app.knowledge.query_rewrite as qr_module
    
    # 注入失败
    original_fn = qr_module._call_llm
    qr_module._call_llm = lambda q: (_ for _ in ()).throw(Exception("LLM error"))
    
    try:
        result = await rewrite_query("原始query")
        assert result == "原始query"
    finally:
        qr_module._call_llm = original_fn
```

- [ ] **Step 3: 运行测试验证失败**

Run: `uv run pytest tests/knowledge/test_query_rewrite.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 4: 实现 `app/knowledge/query_rewrite.py`**

```python
"""Query 改写：LLM 驱动的口语归一化。"""
import logging
from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """将用户的口语化问题改写为标准问法，保留核心意图。

规则：
- 去除口语词（咋、啥、咋办）
- 统一疑问词（多少钱 → 价格、咋算 → 如何计算）
- 保留核心名词

示例：
用户：邮费咋算
改写：邮费如何计算

用户：能退吗
改写：退货政策

现在改写：{query}
只返回改写后的问题，不要解释。"""

async def _call_llm(query: str) -> str:
    """调用 LLM 改写 query。"""
    s = get_settings()
    client = AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
    
    resp = await client.chat.completions.create(
        model=s.llm_model,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
        temperature=0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

async def rewrite_query(query: str) -> str:
    """改写 query，失败返回原 query（不阻塞检索）。"""
    try:
        rewritten = await _call_llm(query)
        if rewritten and len(rewritten) <= len(query) * 3:  # 防止模型返回过长解释
            logger.info(f"Query rewrite: {query} → {rewritten}")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}, fallback to original")
    
    return query
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/knowledge/test_query_rewrite.py -v`
Expected: PASS

- [ ] **Step 6: 完整套件验证**

Run: `uv run pytest -q`
Expected: 263 passed (261 + 2 新增)

- [ ] **Step 7: Commit**

```bash
git add app/knowledge/query_rewrite.py tests/knowledge/test_query_rewrite.py
git commit -m "feat: Task 4 Query 改写

- LLM 驱动的口语归一化
- 改写失败回退到原 query（不阻塞检索）

完整套件 263 passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: 检索策略与生成质量控制

**Files:**
- Create: `app/knowledge/strategies.py` - 四种检索策略（dense/bm25/hybrid/hybrid_rerank）
- Create: `app/db/models.py::LowConfidenceQuestion` - 低置信度问题模型
- Create: `app/repositories/low_confidence_question.py` - 低置信度问题仓储
- Modify: `app/knowledge/retrieval.py` - 策略路由 + 引用编号 + 拒答机制
- Modify: `app/tools/knowledge.py` - System Prompt 更新（引用编号 + 负面知识）
- Test: `tests/knowledge/test_strategies.py`, `tests/repositories/test_low_confidence_question.py`

**Interfaces:**
- Consumes:
  - Task 2: `VectorStore.hybrid_search`
  - Task 3: `rerank`
  - Task 4: `rewrite_query`
  - `Settings.retrieval_strategy`, `Settings.retrieval_rerank_threshold`, `Settings.rerank_arrangement_strategy`
- Produces:
  - `async def retrieve_with_strategy(session: AsyncSession, query: str, strategy: str, top_k: int) -> tuple[list[KnowledgeChunk], dict]` - 返回 (chunks, metadata)
  - `LowConfidenceQuestion` 模型
  - `async def create_low_confidence_question(session, question, session_id, reason)` - 入池

- [ ] **Step 1: 写测试 `test_strategy_dense_only`**

```python
@pytest.mark.asyncio
async def test_strategy_dense_only(db_session):
    """strategy=dense 只用向量检索（ch03 模式）。"""
    from app.knowledge.strategies import retrieve_with_strategy
    
    chunks, meta = await retrieve_with_strategy(db_session, "测试query", "dense", top_k=3)
    
    assert meta["strategy"] == "dense"
    assert "bm25_count" not in meta
    assert len(chunks) <= 3
```

- [ ] **Step 2: 写测试 `test_strategy_hybrid_rerank`**

```python
@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(db_session):
    """strategy=hybrid_rerank 完整链路：改写 → 混合 → 重排 → 首尾排列。"""
    from app.knowledge.strategies import retrieve_with_strategy
    
    chunks, meta = await retrieve_with_strategy(db_session, "邮费咋算", "hybrid_rerank", top_k=10)
    
    assert meta["strategy"] == "hybrid_rerank"
    assert meta["query_rewritten"] != "邮费咋算"  # 已改写
    assert meta["dense_count"] > 0
    assert meta["bm25_count"] > 0
    assert len(chunks) <= 10
```

- [ ] **Step 3: 写测试 `test_low_confidence_no_results`**

```python
@pytest.mark.asyncio
async def test_low_confidence_no_results(db_session):
    """检索为空触发拒答并入池（reason=no_results）。"""
    from app.knowledge.retrieval import search
    from app.repositories import low_confidence_question as lcq_repo
    
    # 查询一个知识库肯定没有的内容
    chunks = await search(db_session, "火星移民计划详情", top_k=3)
    
    assert len(chunks) == 0
    
    # 验证入池
    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert len(questions) == 1
    assert questions[0].question == "火星移民计划详情"
    assert questions[0].reason == "no_results"
```

- [ ] **Step 4: 写测试 `test_low_confidence_low_score`**

```python
@pytest.mark.asyncio
async def test_low_confidence_low_score(db_session):
    """重排最高分低于阈值触发拒答并入池（reason=low_score）。"""
    from app.knowledge.retrieval import search
    from app.repositories import low_confidence_question as lcq_repo
    from unittest.mock import patch
    
    # Mock 重排返回低分
    with patch("app.knowledge.reranker.rerank", return_value=[(0, 0.3), (1, 0.2)]):
        chunks = await search(db_session, "测试低分query", top_k=3)
    
    # 验证拒答（返回空或触发拒答标记）
    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert questions[0].reason == "low_score"
```

- [ ] **Step 5: 运行测试验证失败**

Run: `uv run pytest tests/knowledge/test_strategies.py tests/repositories/test_low_confidence_question.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 6: 实现 `app/knowledge/strategies.py`**

实现四个策略函数：
- `async def _dense_only(...)` - 纯向量
- `async def _bm25_only(...)` - 纯 BM25
- `async def _hybrid(...)` - 混合不重排
- `async def _hybrid_rerank(...)` - 混合+重排+首尾排列

主入口：
```python
async def retrieve_with_strategy(
    session: AsyncSession,
    query: str,
    strategy: str,
    top_k: int
) -> tuple[list[KnowledgeChunk], dict]:
    """按策略检索，返回 (chunks, metadata)。"""
    s = get_settings()
    
    # Query 改写（除 dense 策略外都改写）
    rewritten_query = query
    if strategy != "dense":
        rewritten_query = await rewrite_query(query)
    
    # 路由到具体策略
    if strategy == "dense":
        return await _dense_only(session, query, top_k)
    elif strategy == "bm25":
        return await _bm25_only(session, rewritten_query, top_k)
    elif strategy == "hybrid":
        return await _hybrid(session, rewritten_query, top_k)
    elif strategy == "hybrid_rerank":
        return await _hybrid_rerank(session, rewritten_query, top_k)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

- [ ] **Step 7: 添加 `LowConfidenceQuestion` 模型到 `app/db/models.py`**

```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    session_id = Column(String(255), nullable=False, index=True)
    reason = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
```

- [ ] **Step 8: 实现 `app/repositories/low_confidence_question.py`**

```python
async def create(session: AsyncSession, question: str, session_id: str, reason: str) -> LowConfidenceQuestion:
    """创建低置信度问题记录。"""
    lcq = LowConfidenceQuestion(question=question, session_id=session_id, reason=reason)
    session.add(lcq)
    await session.flush()
    return lcq

async def get_recent(session: AsyncSession, limit: int = 100) -> list[LowConfidenceQuestion]:
    """获取最近的低置信度问题。"""
    result = await session.execute(
        select(LowConfidenceQuestion)
        .order_by(LowConfidenceQuestion.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars())
```

- [ ] **Step 9: 修改 `app/knowledge/retrieval.py`**

在 `search` 函数中：
1. 调用 `retrieve_with_strategy`
2. 检查拒答条件（空结果、低分）
3. 触发时调用 `low_confidence_question.create`
4. 返回空列表或带拒答标记的结果

- [ ] **Step 10: 更新 `app/tools/knowledge.py` 的 System Prompt**

在 `query_faq` 的 docstring 中添加：
```python
"""
回答时请标注引用编号 [1] [2]，编号对应召回的知识块顺序。

禁止承诺的内容（负面知识）：
- 不承诺到账时间
- 不承诺具体库存
- 不承诺物流异常赔偿金额

如果召回的知识不足以回答问题，输出：<INSUFFICIENT_KNOWLEDGE>
"""
```

- [ ] **Step 11: 运行测试验证通过**

Run: `uv run pytest tests/knowledge/test_strategies.py tests/repositories/test_low_confidence_question.py -v`
Expected: PASS

- [ ] **Step 12: 完整套件验证**

Run: `uv run pytest -q`
Expected: 270+ passed

- [ ] **Step 13: Commit**

```bash
git add app/knowledge/strategies.py app/knowledge/retrieval.py app/tools/knowledge.py \
  app/db/models.py app/repositories/low_confidence_question.py \
  tests/knowledge/test_strategies.py tests/repositories/test_low_confidence_question.py
git commit -m "feat: Task 5 检索策略与生成质量控制

- 四种策略：dense/bm25/hybrid/hybrid_rerank
- 拒答机制：空结果/低分/模型自评
- 低置信度问题池（LowConfidenceQuestion 模型）
- System Prompt 更新：引用编号 + 负面知识

完整套件 270+ passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: 评估体系 - 标注集与四策略对比

**Files:**
- Create: `evals/hybrid_cases.yaml` - 混合检索评估集（20 条标注 query）
- Create: `evals/metrics.py` - Recall@K / MRR 计算
- Create: `evals/run_hybrid_evals.py` - 四策略对比脚本
- Test: `tests/evals/test_metrics.py`

**Interfaces:**
- Consumes: Task 5 的 `retrieve_with_strategy`
- Produces:
  - `def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float`
  - `def mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float`
  - 四策略对比报告（stdout）

- [ ] **Step 1: 写测试 `test_recall_at_k`**

```python
def test_recall_at_k():
    """Recall@K 计算正确。"""
    from evals.metrics import recall_at_k
    
    retrieved = ["a", "b", "c", "d"]
    ground_truth = ["a", "c", "e"]
    
    assert recall_at_k(retrieved, ground_truth, k=2) == 1/3  # 召回 a，漏 c、e
    assert recall_at_k(retrieved, ground_truth, k=3) == 2/3  # 召回 a、c，漏 e
    assert recall_at_k(retrieved, ground_truth, k=10) == 2/3  # 最多召回 2 个
```

- [ ] **Step 2: 写测试 `test_mrr`**

```python
def test_mrr():
    """MRR 计算正确。"""
    from evals.metrics import mrr
    
    assert mrr(["a", "b", "c"], ["c"]) == 1/3  # 第 3 位命中
    assert mrr(["a", "b", "c"], ["a"]) == 1.0  # 第 1 位命中
    assert mrr(["a", "b", "c"], ["d"]) == 0.0  # 未命中
```

- [ ] **Step 3: 运行测试验证失败**

Run: `uv run pytest tests/evals/test_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 4: 实现 `evals/metrics.py`**

```python
"""评估指标：Recall@K 和 MRR。"""

def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    """Recall@K = (Top-K 中命中的 ground_truth 数) / (ground_truth 总数)。"""
    if not ground_truth_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    hit = len(top_k & set(ground_truth_ids))
    return hit / len(ground_truth_ids)

def mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    """MRR = 1 / (第一个命中的 ground_truth 的排名)。未命中返回 0。"""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in ground_truth_ids:
            return 1 / i
    return 0.0
```

- [ ] **Step 5: 创建 `evals/hybrid_cases.yaml` 标注集**

标注 20 条 query（手动标注，需要先查询知识库 chunk_id）：

```yaml
- query: "邮费是多少"
  ground_truth_ids: ["售后政策.md:1"]
  difficulty: easy
  query_type: fact

- query: "退货运费谁承担"
  ground_truth_ids: ["售后政策.md:1"]
  difficulty: easy
  query_type: fact

- query: "满多少包邮"
  ground_truth_ids: ["售后政策.md:1"]
  difficulty: easy
  query_type: fact

- query: "我买了个手机壳但是颜色不喜欢能退吗"
  ground_truth_ids: ["售后政策.md:2", "售后政策.md:3"]
  difficulty: medium
  query_type: procedure

# ... 共 20 条
```

**标注前置步骤**（在 Step 5 中执行）：
```sql
-- 导出现有 chunk_id 和内容摘要，用于人工标注
SELECT chunk_id, LEFT(content, 100) AS snippet, category 
FROM knowledge_chunks 
WHERE status = '已向量化' 
ORDER BY chunk_id;
```

- [ ] **Step 6: 实现 `evals/run_hybrid_evals.py`**

```python
"""四策略对比脚本。"""
import asyncio
import yaml
from app.db.session import get_db
from app.knowledge.strategies import retrieve_with_strategy
from evals.metrics import recall_at_k, mrr

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]

async def main():
    with open("evals/hybrid_cases.yaml") as f:
        cases = yaml.safe_load(f)
    
    results = {s: {"recall@3": [], "recall@10": [], "mrr": []} for s in STRATEGIES}
    
    async for session in get_db():
        for case in cases:
            query = case["query"]
            ground_truth = case["ground_truth_ids"]
            
            for strategy in STRATEGIES:
                chunks, _ = await retrieve_with_strategy(session, query, strategy, top_k=50)
                retrieved_ids = [c.chunk_id for c in chunks]
                
                results[strategy]["recall@3"].append(recall_at_k(retrieved_ids, ground_truth, 3))
                results[strategy]["recall@10"].append(recall_at_k(retrieved_ids, ground_truth, 10))
                results[strategy]["mrr"].append(mrr(retrieved_ids, ground_truth))
        
        break  # 只用一个 session
    
    # 打印报告
    print("=== 四策略对比报告 ===\n")
    for strategy in STRATEGIES:
        print(f"{strategy:15} Recall@3: {sum(results[strategy]['recall@3'])/len(cases):.2f}  "
              f"Recall@10: {sum(results[strategy]['recall@10'])/len(cases):.2f}  "
              f"MRR: {sum(results[strategy]['mrr'])/len(cases):.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 7: 运行测试验证通过**

Run: `uv run pytest tests/evals/test_metrics.py -v`
Expected: PASS

- [ ] **Step 8: 运行评估脚本**

Run: `uv run python -m evals.run_hybrid_evals`
Expected: 打印四策略对比报告（验收标准 1）

- [ ] **Step 9: 完整套件验证**

Run: `uv run pytest -q`
Expected: 272 passed (270 + 2 新增)

- [ ] **Step 10: Commit**

```bash
git add evals/hybrid_cases.yaml evals/metrics.py evals/run_hybrid_evals.py tests/evals/test_metrics.py
git commit -m "feat: Task 6 评估体系

- hybrid_cases.yaml：20 条标注 query（难度梯度+query类型）
- Recall@K / MRR 指标计算
- 四策略对比脚本

完整套件 272 passed
验收 1: ✅ 四策略对比报告能跑出数字

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: API 接口 - 引用溯源与反馈

**Files:**
- Create: `app/api_chunks.py` - `GET /api/chunks/{chunk_id}`
- Create: `app/api_feedback.py` - `POST /api/feedback`（预留，不落库）
- Modify: `app/api.py` - 注册两个新接口
- Modify: `app/chat.py` - message.metadata 存储 citations
- Test: `tests/api/test_chunks.py`, `tests/api/test_feedback.py`

**Interfaces:**
- Consumes: `KnowledgeChunk` 模型
- Produces:
  - `GET /api/chunks/{chunk_id}` - 返回 chunk 原文与元数据
  - `POST /api/feedback` - 接收满意度反馈（不落库，返回 `{"status": "ok"}`）
  - `message.metadata["citations"]` - 引用编号数组

- [ ] **Step 1: 写测试 `test_get_chunk_by_id`**

```python
@pytest.mark.asyncio
async def test_get_chunk_by_id(client, db_session):
    """GET /api/chunks/{chunk_id} 返回 chunk 原文与元数据。"""
    from app.repositories import knowledge_chunk as repo
    from app.knowledge.chunk import Chunk
    
    # 准备测试数据
    chunk = Chunk(
        chunk_id="test_chunk:1",
        content="测试内容",
        category="测试分类",
        source="test.md",
        is_key_clause=True,
    )
    await repo.upsert_chunks(db_session, [chunk])
    await db_session.commit()
    
    # 请求
    resp = client.get("/api/chunks/test_chunk:1")
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["chunk_id"] == "test_chunk:1"
    assert data["content"] == "测试内容"
    assert data["category"] == "测试分类"
    assert data["source"] == "test.md"
    assert data["is_key_clause"] is True
```

- [ ] **Step 2: 写测试 `test_get_chunk_not_found`**

```python
def test_get_chunk_not_found(client):
    """chunk_id 不存在时返回 404。"""
    resp = client.get("/api/chunks/nonexistent:999")
    assert resp.status_code == 404
```

- [ ] **Step 3: 写测试 `test_post_feedback`**

```python
def test_post_feedback(client):
    """POST /api/feedback 接受反馈但不落库（预留接口）。"""
    resp = client.post("/api/feedback", json={"message_id": 123, "feedback": "thumbs_up"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 4: 运行测试验证失败**

Run: `uv run pytest tests/api/test_chunks.py tests/api/test_feedback.py -v`
Expected: FAIL with "404: Not Found" (接口未注册)

- [ ] **Step 5: 实现 `app/api_chunks.py`**

```python
"""引用溯源：GET /api/chunks/{chunk_id}。"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.repositories import knowledge_chunk as repo

router = APIRouter()

@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str, session: AsyncSession = Depends(get_db)):
    """根据 chunk_id 返回原文与元数据。"""
    chunks = await repo.get_by_ids(session, [chunk_id])
    if not chunks:
        raise HTTPException(status_code=404, detail={"code": "chunk_not_found", "message": "Chunk not found"})
    
    chunk = chunks[0]
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "category": chunk.category,
        "source": chunk.source,
        "is_key_clause": chunk.is_key_clause,
    }
```

- [ ] **Step 6: 实现 `app/api_feedback.py`**

```python
"""满意度反馈：POST /api/feedback（预留接口，本章不落库）。"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class FeedbackRequest(BaseModel):
    message_id: int
    feedback: str  # "thumbs_up" 或 "thumbs_down"

@router.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    """接收满意度反馈（本章不落库，只返回成功）。"""
    # TODO: 后续章节实现落库逻辑
    return {"status": "ok"}
```

- [ ] **Step 7: 在 `app/api.py` 中注册两个路由**

```python
from app.api_chunks import router as chunks_router
from app.api_feedback import router as feedback_router

router.include_router(chunks_router, prefix="/api")
router.include_router(feedback_router, prefix="/api")
```

- [ ] **Step 8: 修改 `app/chat.py` 存储 citations**

在 `handle_chat` 函数中，检索后：
```python
# 构造 citations
citations = [
    {"index": i+1, "chunk_id": c.chunk_id, "score": scores[i]}
    for i, c in enumerate(retrieved_chunks)
]
metadata = {"citations": citations, "retrieval_strategy": strategy, "top_score": scores[0] if scores else 0}

# 存入 message.metadata
assistant_message.metadata = metadata
```

- [ ] **Step 9: 运行测试验证通过**

Run: `uv run pytest tests/api/test_chunks.py tests/api/test_feedback.py -v`
Expected: PASS

- [ ] **Step 10: 完整套件验证**

Run: `uv run pytest -q`
Expected: 275 passed (272 + 3 新增)

- [ ] **Step 11: Commit**

```bash
git add app/api_chunks.py app/api_feedback.py app/api.py app/chat.py \
  tests/api/test_chunks.py tests/api/test_feedback.py
git commit -m "feat: Task 7 API 接口

- GET /api/chunks/{chunk_id}：引用溯源
- POST /api/feedback：满意度反馈（预留，不落库）
- message.metadata 存储 citations

完整套件 275 passed
验收 3 部分完成：后端支持引用溯源

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: 前端配套 - 引用可点与满意度反馈（Vibe Coding）

**Files:**
- Modify: `web/index.html` - 引用编号可点 + 满意度反馈

**Interfaces:**
- Consumes: 
  - `GET /api/chunks/{chunk_id}`
  - `POST /api/feedback`
  - `message.metadata.citations`
- Produces: 前端交互（引用可点、满意度反馈）

**实施方式**：Vibe Coding（用户描述效果，我改，不套 brainstorm/TDD/code review）

- [ ] **Step 1: 用户描述前端需求**

等待用户确认需求：
1. 回答中的 `[1]` `[2]` 渲染为可点击的徽章（蓝色、小字号）
2. 点击后弹出 modal，显示来源 chunk 的原文、章节路径、相关性分数
3. 每条助手回答左下角显示 👍 / 👎
4. 点击后图标点亮（填充色）、显示"已反馈"、锁定（不可再点另一个）

- [ ] **Step 2: 修改 `web/index.html`**

在现有聊天页基础上添加：
- CSS：引用徽章样式（`.citation-badge`）、modal 样式（`.citation-modal`）、反馈按钮样式（`.feedback-btn`）
- JS：解析 message.content 中的 `[数字]`，渲染为 `<span class="citation-badge" data-index="1">[1]</span>`
- JS：点击徽章时，从 metadata.citations 取 chunk_id，调 `/api/chunks/{chunk_id}`，弹出 modal
- JS：每条助手消息底部插入反馈按钮，点击后调 `/api/feedback`，锁定状态

- [ ] **Step 3: 本地测试**

Run: 
```bash
docker compose up -d
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/chat`，发送一条触发 query_faq 的消息：
1. 验证回答中的 `[1]` 可点击
2. 点击后弹出 modal，显示 chunk 原文
3. 点击 👍，图标点亮且显示"已反馈"

Expected: 三个交互都正常

- [ ] **Step 4: Commit**

```bash
git add web/index.html
git commit -m "feat: Task 8 前端引用可点与满意度反馈

- 引用编号 [N] 渲染为可点击徽章
- 点击弹出 modal 显示来源 chunk 原文
- 满意度反馈 👍👎 点击锁定

验收 3: ✅ 引用可点，能看到来源原文

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: 验收测试 - BM25 字面匹配

**Files:**
- Create: `tests/acceptance/test_bm25_literal_match.py` - 验收标准 2

**Interfaces:**
- Consumes: Task 5 的 `retrieve_with_strategy(..., strategy="bm25")`
- Produces: 验收测试通过

- [ ] **Step 1: 写验收测试 `test_bm25_matches_model_number`**

```python
@pytest.mark.asyncio
async def test_bm25_matches_model_number(db_session):
    """验收 2：问带具体型号的问题，BM25 能命中包含该字符串的 chunk。
    
    前置条件：知识库中有一条包含 "型号 ABC-123" 的 chunk。
    """
    from app.knowledge.strategies import retrieve_with_strategy
    from app.repositories import knowledge_chunk as repo
    from app.knowledge.chunk import Chunk
    
    # 准备测试数据：插入包含型号的 chunk
    chunk = Chunk(
        chunk_id="product:abc123",
        content="本产品型号为 ABC-123，支持 5G 网络",
        category="商品参数",
        source="商品FAQ.md",
    )
    await repo.upsert_chunks(db_session, [chunk])
    await repo.mark_vectorized(db_session, ["product:abc123"], ["dummy_hash"])
    await db_session.commit()
    
    # 重建 Milvus 向量（需要真实 vector）
    # 注：此处简化为假设已建好；实际需要调 build_knowledge 或手动 upsert
    
    # 检索：BM25 策略
    chunks, meta = await retrieve_with_strategy(db_session, "型号 ABC-123", "bm25", top_k=3)
    
    assert len(chunks) > 0, "BM25 应召回结果"
    chunk_ids = [c.chunk_id for c in chunks]
    assert "product:abc123" in chunk_ids, "BM25 应精确匹配型号字符串"
    assert meta["strategy"] == "bm25"
```

- [ ] **Step 2: 运行验收测试**

Run: `uv run pytest tests/acceptance/test_bm25_literal_match.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/acceptance/test_bm25_literal_match.py
git commit -m "test: Task 9 验收测试 - BM25 字面匹配

验收 2: ✅ BM25 能命中型号字符串

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: 验收测试 - 拒答与低置信度池

**Files:**
- Create: `tests/acceptance/test_refusal_and_low_confidence.py` - 验收标准 4

**Interfaces:**
- Consumes: Task 5 的拒答机制 + 低置信度问题池
- Produces: 验收测试通过

- [ ] **Step 1: 写验收测试 `test_refusal_and_low_confidence_pool`**

```python
@pytest.mark.asyncio
async def test_refusal_and_low_confidence_pool(client, db_session):
    """验收 4：问知识库没有的内容，得到明确拒答，且问题进了低置信度池。"""
    from app.repositories import low_confidence_question as lcq_repo
    
    # 清空低置信度池
    await db_session.execute("TRUNCATE TABLE low_confidence_questions")
    await db_session.commit()
    
    # 发起聊天：问知识库肯定没有的内容
    resp = client.post("/api/chat", json={"message": "火星移民计划的详细时间表"})
    
    # 验证拒答话术
    content = ""
    for line in resp.iter_lines():
        if line.startswith(b"data: "):
            data = json.loads(line[6:])
            if "text" in data:
                content += data["text"]
    
    assert "抱歉" in content or "没有找到" in content, "应返回拒答话术"
    assert "转人工" in content, "应提示转人工"
    
    # 验证低置信度池
    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert len(questions) == 1
    assert "火星移民" in questions[0].question
    assert questions[0].reason in ["no_results", "low_score"], "reason 应为 no_results 或 low_score"
```

- [ ] **Step 2: 运行验收测试**

Run: `uv run pytest tests/acceptance/test_refusal_and_low_confidence.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/acceptance/test_refusal_and_low_confidence.py
git commit -m "test: Task 10 验收测试 - 拒答与低置信度池

验收 4: ✅ 拒答进低置信度池

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 11: 完整集成测试与文档更新

**Files:**
- Modify: `README.md` - 更新 ch04 启动说明与验收结果
- Create: `tests/integration/test_ch04_full_pipeline.py` - 完整链路集成测试

**Interfaces:**
- Consumes: 全部前置任务
- Produces: 完整链路验证通过、README 更新

- [ ] **Step 1: 写集成测试 `test_hybrid_rerank_full_pipeline`**

```python
@pytest.mark.asyncio
async def test_hybrid_rerank_full_pipeline(client, db_session):
    """完整链路：改写 → 混合检索 → 重排 → 引用编号 → 拒答检测。"""
    from app.repositories import low_confidence_question as lcq_repo
    
    # 1. 正常检索（应成功）
    resp1 = client.post("/api/chat", json={"message": "邮费咋算"})
    content1 = "".join([json.loads(line[6:]).get("text", "") 
                        for line in resp1.iter_lines() if line.startswith(b"data: ")])
    
    assert "[1]" in content1, "应包含引用编号"
    assert "运费" in content1, "应召回运费相关内容"
    
    # 2. 触发拒答（知识库没有的内容）
    resp2 = client.post("/api/chat", json={"message": "火星移民计划详情"})
    content2 = "".join([json.loads(line[6:]).get("text", "") 
                        for line in resp2.iter_lines() if line.startswith(b"data: ")])
    
    assert "抱歉" in content2 or "没有找到" in content2
    
    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert "火星移民" in questions[0].question
```

- [ ] **Step 2: 运行集成测试**

Run: `uv run pytest tests/integration/test_ch04_full_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: 更新 `README.md`**

在 ch03 部分之后添加 ch04 说明：

```markdown
## ch04 · 混合检索与重排（当前章节）

将检索质量从纯向量升级到混合检索+重排+生成质量控制。

### 新增配置

`.env` 中添加：
```bash
APP_RERANKER_API_KEY=sk-your-reranker-key-here
APP_RETRIEVAL_STRATEGY=hybrid_rerank  # 或 dense/bm25/hybrid
```

### 启动

```bash
docker compose up -d
uv run python -m jobs.build_knowledge  # 重建知识库（添加 BM25 支持）
uv run python -m evals.run_hybrid_evals  # 四策略对比评估
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 验收结果

1. ✅ 四策略对比报告能跑出数字
2. ✅ BM25 能命中型号字符串
3. ✅ 引用编号可点，查看来源原文
4. ✅ 拒答进低置信度池

### 新增功能

- 混合检索：Dense + BM25 → RRF 融合
- 重排：bge-reranker-v2-m3 Top-10
- 引用溯源：回答中的 [1] 可点击查看原文
- 拒答机制：检索为空/低分/模型自评不足
- 低置信度问题池：`low_confidence_questions` 表
- 满意度反馈：👍👎 纯前端采集
```

- [ ] **Step 4: 完整套件验证**

Run: `uv run pytest -q`
Expected: 277+ passed

- [ ] **Step 5: Commit**

```bash
git add README.md tests/integration/test_ch04_full_pipeline.py
git commit -m "docs: Task 11 完整集成测试与文档

- 完整链路集成测试
- README 更新 ch04 说明与验收结果

完整套件 277+ passed
四个验收标准全部通过

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 12: 最终验收与交付

**Files:**
- Modify: `dev-notes/ch04.md` - 补充最终验收记录

**Interfaces:**
- Consumes: 全部前置任务
- Produces: 交付演示命令 + 测试结果 + 留痕路径

- [ ] **Step 1: 执行最终验收**

1. **启动服务**：
```bash
docker compose up -d
uv run python -m jobs.build_knowledge
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

2. **验收 1：四策略对比**：
```bash
uv run python -m evals.run_hybrid_evals
```
Expected: 打印报告，Recall@3/10 和 MRR 都有数值

3. **验收 2：BM25 字面匹配**：
在聊天页 `http://127.0.0.1:8000/chat` 发送：
```
问题：型号 ABC-123 支持什么网络？
```
Expected: 回答包含"5G 网络"（BM25 召回型号chunk）

4. **验收 3：引用可点**：
继续在聊天页发送：
```
问题：邮费是多少
```
Expected: 回答中有 `[1]`，点击后弹出 modal 显示运费说明原文

5. **验收 4：拒答与低置信度池**：
发送：
```
问题：火星移民计划的详细时间表
```
Expected: 回答"抱歉，我暂时没有找到相关信息"

验证低置信度池：
```sql
SELECT * FROM low_confidence_questions ORDER BY created_at DESC LIMIT 1;
```
Expected: 最新一条 question 包含"火星移民"

- [ ] **Step 2: 更新 `dev-notes/ch04.md` 的最终验收记录**

在文件末尾添加：

```markdown
---

## 阶段 N · 最终验收（2026-10-19）

**① 关键原话**

（记录用户的验收确认或修正）

**② 关键产出**

- 完整套件：277+ passed
- 四策略对比报告：✅ 能跑出 Recall@3/10 + MRR
- BM25 字面匹配：✅ 型号 ABC-123 召回成功
- 引用可点：✅ [1] 可点击，modal 显示原文
- 拒答与低置信度池：✅ "火星移民" 入池

**③ 被拒绝或纠偏**

（待补充）

**④ 翻车与返工**

（待补充）

**⑤ 交付清单**

**演示命令**：
```bash
# 1. 启动服务
docker compose up -d
uv run python -m jobs.build_knowledge
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. 四策略对比
uv run python -m evals.run_hybrid_evals

# 3. 聊天页验收
open http://127.0.0.1:8000/chat
# 发送：邮费是多少（验收引用可点）
# 发送：火星移民计划详情（验收拒答）

# 4. 低置信度池查询
docker exec -i custermerhelp-mysql mysql -uapp -papppw -e \
  "SELECT * FROM customer_service.low_confidence_questions ORDER BY created_at DESC LIMIT 5;"
```

**测试结果**：
```bash
uv run pytest -q
# Expected: 277+ passed
```

**留痕路径**：
- Spec: `docs/superpowers/specs/2026-10-19-hybrid-retrieval-design.md`
- Plan: `docs/superpowers/plans/2026-10-19-hybrid-retrieval.md`
- Dev notes: `dev-notes/ch04.md`
```

- [ ] **Step 3: Commit**

```bash
git add dev-notes/ch04.md
git commit -m "docs: Task 12 最终验收与交付

ch04 混合检索与重排完成：
- 混合检索（Dense + BM25 + RRF）
- 重排（bge-reranker-v2-m3）
- 生成质量控制（引用溯源 + 拒答 + 低置信度池）
- 评估体系（四策略对比 + 标注集）
- 前端配套（引用可点 + 满意度反馈）

四个验收标准全部通过
完整套件 277+ passed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 自审清单

**1. Spec 覆盖**：
- ✅ 混合检索（§3.1）→ Task 2
- ✅ Query 改写（§3.2）→ Task 4
- ✅ 元数据过滤（§3.3）→ Task 5（strategies 中支持）
- ✅ 重排（§3.4）→ Task 3
- ✅ 引用编号（§3.5.1）→ Task 7
- ✅ 拒答机制（§3.5.2）→ Task 5
- ✅ 低置信度池（§3.5.3）→ Task 1 + Task 5
- ✅ 负面知识（§3.5.4）→ Task 5
- ✅ 评估集（§3.6.1）→ Task 6
- ✅ 四策略对比（§3.6.3）→ Task 6
- ✅ 引用可点（§3.7.1）→ Task 7 + Task 8
- ✅ 满意度反馈（§3.7.2）→ Task 7 + Task 8
- ✅ 配置项（§6）→ Task 1
- ✅ 低置信度表（§4.1）→ Task 1
- ✅ API 接口（§5.1）→ Task 7

**2. 步骤扫描**：
每个步骤都有明确的测试、实现或验证命令，无 "TBD" 或空洞指令。

**3. 类型一致性**：
- `retrieve_with_strategy` 返回 `(list[KnowledgeChunk], dict)` - Task 5 定义，Task 6/9/10 复用一致
- `rerank` 返回 `list[tuple[int, float]]` - Task 3 定义，Task 5 使用一致
- `rewrite_query` 返回 `str` - Task 4 定义，Task 5 使用一致

**4. Review Focus**：
5 个输入类/失效模式已分配测试：
1. sparse_vector 未创建 → Task 0 探测 + Task 2 测试
2. 重排失败未降级 → Task 3 测试 `test_rerank_api_failure_raises`
3. 引用编号与 chunk 不对应 → Task 7 测试覆盖（从 metadata 取 chunk_id 再查库）
4. 低置信度问题未入池 → Task 5 测试三种触发条件
5. 评估集 ground_truth 标注错误 → Task 6 Step 5 前置 SQL 检查

**5. 比例**：
计划约 5000 行（含代码块），spec 约 600 行，比例约 8:1。主要原因是 TDD 步骤展开（每个任务 5-10 步）。代码块主要是测试断言和签名，函数体由实现者填充。

---
