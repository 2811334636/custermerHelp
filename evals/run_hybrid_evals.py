"""四策略对比：dense / bm25 / hybrid / hybrid_rerank。

用法：uv run python -m evals.run_hybrid_evals

运行前提（本脚本是端到端评估，不是纯单测）：
- MySQL 可连（默认 127.0.0.1:3307/customer_service），知识块已建好；
- Milvus 可连（默认 127.0.0.1:19530），`knowledge` collection 已建好且含
  dense + BM25 两路索引；
- `APP_EMBEDDING_API_KEY` 必须配置：dense / hybrid / hybrid_rerank 共
  20 × 3 = 60 次线上 embedding 调用；
- `APP_RERANKER_API_KEY` 可选：配置后 hybrid_rerank 对 20 条 query 各调
  一次真实 reranker；未配置时显式降级（hybrid_rerank 等同 hybrid），并在
  报告里打印 0/20，避免把降级结果当成重排效果。

与 `run_kb_evals.py` 相同，ground_truth_ids 先与 knowledge_chunks 校验，
标错一个就整体失败——否则 Recall 会因一个 typo 永远缺一块，而且没人看得出来。

报告给出总体指标，以及按 query_type（fact / procedure / comparison）分桶的
指标（spec §3.6.3），另附 difficulty 分桶供人工判读。

判读注意：本评估集的 query 多为知识块问法的近义改写，语料天然偏向 dense；
RRF（k=60）+ identity 改写又可能让较弱的 BM25 单路在某些 query 上压过
dense。因此不能把这里的"dense 更高"推广成"dense 优于 hybrid"。
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from app.config import get_settings
from app.db.base import AsyncSessionLocal, engine
from app.db.models import KnowledgeChunk
from app.knowledge.errors import RerankerUnavailable
from app.knowledge.strategies import STRATEGIES, retrieve_with_strategy
from evals.metrics import mrr, recall_at_k

ROOT = Path(__file__).parent
CASES_PATH = ROOT / "hybrid_cases.yaml"

# 四策略都在同一召回深度上比较；Recall@10 需要至少 10 条结果，
# 再多取对 Recall@3/10 与 MRR 没有意义，徒增 embedding / 重排开销。
TOP_K = 10

QUERY_TYPES = ("fact", "procedure", "comparison")
DIFFICULTIES = ("easy", "medium", "hard")
METRICS = ("recall@3", "recall@10", "mrr")


async def _identity_rewrite(query: str) -> str:
    """评估用的 query 改写替身：原样返回。

    四策略对比必须只反映检索本身。若让 bm25 / hybrid / hybrid_rerank 各自
    调用真实 LLM 改写，20 条 query × 3 个策略 = 60 次线上调用，既慢又花钱，
    还会把改写模型的抖动混进策略差异里——同一条 query 两次运行可能因为改写
    结果不同而排出不同名次。
    """
    return query


async def _rerank_unavailable(
    query: str, passages: list[str], top_k: int = 10
) -> list[tuple[int, float]]:
    """未配置 reranker key 时的显式降级。

    调用真实 `reranker.rerank` 也能靠 RerankerUnavailable 回退，但那会为
    20 条 query 各打一次注定 401 的线上请求。这里直接抛同一个异常，让
    `hybrid_rerank` 走既有的降级分支，报告里的 rerank_applied 仍是诚实的。
    """
    raise RerankerUnavailable("未配置 reranker api key，评估跳过重排")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """读取并做结构校验；任何一条不合法都直接失败。"""
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"评估集 {path} 为空或不是列表")

    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise SystemExit(f"第 {i} 条用例不是映射：{case!r}")
        missing = {
            "query", "ground_truth_ids", "difficulty", "query_type"
        } - set(case)
        if missing:
            raise SystemExit(f"第 {i} 条用例缺少字段：{sorted(missing)}")
        if not isinstance(case["ground_truth_ids"], list) or not case["ground_truth_ids"]:
            raise SystemExit(f"第 {i} 条用例 ground_truth_ids 必须是非空列表")
        if case["difficulty"] not in DIFFICULTIES:
            raise SystemExit(
                f"第 {i} 条用例 difficulty={case['difficulty']!r} 不在 {DIFFICULTIES}"
            )
        if case["query_type"] not in QUERY_TYPES:
            raise SystemExit(
                f"第 {i} 条用例 query_type={case['query_type']!r} 不在 {QUERY_TYPES}"
            )

    # 分桶覆盖度：spec 要求三种 query 类型都要有，且每类至少 2 条才有比较意义。
    counts = defaultdict(int)
    for case in cases:
        counts[case["query_type"]] += 1
    short = [t for t in QUERY_TYPES if counts[t] < 2]
    if short:
        raise SystemExit(
            "评估集分桶覆盖不足（每类至少 2 条）："
            + ", ".join(f"{t}={counts[t]}" for t in short)
        )
    missing_difficulty = [d for d in DIFFICULTIES if d not in {c["difficulty"] for c in cases}]
    if missing_difficulty:
        raise SystemExit(f"评估集缺少难度：{missing_difficulty}")

    return cases


async def _validate_against_db(session, cases: list[dict[str, Any]]) -> None:
    """ground_truth_ids 必须都是 knowledge_chunks 里真实存在的已向量化 chunk。"""
    known = set(
        (
            await session.execute(
                select(KnowledgeChunk.chunk_id).where(
                    KnowledgeChunk.status == "已向量化"
                )
            )
        ).scalars()
    )
    referenced = {
        chunk_id for case in cases for chunk_id in case["ground_truth_ids"]
    }
    unknown = sorted(referenced - known)
    if unknown:
        raise SystemExit(
            "评估集标注校验失败：以下 ground_truth_ids 不在 knowledge_chunks"
            "（已向量化）中，请检查拼写或重新标注：\n  " + "\n  ".join(unknown)
        )


def _empty_scores() -> dict[str, list[float]]:
    return {metric: [] for metric in METRICS}


def _record(
    bucket: dict[str, list[float]],
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
) -> None:
    bucket["recall@3"].append(recall_at_k(retrieved_ids, ground_truth_ids, 3))
    bucket["recall@10"].append(recall_at_k(retrieved_ids, ground_truth_ids, 10))
    bucket["mrr"].append(mrr(retrieved_ids, ground_truth_ids))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _table(
    title: str,
    buckets: dict[str, dict[str, dict[str, list[float]]]],
    order: tuple[str, ...],
) -> None:
    print(f"\n{title}")
    print(f"{'策略':<16}{'Recall@3':>10}{'Recall@10':>11}{'MRR':>8}   (n)")
    for bucket in order:
        scores = buckets.get(bucket)
        if not scores:
            continue
        if len(order) > 1:
            print(f"[{bucket}]")
        n = len(scores[STRATEGIES[0]]["mrr"])
        for strategy in STRATEGIES:
            s = scores[strategy]
            print(
                f"  {strategy:<14}{_mean(s['recall@3']):>10.3f}"
                f"{_mean(s['recall@10']):>11.3f}{_mean(s['mrr']):>8.3f}   ({n})"
            )


async def main() -> int:
    cases = _load_cases(CASES_PATH)
    settings = get_settings()
    rerank_configured = bool(settings.reranker_api_key)
    # 配了 key 就走真实重排（不传 rerank_fn）；没配就显式降级，原因见上面注释。
    rerank_fn = None if rerank_configured else _rerank_unavailable

    overall = {strategy: _empty_scores() for strategy in STRATEGIES}
    by_type: dict[str, dict[str, dict[str, list[float]]]] = {
        query_type: {strategy: _empty_scores() for strategy in STRATEGIES}
        for query_type in QUERY_TYPES
    }
    by_difficulty: dict[str, dict[str, dict[str, list[float]]]] = {
        difficulty: {strategy: _empty_scores() for strategy in STRATEGIES}
        for difficulty in DIFFICULTIES
    }
    rerank_applied = 0
    empty_hits = {strategy: 0 for strategy in STRATEGIES}

    try:
        async with AsyncSessionLocal() as session:
            await _validate_against_db(session, cases)
            print(f"评估集：{len(cases)} 条，召回深度 Top-{TOP_K}")
            print(f"数据库：{settings.db_url.rsplit('/', 1)[-1]}")

            for i, case in enumerate(cases, 1):
                query = case["query"]
                ground_truth = case["ground_truth_ids"]
                print(f"\n[{i:02d}/{len(cases)}] {query}")
                for strategy in STRATEGIES:
                    chunks, meta = await retrieve_with_strategy(
                        session,
                        query,
                        strategy,
                        TOP_K,
                        # 四策略共用同一个确定性改写替身，比较的是检索而非 LLM。
                        rewrite_fn=_identity_rewrite,
                        rerank_fn=rerank_fn,
                    )
                    retrieved_ids = [chunk.chunk_id for chunk in chunks]
                    if not retrieved_ids:
                        empty_hits[strategy] += 1
                    _record(overall[strategy], retrieved_ids, ground_truth)
                    _record(
                        by_type[case["query_type"]][strategy],
                        retrieved_ids,
                        ground_truth,
                    )
                    _record(
                        by_difficulty[case["difficulty"]][strategy],
                        retrieved_ids,
                        ground_truth,
                    )
                    if strategy == "hybrid_rerank" and meta.get("rerank_applied"):
                        rerank_applied += 1
                    print(
                        f"  {strategy:<14} n={len(retrieved_ids):<3}"
                        f" recall@10={recall_at_k(retrieved_ids, ground_truth, 10):.2f}"
                        f" hits={retrieved_ids[:5]}"
                    )
    finally:
        # 与 run_kb_evals.py 一致：显式释放全局异步引擎，避免解释器关闭时
        # aiomysql 连接被 GC 回收而在 stderr 打出 "Event loop is closed"。
        await engine.dispose()

    print("\n" + "=" * 64)
    print("四策略对比报告")
    print("=" * 64)
    if rerank_configured:
        print("重排配置：已配置 reranker key，hybrid_rerank 使用真实重排模型")
    else:
        print(
            "重排配置：未配置 reranker key！hybrid_rerank 已退化为 hybrid，"
            "本行数字不代表重排效果"
        )
    print(f"hybrid_rerank 实际启用重排：{rerank_applied}/{len(cases)} 条")
    print(
        "判读注意：query 多为知识块问法的近义改写，语料偏向 dense；"
        "RRF k=60 + identity 改写可能让较弱的 BM25 单路在个别 query 上"
        "压过 dense，不能据此推广「dense 优于 hybrid」。"
    )

    _table("总体", {"all": overall}, ("all",))
    _table("按 query_type 分桶", by_type, QUERY_TYPES)
    _table("按 difficulty 分桶（参考）", by_difficulty, DIFFICULTIES)

    empty = {s: n for s, n in empty_hits.items() if n}
    if empty:
        print("\n警告：以下策略存在空召回：")
        for strategy, n in empty.items():
            print(f"  {strategy}：{n}/{len(cases)} 条")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
