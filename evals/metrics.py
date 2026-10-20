"""评估指标：Recall@K 和 MRR。

两者都只看「命中的 chunk_id 排在第几」，不看分数，因此可以横跨
dense / bm25 / hybrid / hybrid_rerank 四种策略直接比较。
"""


def recall_at_k(
    retrieved_ids: list[str], ground_truth_ids: list[str], k: int
) -> float:
    """Recall@K = (Top-K 中命中的 ground_truth 数) / (ground_truth 总数)。"""
    if not ground_truth_ids:
        return 0.0

    top_k = set(retrieved_ids[:k]) if k > 0 else set()
    hit = len(top_k & set(ground_truth_ids))
    return hit / len(ground_truth_ids)


def mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    """MRR = 1 / (第一个命中的 ground_truth 的排名)。未命中返回 0。"""
    ground_truth = set(ground_truth_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in ground_truth:
            return 1 / rank
    return 0.0
