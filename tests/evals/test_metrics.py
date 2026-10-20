"""evals/metrics.py 的指标计算单测。

指标是四策略对比的判据：算错了，报告里的数字就是错的，而且错得很隐蔽
（分母用错时两条策略可能都"看起来差不多"）。这里把 Recall@K 与 MRR 的
边界钉死，不依赖网络与数据库。
"""

from evals.metrics import mrr, recall_at_k


def test_recall_at_k():
    """Recall@K 计算正确。"""
    retrieved = ["a", "b", "c", "d"]
    ground_truth = ["a", "c", "e"]

    assert recall_at_k(retrieved, ground_truth, k=2) == 1 / 3  # 召回 a，漏 c、e
    assert recall_at_k(retrieved, ground_truth, k=3) == 2 / 3  # 召回 a、c，漏 e
    assert recall_at_k(retrieved, ground_truth, k=10) == 2 / 3  # 最多召回 2 个


def test_mrr():
    """MRR 计算正确。"""
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3  # 第 3 位命中
    assert mrr(["a", "b", "c"], ["a"]) == 1.0  # 第 1 位命中
    assert mrr(["a", "b", "c"], ["d"]) == 0.0  # 未命中


def test_recall_at_k_empty_ground_truth():
    """没有标注答案时返回 0，而不是 ZeroDivisionError。"""
    assert recall_at_k(["a"], [], k=3) == 0.0


def test_recall_at_k_k_zero():
    """K=0 时没有任何位置可命中。"""
    assert recall_at_k(["a", "b"], ["a"], k=0) == 0.0
