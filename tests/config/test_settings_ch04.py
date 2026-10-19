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
