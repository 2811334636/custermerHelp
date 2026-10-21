import os


def test_ch04_config_fields(monkeypatch):
    """ch04 新增配置字段存在且有默认值。"""
    from app.config import Settings

    # pymilvus 导入时会 load_dotenv()，把本地 .env 写进 os.environ；
    # 清掉 APP_* 再配合 _env_file=None，断言的才是字段声明默认值。
    for k in list(os.environ):
        if k.startswith("APP_"):
            monkeypatch.delenv(k)
    s = Settings(_env_file=None)
    assert s.reranker_base_url == "https://api.siliconflow.cn/v1"
    assert s.reranker_api_key == ""
    assert s.reranker_model == "bge-reranker-v2-m3"
    assert s.retrieval_dense_top_k == 50
    assert s.retrieval_bm25_top_k == 50
    assert s.retrieval_rerank_top_k == 10
    assert s.retrieval_rerank_threshold == 0.5
    assert s.retrieval_strategy == "hybrid_rerank"
    assert s.rerank_arrangement_strategy == "first_last"
