import os


def test_workflow_settings_defaults(monkeypatch):
    from app.config import Settings

    # pymilvus 导入时会 load_dotenv()，把本地 .env 写进 os.environ；
    # 清掉 APP_* 再配合 _env_file=None，断言的才是字段声明默认值。
    for k in list(os.environ):
        if k.startswith("APP_"):
            monkeypatch.delenv(k)
    s = Settings(_env_file=None)
    assert s.checkpoint_db == "./data/langgraph_checkpoints.sqlite"
    assert s.confidence_min_coverage == 0.30
    assert s.agent_max_steps == 6
    assert s.agent_token_budget == 6000
    assert s.intent_model_temperature == 0.0
    assert s.workflow_log_enabled is True


def test_workflow_settings_validation():
    import pytest
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(confidence_min_coverage=1.5)
    with pytest.raises(ValidationError):
        Settings(confidence_min_coverage=-0.01)
    with pytest.raises(ValidationError):
        Settings(agent_max_steps=0)
    with pytest.raises(ValidationError):
        Settings(agent_token_budget=0)
    with pytest.raises(ValidationError):
        Settings(agent_token_budget=-1)
    with pytest.raises(ValidationError):
        Settings(intent_model_temperature=-0.1)
    with pytest.raises(ValidationError):
        Settings(intent_model_temperature=2.1)
