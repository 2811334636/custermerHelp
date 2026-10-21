def test_workflow_settings_defaults():
    from app.config import Settings
    s = Settings()
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
        Settings(agent_max_steps=0)
