import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesTicket, ChatRequest, ExtractRequest


def test_chat_request_session_id_optional():
    r = ChatRequest(message="你好")
    assert r.session_id is None
    assert r.message == "你好"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_overlong_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 4001)


def test_extract_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")


def test_ticket_all_fields_default_to_none():
    t = AfterSalesTicket()
    assert t.order_id is None
    assert t.demand is None
    assert t.expected_solution is None


def test_ticket_accepts_partial_fields():
    t = AfterSalesTicket(order_id="A123")
    assert t.order_id == "A123"
    assert t.demand is None


def test_ticket_field_descriptions_present():
    """描述会进 JSON schema 送给模型，缺失会显著降低抽取质量。"""
    schema = AfterSalesTicket.model_json_schema()
    assert schema["properties"]["order_id"]["description"]
    assert schema["properties"]["demand"]["description"]
    assert schema["properties"]["expected_solution"]["description"]
