import pytest

from app.extract import ExtractionFailed, extract_ticket
from app.schemas import AfterSalesTicket


class FakeStructured:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    async def ainvoke(self, messages):
        self.seen_messages = messages
        if self._raises:
            raise self._raises
        return self._result


class FakeModel:
    def __init__(self, structured):
        self._structured = structured
        self.used_method = None

    def with_structured_output(self, schema, method=None, **kwargs):
        self.schema = schema
        self.used_method = method
        return self._structured


async def test_extract_returns_ticket():
    model = FakeModel(FakeStructured(result=AfterSalesTicket(order_id="A123", demand="退款")))
    ticket = await extract_ticket("订单 A123 我要退款", model=model)
    assert ticket.order_id == "A123"
    assert ticket.demand == "退款"


async def test_extract_uses_function_calling_method():
    """上游 json_schema 完全不可用，function_calling 是唯一路径（spec §3.4）。"""
    model = FakeModel(FakeStructured(result=AfterSalesTicket()))
    await extract_ticket("随便", model=model)
    assert model.used_method == "function_calling"


async def test_extract_uses_ticket_schema():
    model = FakeModel(FakeStructured(result=AfterSalesTicket()))
    await extract_ticket("随便", model=model)
    assert model.schema is AfterSalesTicket


async def test_extract_sends_system_and_user_messages():
    model = FakeModel(FakeStructured(result=AfterSalesTicket()))
    await extract_ticket("订单 A123 要退款", model=model)
    assert model._structured.seen_messages[0].type == "system"
    assert model._structured.seen_messages[1].content == "订单 A123 要退款"


async def test_extract_raises_extraction_failed_when_model_returns_nothing():
    model = FakeModel(FakeStructured(raises=ValueError("no tool call")))
    with pytest.raises(ExtractionFailed):
        await extract_ticket("随便", model=model)
