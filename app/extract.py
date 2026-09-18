from app.prompts import build_extract_prompt
from app.providers import get_chat_model
from app.schemas import AfterSalesTicket


class ExtractionFailed(Exception):
    """模型未能按 schema 返回结构化结果。"""


async def extract_ticket(text: str, *, model=None) -> AfterSalesTicket:
    """把一段售后描述抽取成结构化工单。

    method="function_calling" 是必须显式写出的：上游 response_format 的
    json_schema 完全不可用，function_calling 是唯一可行路径（spec §3.4）。
    显式写出也能防止 langchain-openai 默认值变动时静默漂移。
    """
    model = model or get_chat_model()
    messages = build_extract_prompt().format_messages(text=text)
    try:
        # with_structured_output 也要在 try 内：它本身会抛（上游不支持
        # tool_call 时），那同样属于"抽取失败"，应按 spec §9 走 502
        # extraction_failed，而不是漏成 500。
        structured = model.with_structured_output(
            AfterSalesTicket, method="function_calling"
        )
        result = await structured.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 — 上游任意异常统一归为抽取失败
        raise ExtractionFailed(str(exc)[:500]) from exc

    if result is None:
        raise ExtractionFailed("模型未返回结构化结果")
    return result
