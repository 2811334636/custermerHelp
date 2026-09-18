import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response

from app.api import router

logger = logging.getLogger(__name__)

app = FastAPI(title="电商智能客服 ch01", version="0.1.0")
app.include_router(router)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.exception_handler(HTTPException)
async def flat_http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """业务错误摊平成顶层 {code, message}。

    注意范围（不要过度声称）：只有**我们主动抛出、且 detail 是 dict** 的错误
    才是扁平的——目前就是 extract 的 502 extraction_failed。其余错误仍走
    FastAPI 默认形态：422 是 {"detail": [...]}，404/405 是 {"detail": "..."}。

    FastAPI 默认把 HTTPException(detail={...}) 包成 {"detail": {...}}，
    这里把 dict 形态的 detail 摊平到顶层；非 dict 的 detail 交回默认行为。
    """
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # spec §9：未处理异常对客户端只暴露扁平 message，但服务端必须留完整堆栈，
    # 否则 500 只剩一句 "internal_error"，无法定位。
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": str(exc)[:500]},
    )
