from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response

from app.api import router

app = FastAPI(title="电商智能客服 ch01", version="0.1.0")
app.include_router(router)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.exception_handler(HTTPException)
async def flat_http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """Ruling 4：spec §9 规定所有错误响应都是扁平的 {code, message}。

    FastAPI 默认把 HTTPException(detail={...}) 包成 {"detail": {...}}，
    这里把 dict 形态的 detail 摊平到顶层；非 dict 的 detail 交回默认行为。
    """
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": str(exc)[:500]},
    )
