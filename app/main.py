import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api import router
from app.config import get_settings
from app.workflow.graph import build_workflow

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """进程生命周期内保持 checkpointer 连接，并只编译一次工作流。

    ``AsyncSqliteSaver.from_conn_string`` 是异步上下文管理器；退出 lifespan
    时关闭连接，避免进程挂住。checkpoint 目录不存在时先建出来（首次部署
    ``./data`` 通常还没有）。
    """
    settings = get_settings()
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(
        settings.checkpoint_db
    ) as checkpointer:
        app.state.checkpointer = checkpointer
        app.state.workflow = build_workflow(checkpointer)
        yield


app = FastAPI(
    title="电商智能客服 ch05 · LangGraph 工作流",
    version="0.5.0",
    lifespan=lifespan,
)
app.include_router(router)

# 聊天页：单文件原生 JS，零构建。挂载在 /chat，与 /api 前缀不冲突。
# 目录不存在时跳过（例如只装了后端、没带 web/ 的部署），不让应用起不来。
_web = Path(__file__).resolve().parent.parent / "web"
if _web.is_dir():
    app.mount("/chat", StaticFiles(directory=_web, html=True), name="chat")


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
