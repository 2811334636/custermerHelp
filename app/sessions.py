from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.messages_io import from_rows, to_rows
from app.db.models import Conversation
from app.repositories import conversation as conversation_repo
from app.repositories import message as message_repo


class SessionStore(ABC):
    """会话历史存储抽象。

    ★ ch02 起三个方法都是 async —— 因为 DB 访问要 await。
    这相对 ch01 是破坏性变更，调用方（api.py）必须跟着加 await。
    """

    @abstractmethod
    async def get(self, session_id: str) -> list[BaseMessage]: ...

    @abstractmethod
    async def append(
        self,
        session_id: str,
        *messages: BaseMessage,
        metadata: dict | None = None,
    ) -> list[int]: ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool: ...


class MySQLSessionStore(SessionStore):
    """基于 MySQL 的实现：上下文从 messages 表读回，重启不丢。

    每通会话在 conversations 表里有一行，messages 通过 conversation_id 挂上去。
    对外仍用 UUID 形式的 session_id（= conversations.session_uuid）。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _conversation_id(self, session_id: str) -> int | None:
        conv = await self._find(session_id)
        return None if conv is None else conv.id

    async def _find(self, session_id: str) -> Conversation | None:
        from sqlalchemy import select

        stmt = select(Conversation).where(Conversation.session_uuid == session_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get(self, session_id: str) -> list[BaseMessage]:
        cid = await self._conversation_id(session_id)
        if cid is None:
            return []
        rows = await message_repo.list_by_conversation(self._session, cid)
        return from_rows(rows)

    async def append(
        self,
        session_id: str,
        *messages: BaseMessage,
        metadata: dict | None = None,
    ) -> list[int]:
        """追加消息并返回插入的主键列表（与 messages 顺序一致）。

        metadata 只挂到**最后一条**消息上：调用方把整轮序列一次性交进来时，
        citations 属于最终的 assistant 回答，不属于 user/tool 行。
        """
        conv = await conversation_repo.get_or_create(
            self._session, session_id, get_settings().default_user_id
        )
        rows = to_rows(list(messages))
        ids: list[int] = []
        for i, row in enumerate(rows):
            ids.append(
                await message_repo.append(
                    self._session,
                    conv.id,
                    **row,
                    metadata=metadata if i == len(rows) - 1 else None,
                )
            )
        return ids

    async def exists(self, session_id: str) -> bool:
        return await self._find(session_id) is not None
