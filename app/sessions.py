from abc import ABC, abstractmethod
from collections import OrderedDict

from langchain_core.messages import BaseMessage


class SessionStore(ABC):
    """会话历史存储抽象。

    当前只有进程内实现（重启即丢、不支持多 worker）。
    抽象的意义：以后换 SQLite / Redis 不需要动任何业务代码。
    """

    @abstractmethod
    def get(self, session_id: str) -> list[BaseMessage]: ...

    @abstractmethod
    def append(self, session_id: str, *messages: BaseMessage) -> None: ...

    @abstractmethod
    def exists(self, session_id: str) -> bool: ...


class InMemorySessionStore(SessionStore):
    """进程内实现。

    内存上界由两个 knob 共同保证：
    - max_sessions：会话总数上限，超出按 LRU 淘汰整个会话
    - max_messages：单会话消息数上限，超出丢弃最旧消息
    """

    def __init__(self, max_sessions: int = 200, max_messages: int = 100) -> None:
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._data: OrderedDict[str, list[BaseMessage]] = OrderedDict()

    def get(self, session_id: str) -> list[BaseMessage]:
        if session_id not in self._data:
            return []
        self._data.move_to_end(session_id)  # 标记为最近使用
        return list(self._data[session_id])  # 返回副本，防止调用方污染存储

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        bucket = self._data.setdefault(session_id, [])
        bucket.extend(messages)
        if len(bucket) > self._max_messages:
            del bucket[: len(bucket) - self._max_messages]
        self._data.move_to_end(session_id)
        while len(self._data) > self._max_sessions:
            self._data.popitem(last=False)  # 淘汰最久未使用的会话

    def exists(self, session_id: str) -> bool:
        return session_id in self._data
