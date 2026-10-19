import datetime

from sqlalchemy import BigInteger, Boolean, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_uuid: Mapped[str] = mapped_column(String(36), unique=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"),
        default="进行中",
    )
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_user_id", "user_id"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="message_role"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("idx_conversation_id", "conversation_id"),)


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_category", "category"),)


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type")
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), default="待处理"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("idx_tickets_conversation_id", "conversation_id"),)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(
        Enum("policy", "faq", "mined", name="knowledge_content_type")
    )
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("待向量化", "已向量化", name="knowledge_status"), default="待向量化"
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class MinedQaStaging(Base):
    __tablename__ = "mined_qa_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("待入库", "已入库", "已丢弃", name="mined_status"), default="待入库"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
