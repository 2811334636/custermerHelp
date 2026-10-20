"""GET /api/chunks/{chunk_id} —— 引用溯源接口。"""

import pytest
from fastapi.testclient import TestClient

import app.api_chunks as api_chunks
from app.api_chunks import get_chunk
from app.db.base import get_db
from app.knowledge.chunk import Chunk
from app.main import app
from app.repositories import knowledge_chunk as repo


@pytest.fixture
def client():
    """HTTP 层用例不碰库：repo 由各用例打桩，get_db 只提供占位值。"""
    app.dependency_overrides[get_db] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="test.md:1",
        category="测试分类",
        questions="测试问题",
        answer="测试内容",
        section_path="测试分类 > 测试问题",
        content_type="policy",
        is_key_clause=True,
    )


async def test_get_chunk_by_id_queries_repository(db_session):
    """路由逻辑 + 真实 repo：返回 chunk 原文与元数据，source 由 id 前缀还原。"""
    await repo.upsert_chunks(db_session, [_chunk()])

    data = await get_chunk("test.md:1", session=db_session)

    assert data == {
        "chunk_id": "test.md:1",
        "content": "测试内容",
        "questions": "测试问题",
        "category": "测试分类",
        "section_path": "测试分类 > 测试问题",
        "content_type": "policy",
        "is_key_clause": True,
        "source": "test.md",
    }


def test_get_chunk_endpoint(client, monkeypatch):
    async def fake_get_by_ids(session, chunk_ids, *, status=None):
        return [_chunk()]

    monkeypatch.setattr(api_chunks.repo, "get_by_ids", fake_get_by_ids)

    resp = client.get("/api/chunks/test.md:1")
    assert resp.status_code == 200
    assert resp.json()["chunk_id"] == "test.md:1"
    assert resp.json()["source"] == "test.md"


def test_get_chunk_not_found(client, monkeypatch):
    """chunk_id 不存在时返回 404 扁平错误体。"""
    async def fake_get_by_ids(session, chunk_ids, *, status=None):
        return []

    monkeypatch.setattr(api_chunks.repo, "get_by_ids", fake_get_by_ids)

    resp = client.get("/api/chunks/nonexistent:999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "chunk_not_found"
    assert "message" in body
