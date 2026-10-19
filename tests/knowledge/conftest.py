"""Fixtures for knowledge tests."""

import time

import pytest
from pymilvus import MilvusClient

from app.config import get_settings


@pytest.fixture
def test_collection_bm25():
    """为 BM25 测试创建并清理独立 collection。"""
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    coll = f"test_bm25_{int(time.time() * 1000)}"

    yield coll

    if client.has_collection(coll):
        client.drop_collection(coll)
