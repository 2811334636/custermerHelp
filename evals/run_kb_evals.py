"""ch03 检索质量评估集。

用法：uv run python -m evals.run_kb_evals

标注口径：期望命中的是知识块的叶子 heading（policy 块的 questions 字段）。
任何一条失败都必须以非 0 退出——本评估集是语义塌陷（向量检索"看起来正常
但召回的都不是答案"）唯一的自动防线。
"""

import asyncio
import sys
from pathlib import Path

import yaml

from app.config import get_settings
from app.db.base import AsyncSessionLocal, engine
from app.knowledge.retrieval import search

ROOT = Path(__file__).parent


async def main() -> int:
    cases = yaml.safe_load((ROOT / "kb_cases.yaml").read_text(encoding="utf-8"))
    top_k = get_settings().retrieval_top_k
    ok = 0

    try:
        async with AsyncSessionLocal() as session:
            for c in cases:
                hits = await search(session, c["query"], top_k=top_k)
                sections = [h.questions for h in hits]
                hit = c["expect_section"] in sections
                ok += hit
                print(f"  [{'PASS' if hit else 'FAIL'}] {c['query']:<14} → {sections}")
    finally:
        # 与 jobs/build_knowledge.py 一致：显式释放全局异步引擎，避免解释器
        # 关闭时 aiomysql 连接被 GC 回收而在 stderr 打出 "Event loop is closed"。
        await engine.dispose()

    total = len(cases)
    print(f"\nTop-{top_k} 命中率：{ok}/{total} = {ok / total:.1%}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
