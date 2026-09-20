"""灌种子数据，并创建测试库。

用法：uv run python -m scripts.seed
幂等：faq 表先清空再灌，可重复执行。
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

FAQ_SEED = [
    ("退货政策是什么", "签收后 7 天内可无理由退货，商品需保持吊牌完整、不影响二次销售。定制类商品不支持无理由退货。", "售后"),
    ("怎么申请退货", "在 App「我的订单」里找到对应订单，点「申请售后」选退货，按提示填写后寄回即可。", "售后"),
    ("退货多久能退款", "仓库签收并验收通过后 1-3 个工作日内原路退回，具体到账时间以支付渠道为准。", "售后"),
    ("怎么换货", "在订单页点「申请售后」选换货，填写要更换的规格。换货商品寄回验收后我们会重新发货。", "售后"),
    ("换货运费谁承担", "因商品质量问题导致的换货由我们承担运费；因个人原因（如尺码不合适）需自行承担寄回运费。", "售后"),
    ("怎么开发票", "订单完成后在订单详情页点「申请开票」，填写抬头和税号，电子发票会在 3 个工作日内发送到您的邮箱。", "咨询"),
    ("发票能改抬头吗", "电子发票未开具前可以修改抬头；已开具的需要先冲红再重开，请联系人工客服处理。", "咨询"),
    ("多久发货", "现货商品在付款后 48 小时内发出；预售商品以商品页标注的发货时间为准。", "咨询"),
    ("发什么快递", "默认发中通或圆通，偏远地区可能转邮政。下单时无法指定快递公司。", "咨询"),
    ("怎么查物流", "在 App「我的订单」里点对应订单即可看到物流轨迹。", "咨询"),
    ("有价保吗", "支持 15 天价保：下单后 15 天内同商品官方渠道降价，可申请退还差价。", "咨询"),
    ("尺码怎么选", "商品详情页有尺码对照表；鞋类建议按脚长选，介于两个码之间时建议选大半码。", "咨询"),
]


async def main() -> None:
    s = get_settings()

    # 1. 生产库灌 faq
    engine = create_async_engine(s.db_url, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM faq"))
        for q, a, c in FAQ_SEED:
            await conn.execute(
                text("INSERT INTO faq (question, answer, category) VALUES (:q, :a, :c)"),
                {"q": q, "a": a, "c": c},
            )
        n = (await conn.execute(text("SELECT COUNT(*) FROM faq"))).scalar_one()
    print(f"customer_service.faq 已灌入 {n} 条")

    # 2. 建测试库并执行同一份 schema（保证表结构只有一份真相）
    test_url = s.db_url.rsplit("/", 1)[0] + "/"
    admin = create_async_engine(test_url, echo=False, isolation_level="AUTOCOMMIT")
    async with admin.begin() as conn:
        await conn.execute(text("CREATE DATABASE IF NOT EXISTS customer_service_test "
                                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    await admin.dispose()

    from pathlib import Path
    schema = Path("scripts/schema.sql").read_text(encoding="utf-8")
    test_engine = create_async_engine(test_url + "customer_service_test", echo=False)
    async with test_engine.begin() as conn:
        for stmt in [x.strip() for x in schema.split(";") if x.strip()]:
            await conn.execute(text(stmt))
        # 测试库也要灌种子：tests/test_repositories.py 的 faq 用例读的就是
        # customer_service_test，而 db_session fixture 用完即回滚，种子必须在
        # 库中预先存在。复用同一份 FAQ_SEED，避免两份数据各自漂移。
        await conn.execute(text("DELETE FROM faq"))
        for q, a, c in FAQ_SEED:
            await conn.execute(
                text("INSERT INTO faq (question, answer, category) VALUES (:q, :a, :c)"),
                {"q": q, "a": a, "c": c},
            )
        m = (await conn.execute(text("SELECT COUNT(*) FROM faq"))).scalar_one()
    await test_engine.dispose()
    print("customer_service_test 已建表")
    print(f"customer_service_test.faq 已灌入 {m} 条")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
