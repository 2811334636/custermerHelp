"""三个返回确定性假数据的工具。

ch02 不接真实的电商/物流系统，但数据必须**可重复**：
同一个订单号两次调用返回完全相同的内容，这样验收时能对着工具结果
逐字核对模型的回答（spec §6.1）。

确定性来自 key 的 sha256 派生的种子——不是「多半一样」，是逐字节相同。
"""

import hashlib
import random

from langchain_core.tools import tool

_CARRIERS = ["顺丰速运", "中通快递", "圆通速递", "韵达快递", "京东物流"]
_CITIES = ["杭州", "上海", "北京", "广州", "成都", "武汉", "西安", "南京"]
_STATUSES = ["已下单", "已发货", "运输中", "已到达", "派送中", "已签收"]
_GOODS = ["轻量跑步鞋", "纯棉圆领T恤", "不锈钢保温杯", "降噪蓝牙耳机", "双肩通勤背包"]
_SPECS = ["黑色 / 42 码", "白色 / L", "500ml", "深空灰", "藏青"]


def _rng(key: str) -> random.Random:
    """由 key 派生的确定性随机源。同一个 key 永远得到同一串数据。

    ★ 不要换成 `random` 模块的全局函数：那会让同一个订单号每次查出不同结果，
    验收标准 1 的「模型回答 vs 工具结果」对账就退化成「看着像对」。
    """
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return random.Random(seed)


@tool
async def query_order(order_id: str) -> str:
    """按订单号查询订单详情，返回商品、下单时间、金额与订单状态。"""
    r = _rng(f"order:{order_id}")
    if r.random() < 0.05:
        return f"没有找到订单 {order_id}。请确认订单号是否正确，或联系人工客服。"
    return (
        f"订单号 {order_id}：\n"
        f"- 商品：{r.choice(_GOODS)}\n"
        f"- 规格：{r.choice(_SPECS)}\n"
        f"- 下单时间：2026-0{r.randint(1, 9)}-{r.randint(10, 28):02d} "
        f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}\n"
        f"- 实付金额：{r.randint(59, 899)}.{r.randint(0, 99):02d} 元\n"
        f"- 订单状态：{r.choice(_STATUSES)}"
    )


@tool
async def query_product(product_id: str) -> str:
    """按商品号查询商品信息，返回名称、价格、规格与库存。"""
    r = _rng(f"product:{product_id}")
    stock = r.randint(0, 500)
    return (
        f"商品号 {product_id}：\n"
        f"- 名称：{r.choice(_GOODS)}\n"
        f"- 售价：{r.randint(59, 899)}.{r.randint(0, 99):02d} 元\n"
        f"- 可选规格：{'、'.join(r.sample(_SPECS, 2))}\n"
        f"- 库存：{'缺货' if stock == 0 else str(stock) + ' 件'}"
    )


@tool
async def query_logistics(order_id: str) -> str:
    """按订单号查询物流轨迹，返回承运商、运单号与若干条轨迹节点。"""
    r = _rng(f"logistics:{order_id}")
    carrier = r.choice(_CARRIERS)
    waybill = f"{r.choice('SFYTJD')}{r.randint(10**11, 10**12 - 1)}"
    n = r.randint(3, 5)
    cities = r.sample(_CITIES, n)
    lines = [
        f"- 2026-09-{r.randint(10, 19):02d} {r.randint(8, 22):02d}:{r.randint(0, 59):02d} "
        f"{c}转运中心：{r.choice(['已揽收', '已发出', '已到达', '运输中'])}"
        for c in cities
    ]
    latest = (
        f"- 2026-09-20 {r.randint(8, 22):02d}:{r.randint(0, 59):02d} "
        f"{r.choice(_CITIES)}：{r.choice(['派送中', '已签收', '待取件'])}"
    )
    return (
        f"订单 {order_id} 的物流信息：\n"
        f"- 承运商：{carrier}\n"
        f"- 运单号：{waybill}\n"
        f"- 轨迹：\n" + "\n".join(lines) + "\n" + latest
    )
