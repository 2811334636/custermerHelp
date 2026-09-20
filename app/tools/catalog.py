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


# ★ 下面两个判定必须由 query_order 与 query_logistics **共用**。
# 让它们各自从独立的种子流里判定，会出现同一个订单号"不存在却又有完整
# 物流轨迹"，或者订单说"已发货"而物流最后一条说"已签收"——验收标准 1 要
# 拿模型回答与工具输出逐字对账，两个工具自相矛盾会直接毁掉这个验收。
def _order_exists(order_id: str) -> bool:
    """订单是否存在。两个工具共用，故不能并进各自的种子流。"""
    return _rng(f"order-exists:{order_id}").random() >= 0.05


def _order_status(order_id: str) -> str:
    """订单状态。同样共用；种子前缀与 order-exists 分开，避免同一条流互相挤占。"""
    return _rng(f"order-status:{order_id}").choice(_STATUSES)


_STATUS_TO_LAST_NODE = {
    "已下单": "已揽收",
    "已发货": "已发出",
    "运输中": "运输中",
    "已到达": "已到达",
    "派送中": "派送中",
    "已签收": "已签收",
}


@tool
async def query_order(order_id: str) -> str:
    """按订单号查询订单详情，返回商品、下单时间、金额与订单状态。"""
    if not _order_exists(order_id):
        return f"没有找到订单 {order_id}。请确认订单号是否正确，或联系人工客服。"

    # 商品/时间/金额仍走本工具自己的种子流（order:），与物流的 logistics: 独立，
    # 免得 query_order 的"商品"和 query_logistics 的"承运商"变成同一个值。
    r = _rng(f"order:{order_id}")
    return (
        f"订单号 {order_id}：\n"
        f"- 商品：{r.choice(_GOODS)}\n"
        f"- 规格：{r.choice(_SPECS)}\n"
        f"- 下单时间：2026-0{r.randint(1, 9)}-{r.randint(10, 28):02d} "
        f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}\n"
        f"- 实付金额：{r.randint(59, 899)}.{r.randint(0, 99):02d} 元\n"
        f"- 订单状态：{_order_status(order_id)}"
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
    # 与 query_order 共用同一个存在性判定，避免"订单不存在却有完整轨迹"
    if not _order_exists(order_id):
        return f"没有找到订单 {order_id} 的物流信息。请确认订单号是否正确。"

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
    # ★ 最后一条必须与 query_order 报的订单状态一致，否则两个工具会自相矛盾
    last_node = _STATUS_TO_LAST_NODE[_order_status(order_id)]
    latest = (
        f"- 2026-09-20 {r.randint(8, 22):02d}:{r.randint(0, 59):02d} "
        f"{r.choice(_CITIES)}：{last_node}"
    )
    return (
        f"订单 {order_id} 的物流信息：\n"
        f"- 承运商：{carrier}\n"
        f"- 运单号：{waybill}\n"
        f"- 轨迹：\n" + "\n".join(lines) + "\n" + latest
    )
