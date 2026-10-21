"""两个评估集共用的底座：生产回合驱动器 + 真实测试库会话 + 确定性转述判据。

★ 为什么直接驱动生产工作流（ch05）
评估的价值在于测"真身"。ch05 起回合由 ``app/workflow`` 的 LangGraph 图驱动，
本文件不再复刻"先转发后决断 / 工具执行 / 收敛"的循环，而是把同一张图
（``build_workflow``）跑一遍，只做事件翻译。checkpointer 用进程内
``InMemorySaver``：评估是一问一答，不需要跨轮持久化。

★ 为什么连真库（修复轮 · Important 3）
session=None 时 query_faq / create_ticket 一律回"暂时不可用"，5 条 faq/工单用例
全部落到 _relay_check 的弱分支——一条"什么都没转述"的回复也能过，验收标准 2
（工具拿真数据）于是没有任何自动化覆盖。这里连 spec §5.6 的 customer_service_test，
跑在**末尾回滚**的一笔事务里，库里不留痕。

★ 为什么转述判据是确定性的（修复轮 · Important 1）
最初它用 LLM judge。实测 judge 不是"偶尔抖"，是**系统性判错**：给一条逐字转述了
商品/规格/金额/状态的回复连判 4 次全是"否"。拿它当闸门，评估本身就变成随机的。
两个评估集现在都以这里的确定性判据为闸门，judge 只作为参考打印。
"""

import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.repositories import conversation as conv_repo
from app.workflow.graph import build_workflow, workflow_deps

# ---------------------------------------------------------------- 回合驱动器


@dataclass(frozen=True)
class Turn:
    """一个完整回合的可断言产物，全部来自生产函数的事件流。"""

    tool_calls: list[dict]
    tool_output: str        # 全部工具结果的拼接；没调工具时为空串
    tool_ok: list[bool]     # 每次工具执行是否成功（查不到也算成功，见 spec §11.1）
    first_round_text: str   # 首轮正文（模型决定调工具前说的话，已实时转发给用户）
    final_text: str         # 末轮正文；没调工具时它就是整条回复
    errors: list[str] = field(default_factory=list)

    @property
    def visible_text(self) -> str:
        """用户实际看到的两段文本。

        ★ 为什么断言要跑在这上面而不是 final_text：工具轮把一次回复拆成了两段，
        只查 final_text 会把首轮那段（"稍等，我帮您查一下"）漏掉——长度上限与
        禁用词都可能在那段里被突破，而套件结构性看不见。没调工具时
        first_round_text 为空、final_text 就是整条回复，两段拼起来与旧口径一致。
        """
        parts = [t.strip() for t in (self.first_round_text, self.final_text) if t.strip()]
        return "\n".join(parts)


async def run_production_turn(
    message: str,
    *,
    model,
    registry,
    conversation_id: int = 0,
    history: list | None = None,
    token_budget: int | None = None,
    session=None,
) -> Turn:
    """喂给生产工作流一句话，把 custom 事件流翻译成 Turn。

    事件契约（``stream_mode="custom"``）：``{"type": "delta", "text": ...}`` 与
    ``{"type": "tool", "phase": "start|end", ...}``。首轮正文 = 第一次 tool end
    之前的增量，末轮正文 = tool end 之后的增量；没工具时全部增量就是整条回复。

    工具回灌全文不在 custom 事件里（帧里只有摘要），所以从最终 State 的
    ``ToolMessage`` 取——评估的转述判据必须看模型真正拿到的内容。

    上游报错**直接抛**，不降级：否则一次网络故障会伪装成"模型没调工具"，
    评估转绿而实际什么都没测到。
    """
    settings = get_settings()
    budget = token_budget if token_budget is not None else settings.agent_token_budget
    graph = build_workflow(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": f"eval-{uuid.uuid4()}",
            **workflow_deps(model, registry, session, conversation_id),
        }
    }
    initial = {
        "messages": [*(history or []), HumanMessage(message)],
        "user_input": message,
        "max_steps": settings.agent_max_steps,
        "token_budget": budget,
    }

    calls: list[dict] = []
    oks: list[bool] = []
    round1: list[str] = []
    round2: list[str] = []
    seen_result = False

    try:
        async for _namespace, payload in graph.astream(
            initial, config, stream_mode="custom", subgraphs=True
        ):
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "delta":
                (round2 if seen_result else round1).append(payload.get("text") or "")
            elif payload.get("type") == "tool":
                if payload.get("phase") == "start":
                    calls.append(
                        {
                            "name": payload.get("name"),
                            "args": payload.get("arguments") or {},
                            "id": payload.get("call_id"),
                        }
                    )
                elif payload.get("phase") == "end":
                    oks.append(bool(payload.get("ok")))
                    seen_result = True
        state = await graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001 — 评估里必须响亮失败
        raise RuntimeError(f"生产回合报错：{exc}") from exc

    messages = list((state.values or {}).get("messages") or [])
    tool_outputs = [
        message.content
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    fallback_text = (last_ai.text or "") if last_ai is not None else ""

    if calls:
        return Turn(
            calls,
            "\n\n".join(tool_outputs),
            oks,
            "".join(round1),
            "".join(round2) or fallback_text,
        )
    # 没调工具：首轮增量就是用户看到的全部正文；固定话术出口没有流式增量，
    # 用最终 State 里的 AI 消息补上（与 API 适配层一致）。
    return Turn([], "", [], "", "".join(round1) or fallback_text)


# ------------------------------------------------------------ 负对照开关


def apply_prompt_override() -> str | None:
    """负对照开关：`EVAL_PROMPT_FILE=<某个 prompts.py>` 时临时换掉 System Prompt。

    ★ 这是证明"闸门真的在守东西"的唯一办法（修复轮 · Important 1）：把修复前的
    Prompt 接进修复后的套件，它**必须**拿不到满分；否则闸门就是个摆设。
    只改内存里的模块属性（build_chat_prompt 调用时才读它），不碰磁盘上的
    app/prompts.py，进程退出即还原。judge 模板里嵌的仍是当前 SYSTEM_PROMPT——
    不影响，judge 已经不是闸门了。

    两个评估集都支持它，用法（旧 Prompt 用 git show 取到临时文件）：
        git show 18d2b45:app/prompts.py > /tmp/old_prompts.py
        EVAL_PROMPT_FILE=/tmp/old_prompts.py uv run python -m evals.run_evals
    """
    path = os.environ.get("EVAL_PROMPT_FILE")
    if not path:
        return None
    import app.prompts as prompts
    import app.workflow.agent as agent

    ns: dict = {}
    exec(Path(path).read_text(encoding="utf-8"), ns)  # noqa: S102 — 输入由调用者指定的负对照工具
    prompts.SYSTEM_PROMPT = ns["SYSTEM_PROMPT"]
    # Agent 子图在模块导入时 `from app.prompts import SYSTEM_PROMPT`，名字已
    # 绑定；负对照必须同时改这份引用，否则接旧 Prompt 的评估会是假绿。
    agent.SYSTEM_PROMPT = ns["SYSTEM_PROMPT"]
    return path


# ------------------------------------------------------------------ 测试库会话


def test_db_url() -> str:
    """把配置里的库名换成 spec §5.6 的 customer_service_test。

    不另写一个 URL：库名/账号/端口只有 app/config.py 一处真相，
    scripts/seed.py 建测试库时用的是同一个"换名"手法。
    """
    base = get_settings().db_url.rsplit("/", 1)[0]
    return f"{base}/customer_service_test?charset=utf8mb4"


@asynccontextmanager
async def open_test_db():
    """开一笔"跑完就回滚"的真实测试库事务，产出 (session, conversation_id)。

    conversation_id 是**先建好的一行 conversations**：create_ticket 往 tickets
    插行、外键指向 conversations，没有会话行就撞外键——工具回"执行失败"，
    评估测到的就变成"我忘了建会话"而不是模型行为。

    整段跑在一笔末尾回滚的事务里：建单、改会话状态都是真实写入，回滚后库里
    不留痕，可重复跑。
    """
    engine = create_async_engine(test_db_url(), echo=False)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            trans = await session.begin()
            try:
                conv = await conv_repo.get_or_create(
                    session, str(uuid.uuid4()), "eval"
                )
                yield session, conv.id
            finally:
                await trans.rollback()
    finally:
        await engine.dispose()


# ------------------------------------------------------------ 确定性转述判据

# 工具明明返回了数据、客服却说"看不到/查不到"——旧 Prompt 的典型失效模式。
REFUSAL_PATTERNS = ["无法查询", "没法帮您查", "查询不到您的"]

# 裸「看不到」单独处理（修复轮 · Minor 5）：它是**口语**，光凭它定罪会误伤
# 如实回复。实测要保护的那条诚实行为正是 fabricate-3 的回复——工具只回了订单
# 状态、没有退款字段，模型如实说"退款情况这边看不到"是对的。所以它只在回复
# **一条工具事实都没转述**时才算罪证：既没转述、又说看不到 = 典型拒绝。
BARE_REFUSAL_PATTERN = "看不到"

# 工具"没查到"时的返回文案。★ 拒绝措辞只在**工具确实返回了数据**时才是罪证：
# 工具自己说"没有找到/暂时不可用"时，模型回一句"这边查不到"是**如实转述**，
# 不是拒绝。不设这道闸，faq-miss 这类用例会把正确行为判成 FAIL。
# 这些措辞与工具实现耦合（见 tests/test_evals_harness.py 的钉子测试）。
NO_DATA_MARKERS = ("没有找到", "暂时不可用", "暂时查不到")

# 工具说办不到、客服却说办成了 —— 反向的失效模式（编造成功）。
SUCCESS_CLAIM_PATTERNS = [
    "已为您创建工单", "已创建工单", "已帮您建单", "已为您建单",
    "已为您转人工", "已转人工", "已帮您转接", "已经帮您转",
]

# ★ 从工具输出里抽出"具体事实"。判的都是可核对的硬事实：单号、金额、日期、
# 时刻、轨迹里的城市与状态词。不抽「整句相似度」——实测它分不开：旧 Prompt 的
# 拒绝语里"看不到订单的物流信息"与工具输出的"的物流信息"有 5 字公共子串。
_FACT_PATTERNS = (
    r"[A-Za-z]{1,2}\d{6,}",             # 运单号 / 工单号（Y767075701349、T20260920001）
    r"\d+\.\d{2}",                      # 金额（583.79）
    r"\d{4}-\d{2}-\d{2}",               # 日期
    r"\d{2}:\d{2}",                     # 时刻
    r"\d{2}:\d{2}\s+([一-龥]{2,4})",    # 轨迹行里的城市
    r"[：:]([一-龥]{2,4})\s*$",         # 行尾的状态/名称词（已签收、运输中、缺货…）
)

# 高精度事实：只用于**抓编造**（回复里出现、工具却没给过的原样 token）。
# 只认这几种不会被口语改写的形态，宁可漏抓也不误伤——把"今天/尽快"之类
# 当事实会让正确回复被判编造。
_STRICT_FACT_PATTERNS = (r"\d{4}-\d{2}-\d{2}", r"\d{2}:\d{2}", r"\d+\.\d{2}")

# FAQ 答案是**散文**，里面往往一个数字都没有，硬事实判据抽不到东西。
# 于是加一条散文回落：回复与工具输出的最长公共片段 ≥ 这个字数就算转述了。
# 取 6：中文里 6 个连续相同的字基本就是照搬了原句（"保持吊牌完整"）；
# 而拒绝语与答案的公共片段通常只有 2~3 字（"退货"）。实测值打印在失败原因里。
PROSE_MIN_RUN = 6


def _normalize(text: str) -> str:
    """去掉空白与标点，只留字/数字/CJK——比对的是内容，不是排版。"""
    return re.sub(r"\W+", "", text)


def extract_facts(tool_output: str) -> set[str]:
    found: set[str] = set()
    # ★ 「问：<章节名>」是 query_faq 的排版行，不是业务事实。ch03 的章节名
    # 常是 2–4 字（运费说明/退货政策/换货流程），会命中行尾状态正则
    # `[：:]([一-龥]{2,4})$`，把正确转述判成「没转述事实」。FAQ 答案是散文，
    # 应该走 prose 回落，而不是被问题的标题带进硬事实分支。
    text = "\n".join(
        line for line in tool_output.splitlines() if not line.strip().startswith("问：")
    )
    for pattern in _FACT_PATTERNS:
        for m in re.finditer(pattern, text, re.MULTILINE):
            found.add(m.group(1) if m.groups() else m.group(0))
    return found


def longest_common_run(a: str, b: str) -> int:
    """两段文本的最长公共子串长度（先归一化）。"""
    na, nb = _normalize(a), _normalize(b)
    return SequenceMatcher(None, na, nb).find_longest_match(0, len(na), 0, len(nb)).size


def fabricated_facts(reply: str, tool_output: str) -> list[str]:
    """回复里出现、工具输出里没有的高精度事实（日期/时刻/金额）。

    ★ 已知取舍：只能抓"原样照抄"式的编造。模型把日期改写成"9 月 21 日"就抓
    不到（_STRICT_FACT_PATTERNS 不认）。这是**故意**的——这条判据只做闸门，
    宁可漏抓也不容许因为改写口径把正确回复误判成编造。
    """
    return sorted(
        {
            m.group(0)
            for pattern in _STRICT_FACT_PATTERNS
            for m in re.finditer(pattern, reply)
            if m.group(0) not in tool_output
        }
    )


def returned_data(tool_output: str) -> bool:
    """工具是否真的返回了数据（而不是"没查到"或"不可用"）。"""
    return not any(m in tool_output for m in NO_DATA_MARKERS)


@dataclass(frozen=True)
class RelayVerdict:
    ok: bool
    why: str


def relay_check(
    tool_output: str, reply: str, *, prose_min: int = PROSE_MIN_RUN
) -> RelayVerdict:
    """确定性转述判据。返回 (是否如实转述, 失败原因)。

    分支：
    - 工具返回了数据 → 回复里必须出现工具给过的具体事实（单号/金额/日期/轨迹
      城市…）；**只有**工具输出里抽不到任何硬事实时（FAQ 散文答案）才回落成
      "与工具输出有 ≥ prose_min 字的公共片段"。且不得用"看不到"这类拒绝措辞，
      也不得出现工具没给过的日期/时刻/金额；
    - 工具没返回数据（查不到 / 库没绑）→ 回复里不得**声称办成了**。工具说办不到
      而客服说"已为您转人工"，就是编造。
    """
    if not reply.strip():  # 空回复不能"真空通过"（模型只调工具没给用户任何话）
        return RelayVerdict(False, "最终回复为空")

    if returned_data(tool_output):
        facts = extract_facts(tool_output)
        relayed = any(f in reply for f in facts)

        # 拒绝措辞：明确的那几条无条件算；裸「看不到」只在"一条事实都没转述"
        # 时才算（见 BARE_REFUSAL_PATTERN 处的说明）——否则会把如实回复判死。
        hit = [p for p in REFUSAL_PATTERNS if p in reply]
        if not relayed and BARE_REFUSAL_PATTERN in reply:
            hit.append(BARE_REFUSAL_PATTERN)
        if hit:
            return RelayVerdict(False, f"工具返回了数据却回拒绝措辞 {hit}")

        if facts:
            # ★ 硬事实抽得到时**不许**回落散文判据（修复轮 · Minor 6）：
            #   回落只看最长公共片段，而工具输出的**表头**（"物流信息""运单号"）
            #   就能凑够 6 字——一条只复读表头、什么事实都没给的回复会蒙混过关。
            #   有硬事实可对账，就只认真事实。
            if not relayed:
                return RelayVerdict(
                    False,
                    f"没有转述工具给出的任何具体事实（工具给了 {sorted(facts)[:4]}…）",
                )
        else:
            # 工具输出里抽不到任何硬事实（FAQ 散文答案）→ 才走散文回落。
            run = longest_common_run(tool_output, reply)
            if run < prose_min:
                return RelayVerdict(
                    False,
                    f"没有转述工具输出的任何内容（工具输出里抽不到硬事实；"
                    f"最长公共片段仅 {run} 字，下限 {prose_min}）",
                )
        fake = fabricated_facts(reply, tool_output)
        if fake:
            return RelayVerdict(False, f"回复里出现工具没给过的具体事实 {fake}")
        return RelayVerdict(True, "")

    claim = [p for p in SUCCESS_CLAIM_PATTERNS if p in reply]
    if claim:
        return RelayVerdict(False, f"工具没办成却声称办成了 {claim}")
    return RelayVerdict(True, "")
