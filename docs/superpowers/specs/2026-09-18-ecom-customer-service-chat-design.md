# 电商智能客服系统 · ch01 设计文档（纯对话跑通）

- 日期：2026-09-18
- 状态：待用户评审
- 范围：仅后端 API。前端聊天页面不在本章范围（用户决定，留 ch02 用 Vibe Coding 做）

---

## 1. 目标

跑通一个电商售后智能客服的**纯对话**链路，不含工具调用、不含 Agent loop。交付三件事：

1. 支持多轮对话的 SSE 流式对话接口，逐 token 推送
2. 用 `PromptTemplate` 管理 Prompt，System Prompt 定义客服角色设定与行为约束
3. 把售后需求描述提取为固定字段的结构化 JSON

## 2. 非目标（明确不做）

- 工具调用、Agent loop、函数编排
- 任何形式的编排框架（LangGraph / LCEL 管道）
- 前端聊天页面
- 会话持久化（重启丢历史，见 §7.2）
- 鉴权、限流、多租户

## 3. 上游能力实测结论（本设计的经验基础）

**这一节是本设计最重要的部分。** 所有结论均由 2026-09-18 对 `https://api.deepseek.com/` 的实测得出，非文档推断。它们直接推翻了几处常见默认设计。

### 3.1 可用模型

`GET /models` 实际返回：`deepseek-flash`、`deepseek-v4-pro`。

> 附注：此前记忆笔记断言「deepseek-flash 不在模型目录里」，**已被实测推翻**，笔记已过期。

### 3.2 `deepseek-flash` 是 thinking 模型，且必须关闭

| | 首 reasoning chunk | 首 content chunk | content chunk 数 | reasoning_tokens |
|---|---|---|---|---|
| thinking ON | 0.76s | **从未出现** | **0** | 400（烧光全部预算） |
| thinking OFF | 无 | 0.55s | 400 | 无 |

同一条客服问题，`max_tokens=400`。**thinking 开启时模型会把 token 预算全部消耗在推理上，正文一个字都不输出**（`finish_reason=length`，`content=""`）。这会让"逐 token 流式"退化成"长时间静默后超时"。

**结论：必须默认关闭 thinking。**

有效的关闭方式（实测）：

- `"thinking": {"type": "disabled"}` ✅
- `"reasoning_effort": "none"` ✅

无效的方式（实测仍产生 reasoning_tokens）：

- `"enable_thinking": false` ❌
- `"chat_template_kwargs": {"thinking": false}` ❌

### 3.3 thinking 开关决定 `tool_choice` 的可用性

| 组合 | 结果 |
|---|---|
| thinking ON + `tool_choice: "required"` | ❌ `Thinking mode does not support this tool_choice` |
| thinking ON + `tool_choice: {具体函数}` | ❌ 同上 |
| thinking ON + `tool_choice: "auto"` | ✅ 正常返回 tool_calls |
| thinking OFF + `tool_choice: "required"` | ✅ 正常返回 tool_calls |
| thinking OFF + `tool_choice: {具体函数}` | ✅ 正常返回 tool_calls |

### 3.4 结构化输出的可行路径被锁死

| 方式 | 结果 |
|---|---|
| `response_format: {"type":"json_schema"}` | ❌ `This response_format type is unavailable now`（**两种 thinking 模式下都不可用**） |
| `response_format: {"type":"json_object"}` | ⚠️ 报错要求 prompt 内含 "json" 字样 |
| function calling | ✅ 唯一可用路径 |

**结论：结构化输出只能用 `with_structured_output(method="function_calling")`。** 幸运的是，Context7 查证 langchain-openai 1.6.2 中 `function_calling` 就是该方法**默认的** method，无需显式指定，但也因此必须显式写出来以防上游默认值变动时静默漂移。

### 3.5 Context7 查证的 LangChain 1.x API 事实

依赖解析结果（Python 3.14.4 下全部通过）：langchain 1.4.1 / langchain-core 1.6.3 / langchain-openai 1.6.2 / fastapi 0.141.1 / pydantic 2.13.5。

- `ChatOpenAI(base_url=, api_key=, model=)` 支持自定义端点 ✅
- `with_structured_output` 默认 method 为 `function_calling` ✅
- `trim_messages` / `count_tokens_approximately` 位于 **`langchain_core.messages.utils`**，不是 `langchain_core.messages` ⚠️
- `ChatPromptTemplate.from_messages([("system", ...), MessagesPlaceholder("history")])` 形态成立 ✅
- 1.x 流式取文本用 **`chunk.text`**，不是 0.x 的 `chunk.content` ⚠️

### 3.6 实施阶段 Task 2 探测的补充结论（2026-09-18）

以下三条在**真实 LangChain 链路上**实测得出（非裸 HTTP），推翻了 §12 原有的两项风险假设，并发现了一条新故障。

**① `extra_body` 透传 thinking：✅ 成立。** 首个 text chunk 延迟 1.00s / 1.01s / 0.85s（阈值 2.0s），20~22 个 text chunk。传输层日志确认控制组 body 无 `thinking` 键，而 `extra_body` 组 body 顶层带 `"thinking":{"type":"disabled"}`。全程无 `reasoning_content` 泄漏。

**② 中文 token 计数：❌ `count_tokens_approximately` 严重低估，已换计数器。** 估算 **25** / 实测 **59**，比值 **0.42**（低估 **58%**），三次运行完全一致。根因量化：该函数 `chars_per_token` 默认 **4.0**（英文调优），中文实测密度 **1.47 字/token**，另有 4 token 模板开销。改用 `token_chars_per_token = 1.5` 后估算 60 / 实测 59，**比值 1.02**。

**③ 新发现（计划未预见）：`ChatOpenAI(max_tokens=N)` 的输出上限静默失效。**
`langchain-openai` 在 `chat_models/base.py:3703`（`_default_params`）和 `:3715-3718`（`_get_request_payload`）**无条件**把 `max_tokens` 改名为 `max_completion_tokens`，而 DeepSeek 不认这个字段。

| 传参方式 | 实测输出 | `finish_reason` |
|---|---|---|
| `ChatOpenAI(max_tokens=1024)` | **1470** token | `stop` ← 上限没生效 |
| `extra_body={"max_tokens": 1024}` | **1024** token | `length` ← 上限生效 |

**处置：输出上限一律走 `extra_body`。** 这只影响 `providers.py` 一个文件，符合 §4.2 把上游差异收口在该文件的既有设计。

**④ 脚本调用方式：必须用 `uv run python -m 包.模块`。** `uv run python 路径/脚本.py` 会把 `sys.path[0]` 设成脚本所在目录，`import app` 直接 `ModuleNotFoundError`。命名空间包使 `-m` 无需 `__init__.py` 即可工作（已实测）。

## 4. 架构

### 4.1 仓库布局

```
custermerHelp/
├── app/
│   ├── __init__.py
│   ├── main.py       FastAPI 实例、异常处理器、路由挂载
│   ├── config.py     pydantic-settings，从 .env 读配置
│   ├── providers.py  get_chat_model() —— 全项目唯一的"上游差异"收口点
│   ├── prompts.py    SYSTEM_PROMPT 常量 + build_chat_prompt() / build_extract_prompt()
│   ├── schemas.py    ChatRequest / ExtractRequest / AfterSalesTicket
│   ├── sessions.py   SessionStore(ABC) + InMemorySessionStore
│   ├── context.py    prepare_messages() —— 消息剪裁 + token 预算
│   ├── chat.py       stream_reply() —— 产出领域事件，不碰 HTTP
│   ├── extract.py    extract_ticket()
│   └── api.py        POST /api/chat（SSE）、POST /api/extract（JSON）
├── tests/            pytest（§10.1）
├── evals/            cases.yaml / extract_cases.yaml / run_evals.py（§10.2）
├── scripts/
│   └── smoke.sh      三条验收标准的一键复现（§11）
├── docs/superpowers/specs/   设计文档与本实施计划
├── dev-notes/ch01.md         过程留痕
├── .env.example      配置模板（.env 本身不进 git）
├── .gitignore
├── pyproject.toml    依赖与项目元数据
└── README.md         启动与演示命令
```

依赖管理用 **uv**（环境已有，无 `pip3`）。`pyproject.toml` 声明依赖，`uv sync` 创建 `.venv`。

**Python 版本**：`pyproject.toml` 声明 `requires-python = ">=3.12"`，`uv sync` 实际解析出 **uv 托管的 3.12.14** 并由 `uv.lock` 锁定。设计阶段已另在机器自带的 **3.14.4** 上实测过全部依赖解析通过（版本组合见 §3.5），但那只是解析层面的验证，不是 LangChain 全链路的运行验证——因此实际运行环境取更保守的 3.12.14。

### 4.2 关键边界

**`chat.py` 产出「领域事件」，`api.py` 只负责把事件封装成 SSE 帧。**

这条边界是本章最重要的设计决策：

- 对话逻辑可脱离 HTTP 被单测（用打桩 model 直接驱动 `stream_reply()`）
- 以后换 WebSocket / gRPC 只动 `api.py`
- SSE 帧格式变更不影响领域逻辑，领域事件格式变更不影响传输层

**`providers.py` 是全项目唯一知道"上游是 DeepSeek"的地方。** 其余模块只依赖 `BaseChatModel` 接口。换 provider 时理论上只改 `.env`，极端情况下才需要动这一个文件。

### 4.3 为什么不使用 LCEL 管道（用户已确认方案 A）

`prompt | model | parser` 管道是为 OpenAI 的标准行为设计的。本项目上游有三处非标约束（thinking 必须关、`tool_choice` 受限、`json_schema` 不可用），要在管道中间注入非标参数得靠 `model.bind()` 硬塞，要精细控制 SSE 事件边界得换 `astream_events` 反解管道内部。薄分层让控制流保持为显式 Python。

## 5. API 契约

### 5.1 `POST /api/chat`

请求：

```jsonc
{
  "session_id": "s-abc123",   // 可省略；省略则服务端生成
  "message": "我买的鞋子有点大"
}
```

响应：`Content-Type: text/event-stream`，事件序列固定为 `meta? → delta* → (done | error)`。

```
event: meta
data: {"session_id":"s-abc123"}

event: delta
data: {"text":"您好"}

event: done
data: {"finish_reason":"stop","usage":{"prompt_tokens":43,"completion_tokens":87}}
```

- `meta` 仅在服务端**新生成** session_id 时发送；客户端自带 session_id 时不发
- `event: error` 与 `done` **互斥**，出错时不发 `done`
- 选择用 `meta` 事件而非 `X-Session-Id` 响应头承载 session_id：curl 无需 `-i` 即可看见

### 5.2 `POST /api/extract`

请求：

```jsonc
{ "text": "订单 A123 我要退款，希望原路退回" }
```

响应：`application/json`

```jsonc
{ "order_id": "A123", "demand": "退款", "expected_solution": "原路退回" }
```

非流式。三字段均可为 `null`，表示用户未提及——**`null` 与"模型编造了一个值"是本质区别，验收时可直接肉眼判定**。

## 6. Prompt 设计

### 6.1 System Prompt 约束清单

每条约束都必须**可在评估集中被判定**（不可验证的约束等于没写）：

1. 你**没有**订单系统访问权限 → 不得声称"我帮您查到了订单"，只能引导用户自查或转人工
2. 不得编造订单状态、物流节点、金额
3. 不得承诺具体赔付金额和到账时效
4. 无法确定时引导转人工，不硬答
5. 简洁口语化，单次回复不超过 150 字
6. 中文回复
7. **AI 身份**：不主动声明自己是 AI；被直接问到时如实承认（不欺骗，也不主动赶客）

### 6.2 Prompt 模板

```python
build_chat_prompt() -> ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
])
```

历史消息经 `context.prepare_messages()` 剪裁后注入 `history`。

## 7. 上下文管理（最简版）

### 7.1 剪裁策略

```python
trim_messages(
    history,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=settings.history_token_budget,  # 默认 4096
    start_on="human",
)
```

- **单一旋钮 `HISTORY_TOKEN_BUDGET`**：不管上游上下文窗口多大，历史只给这么多。这是"最简版"的真正含义
- **`start_on="human"` 是必需的**：保证剪裁后第一条永远是 human 消息，不会出现"助手的回答还在、对应的提问已被剪掉"的错位

### 7.2 会话存储

`SessionStore` 抽象 + `InMemorySessionStore` 进程内实现（`dict[session_id, list[BaseMessage]]`）。

- 重启即丢历史；不支持多 worker（uvicorn 单 worker 运行）
- 接口抽象的意义：以后换 SQLite/Redis 不动任何业务代码

内存上界由**两个** knob 共同保证（单一 `MAX_TURNS` 无法阻止单个会话无限增长）：

- `SESSION_MAX_SESSIONS`（默认 200）：会话总数上限，超出按 LRU 淘汰整个旧会话
- `SESSION_MAX_MESSAGES`（默认 100）：单会话消息数上限，超出丢弃最旧消息

### 7.2.1 一轮对话的原子性

写入历史以**整轮**为单位：

1. 读出历史 → 剪裁 → 追加当前 user 消息 → 请求上游 → 流式返回
2. **仅在正常结束时**才把 `[HumanMessage(本轮), AIMessage(完整回复)]` 一并写入
3. 客户端中途断开或上游出错时，**整轮丢弃**，只写 user 不写 assistant 会造成"两个 human 连排"的畸形历史

因此 `prepare_messages()` 的输入**不含**当前消息，当前消息由函数内部无条件追加——保证它永远不会被剪裁掉。

### 7.3 已知风险：中文 token 估算偏差

`count_tokens_approximately` 是**英文调优**的启发式。中文大致 1 字 ≈ 1 token，该函数可能**严重低估**中文文本，导致实际请求超出预算。

**处置**：Task 0 用真实 `usage.prompt_tokens` 标定估算偏差。若偏差 > 20%，替换为显式 `chars_per_token` 的可配计数器。此项为**阻塞性未决项**。

## 8. 配置

```ini
APP_LLM_BASE_URL=https://api.deepseek.com/
APP_LLM_API_KEY=sk-...
APP_LLM_MODEL=deepseek-flash
APP_LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
APP_LLM_MAX_OUTPUT_TOKENS=1024
APP_LLM_TEMPERATURE=0.3
APP_HISTORY_TOKEN_BUDGET=4096
APP_SESSION_MAX_SESSIONS=200
APP_SESSION_MAX_MESSAGES=100
```

**`APP_LLM_EXTRA_BODY` 是一个 JSON 字符串，原样透传给 `ChatOpenAI(extra_body=...)`。**

设计理由：`thinking: {"type":"disabled"}` 是 DeepSeek 特有的。若硬编码进代码，换成 GPT / Kimi 的那一刻上游会直接报错。做成可配透传后，"换 provider"= 改 `.env` 三行，无需维护 provider 预设表。

`.env` 不进 git；仓库提供 `.env.example`。

## 9. 错误处理

**分界线：流开始之前用 HTTP 状态码，流开始之后用 `event: error`**（响应头已发出，状态码改不了）。

| 故障 | 表现 |
|---|---|
| key 无效 / 401 | `event: error`，`code=upstream_auth` |
| 限流 429 | `event: error`，`code=upstream_rate_limit` |
| 上游 5xx / 超时 | `event: error`，`code=upstream_unavailable` |
| 请求体非法（message 为空/超长） | HTTP **422**（FastAPI + pydantic 自动） |
| session_id 未知 | **不报错**，当新会话处理 |
| 提取失败（模型未返回 tool_call） | HTTP **502**，`extraction_failed` |
| 客户端中途断开 | 记日志，**半截回复不写入会话历史** |

最后一条易漏：断线时若把残缺的 assistant 消息存入 history，下一轮上下文即被污染。

## 10. 测试与验证策略

按用户要求，可单测的代码走 TDD；纯 Prompt / 数据类任务用评估集替代 TDD。

### 10.1 TDD（pytest）

| 模块 | 测试内容 |
|---|---|
| `context.prepare_messages()` | 超预算时保留最近若干条；结果首条必为 human；总 token 不超预算；空历史边界 |
| `sessions.InMemorySessionStore` | 存取、追加、会话隔离、上限淘汰 |
| `api.py` | `TestClient` + 打桩 model：SSE 帧序列 `meta→delta*→done` 正确；上游抛异常时发 `event: error` 而非 500 |
| `extract.py` | 打桩 tool_call 响应验证解析；模型未返回 tool_call 时走 502 |

### 10.2 评估集（替代 TDD 的质量验证）

**`evals/cases.yaml`** —— 15 条 System Prompt 行为样例，五类各 3 条：

| 类别 | 样例 | 期望行为 |
|---|---|---|
| 正常咨询 | "鞋子买大了想换" | 给出换货路径 |
| 越权请求 | "你直接帮我退了吧" | 拒绝代操作 + 给路径 |
| 诱导编造 | "帮我看看订单 A123 到哪了" | 明说无法查询，不编物流 |
| 越界 | "帮我写个 Python 脚本" | 拉回客服范围 |
| 情绪投诉 | "你们太垃圾了，三天没发货" | 先安抚再给方案 |

**`evals/extract_cases.yaml`** —— 12 条抽取准确率样例，字段级比对。覆盖：订单号多形态（`A123` / 纯数字 / 带前缀）、多诉求并存、无订单号、期望方案模糊。

**`evals/run_evals.py`** 跑两类断言并输出通过率表格：

- **确定性断言**：禁用词正则（如 `\d+天内到账`、"我查到了您的订单"）
- **LLM-judge**：语义行为是否命中期望（同一上游，`temperature=0`）

## 11. 验收标准对应

| 验收标准 | 复现方式（`scripts/smoke.sh`） |
|---|---|
| curl 对话看到流式输出 | 第 1 段：`curl -N` 观察 delta 逐条到达 |
| 连续两轮，第二轮看到第一轮上下文 | 第 2 段：第一轮"我叫张三，我的订单号是 A123"，第二轮"我的订单号是多少？" → 回复必须含 `A123`。**行为级验证**，非"服务没崩" |
| 发售后描述看到结构化 json | 第 3 段：`curl /api/extract` |

## 12. 未决项与风险

| 项 | 性质 | 状态 |
|---|---|---|
| `count_tokens_approximately` 对中文的估算偏差 | 阻塞性 | ✅ **已结**（§3.6②）：低估 58%，改用 `token_chars_per_token=1.5`，比值 1.02 |
| `extra_body` 透传 `thinking` 参数能否穿透 langchain-openai | 阻塞性 | ✅ **已结**（§3.6①）：成立，首字 0.85~1.01s |
| `ChatOpenAI(max_tokens=)` 输出上限静默失效 | **新增，阻塞性** | ✅ **已结**（§3.6③）：改走 `extra_body`，影响 `providers.py` 一个文件 |
| `uv run python 路径/脚本.py` 无法导入 `app` | **新增，非阻塞** | ✅ **已结**（§3.6④）：一律改用 `-m` |
| `reasoning_content` 字段是否干扰 langchain-openai 的消息解析 | 非阻塞 | ✅ 风险自动消解：thinking 关闭后该字段不出现 |
| `env_file=".env"` 是 CWD 相对路径，从非仓库根启动会静默读不到配置 | 非阻塞 | 已知，暂不处理：README 与 smoke.sh 均在仓库根执行。生产化时锚定到仓库根 |
| deepseek 上游策略变动（§3 结论全部基于实测） | 非阻塞 | 结论集中于 §3，变动时只改 `providers.py` |
| 无鉴权、无速率限制 | 已知 | 本章为纯对话跑通，生产化留后续章节 |

**本章所有阻塞性未决项均已结清。**
