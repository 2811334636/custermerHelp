# 电商智能客服系统 · ch02 设计文档（Function Calling 工具链）

- 日期：2026-09-20
- 状态：待用户评审
- 前置：ch01（纯对话跑通）已交付并合入 `main`

---

## 1. 目标

给现有客服系统装上**查数据的能力**：用 Function Calling 让模型自己决定调什么工具，工具链长在客服聊天本身，用户在聊天页问一句就能触发。

交付**六**件事：

1. FastAPI + SQLAlchemy 的**分层骨架**，Docker 起 MySQL，四张表 + 测试数据
2. 五个用 LangChain `@tool` 装饰器定义的业务工具
3. 工具基础设施：注册管理、参数 Schema 校验、执行错误处理、超时重试
4. **改写 System Prompt**（见 §6.4）——ch01 的"你没有订单系统访问权限"与本章功能直接冲突，不改则验收 1 必然失败
5. 工具链并入现有 SSE 聊天入口，最终回答仍逐 token 流式吐出，工具执行阶段先推状态帧
6. 聊天页在气泡里显示本轮调了哪个工具（工具轨迹徽章）

## 2. 非目标（明确不做）

- **多轮自动循环的 Agent Loop**（本章只做单轮调用，见 §7.4）
- 向量检索、RAG（`faq` 表的检索本章仍是 SQL `LIKE`）
- 真实对接公司电商系统 / 物流系统 API（`query_order` / `query_product` / `query_logistics` 返回工具内确定性生成的假数据）
- 刷新页面恢复历史对话（见 §14.1，数据已落库，缺的是读接口）
- 鉴权、限流、多租户、会话超时自动结束

## 3. 前置条件与环境

- **Docker 必须可用**。Docker Desktop 装在 Windows 侧，需在 `Settings → Resources → WSL Integration` 中开启当前发行版的集成，否则 WSL 内 `docker` 命令不可达（已实测确认本章开始时为不可用状态，后由用户开启）
- 端口 3306（MySQL）、8000（应用）需空闲
- 沿用 ch01 的上游配置：`https://api.deepseek.com/` + `deepseek-flash`，**thinking 必须关闭**（见 ch01 spec §3.2，开着会正文零输出）

## 4. 架构与分层

### 4.1 目录结构

```
app/
├── db/                        ← 新增：数据层（只管连接与表结构）
│   ├── __init__.py
│   ├── base.py                Base、async engine、AsyncSessionLocal、get_db() 依赖
│   ├── models.py              四张表的 ORM 模型
│   └── messages_io.py         ★ 唯一的 LangChain 消息 ↔ 表行 转换层
│
├── repositories/              ← 新增：数据访问层（只管读写，不含业务判断）
│   ├── __init__.py
│   ├── faq.py                 search_by_keyword(keyword, limit)
│   ├── conversation.py        get_or_create / set_status
│   ├── message.py             append / list_by_conversation
│   └── ticket.py              create / get_by_no / count_today
│
├── tools/                     ← 新增：工具层（业务能力，调用 repositories）
│   ├── __init__.py            build_registry()
│   ├── registry.py            ★ 工具基础设施（§6.2）
│   ├── context.py             ToolContext + ContextVar，用于注入模型看不到的上下文
│   ├── catalog.py             query_order / query_product / query_logistics
│   ├── knowledge.py           query_faq
│   └── ticketing.py           create_ticket
│
├── sessions.py                ← 改：ABC 改 async，新增 MySQLSessionStore
├── chat.py                    ← 改：加"工具轮"（§7）
├── api.py                     ← 改：新 SSE 事件 + 挂载静态页
└── （config.py / context.py / extract.py / main.py / prompts.py /
      providers.py / schemas.py 不动）

web/
└── index.html                 ← 新增：聊天页（Vibe Coding，不套流程）

docker-compose.yml             ← 新增：MySQL 8.4
scripts/schema.sql             ← 新增：建表 DDL（用户提供，§5）
scripts/seed.py                ← 新增：灌种子数据
evals/tool_cases.yaml          ← 新增：工具选择评估集（§12.2）
```

### 4.2 四层职责边界

| 层 | 只做 | 绝不做 |
|---|---|---|
| `db/` | 定义表和连接 | 任何查询逻辑 |
| `repositories/` | 拼 ORM 查询、返回数据 | 判断"该不该建工单"这类业务规则 |
| `tools/` | 业务语义、调 repositories、返回给模型看的文本 | 直接写 `session.execute(...)` |
| `chat.py` | 编排模型调用与工具执行 | 知道 SQL 或表结构 |

**边界靠一句话判断：这一层"知道"什么？**

| 层 | 它知道 | 它不知道 |
|---|---|---|
| `db/` | 表长什么样、怎么连 | 什么叫"物流" |
| `repositories/` | 怎么把 faq 表按关键词捞出来 | 用户问这句话是什么意思 |
| `tools/` | "查物流"这个业务动作要什么参数、返回给模型看什么 | SQL 怎么写、表叫什么 |
| `chat.py` | 模型和工具之间怎么传话 | 世界上有 MySQL 这个东西 |

**为什么要这么切**——为了换东西时只改一层：

- `query_faq` 从 `LIKE` 换成向量检索 → 只改 `tools/knowledge.py` + 加一个 repository，`chat.py` 一行不动
- MySQL 换 PostgreSQL → 只改 `db/` 和 repositories 的方言，`tools/` 一行不动
- 单测 `create_ticket` → 不用起 FastAPI、不用起模型，直接给 repository 一个测试 session

### 4.3 为什么不用 LangChain 的 `create_agent`

`create_agent` 会拉回 `langgraph` 整套编排栈——ch01 刚因为 spec §2 把它列为非目标而从依赖里删除。且它内建的 agent 循环与本章"只做单轮"的目标相反，用 `max_iterations=1` 限制是逆着工具的设计意图走。

替代方案见 §7.4：**第二次模型调用不绑工具**，从调用方式上结构性保证单轮收敛。

## 5. 数据库设计

### 5.1 DDL（用户提供）

以下 DDL 由用户提供，**除 §5.2 标注的一列外逐字采用**。全库 `ENGINE=InnoDB`、`CHARSET=utf8mb4`。

```sql
-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE conversations (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  session_uuid VARCHAR(36)     NOT NULL                COMMENT '对外会话标识(ch02 新增)',
  user_id      VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status       ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_session_uuid (session_uuid),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```

### 5.2 对用户 DDL 的唯一改动

`conversations` 增加一列 `session_uuid VARCHAR(36) NOT NULL` + `UNIQUE KEY uk_session_uuid`。

**理由**：ch01 的 API 契约里 `session_id` 是**服务端生成的 UUID 字符串**（经 SSE `meta` 帧下发给前端），而 `conversations.id` 是 `BIGINT AUTO_INCREMENT`，两者对不上号。加这一列后：

- 对外契约（`session_id` = UUID）、现有 74 个测试、前端逻辑**全部不用改**
- 内部外键仍用自增 `id`（`messages` / `tickets` 引用它，索引更小）
- UUID 不可枚举——自增 id 可被猜测，改一个数字就可能读到别人的会话（本章无鉴权，风险不会实际暴露，但结构上不该留这个口子）

### 5.3 建表时机

`scripts/schema.sql` 挂载为 MySQL 容器的 `/docker-entrypoint-initdb.d/01-schema.sql`，**容器首次启动时自动执行**。

### 5.4 `conversations.user_id` 的数据来源

DDL 里 `user_id VARCHAR(64) NOT NULL` 是必填，但 **ch01 的 `ChatRequest` 只有 `session_id` + `message`，没有任何用户标识**。本章也不做鉴权。所以这一列需要一个明确的来源，否则插入会失败。

**决策：固定写入 `'anonymous'`，并把该值提为配置项。**

```ini
APP_DEFAULT_USER_ID=anonymous
```

**理由**：本章无鉴权，"用户是谁"这个事实系统无法得知；写一个诚实的占位值，比伪造一个看起来像用户名的东西好。提为配置项是为了让"以后接入登录"时只改 `.env` 与一处理赋值，不改表结构。

**副作用（记入 §14 已知限制）**：所有会话的 `user_id` 都是 `anonymous`，所以 `idx_user_id` 这个索引在本章没有区分度。它仍然值得保留——那是为有了真实用户之后准备的。

### 5.5 `query_faq` 检索哪一列

**只检索 `question` 列**（`WHERE question LIKE '%kw%'`），不检索 `answer`。

**理由**：验收 3 要求「邮费」查不到。如果连 `answer` 一起搜，某条答案里顺带提到"邮费自理"就会命中——**那会让这条验收的结论变得依赖种子数据的措辞细节**，测的就不是检索能力而是数据巧合了。只搜 `question` 让这个结果确定且可解释。

### 5.6 测试库

`scripts/seed.py` 除灌种子数据外，**还负责创建测试库** `customer_service_test` 并执行同一份 `scripts/schema.sql`。测试 fixture 连这个库，每个用例在事务里跑完回滚。这样"表结构只有一份真相"，不会出现生产库和测试库结构漂移。

### 5.7 种子数据（`scripts/seed.py`）

| 表 | 灌什么 |
|---|---|
| `faq` | 8~12 条常见问答，覆盖：退货政策、换货流程、发票、发货时效、价保、尺码建议 |
| `conversations` / `messages` / `tickets` | **不灌**——由真实对话产生 |

> **★ `faq` 的种子数据里故意不含「邮费」「运费」相关条目。**
>
> 这是**设计要求，不是遗漏**。验收标准 3 要求「邮费是多少」查不到并把这个漏召回记录下来，作为下一步升级的输入。**不要为了让验收 3 通过而补数据**——那会把要暴露的问题掩盖掉。

## 6. 工具设计

### 6.1 五个工具

```python
# app/tools/catalog.py —— 三个假数据工具共享一个确定性生成器
def _rng(key: str) -> random.Random:
    """同一个 key 永远得到同一串数据。"""
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))

@tool
async def query_order(order_id: str) -> str:
    """按订单号查询订单详情（商品、下单时间、金额、订单状态）。"""

@tool
async def query_product(product_id: str) -> str:
    """按商品号查询商品信息（名称、价格、规格、库存）。"""

@tool
async def query_logistics(order_id: str) -> str:
    """按订单号查询物流轨迹（承运商、运单号、若干条轨迹节点）。"""

# app/tools/knowledge.py
@tool
async def query_faq(keyword: str) -> str:
    """按关键词检索常见问答库。keyword 请传『核心业务词』，
    例如用户问『退货政策是什么』传『退货』。"""

# app/tools/ticketing.py
@tool
async def create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"]) -> str:
    """创建人工工单，返回工单号。仅当用户明确要求转人工时调用。"""
```

**三个设计点：**

**① 确定性假数据（`_rng`）**：`query_logistics("1001")` 每次返回同一份轨迹，不同订单号返回不同数据。**这是为了让验收可做**——模型说"已到杭州中转站"，你能去工具结果里查到同一句。纯随机会让验收退化成"看着像对"。

**② `ticket_type` 用 `Literal` 而非 `str`**：LangChain 会翻译成 JSON Schema 的 `enum`，模型看到三个合法值就不会编第四个。正好对上 DDL 的 `ENUM('售后','投诉','咨询')`。

**③ `query_faq` 的描述里教模型抽关键词**：这是让「退货政策是什么」能查到、「邮费是多少」查不到的关键。**不做任何模糊匹配兜底**——漏召回是预期要暴露的结果（§5.7）。

### 6.2 注册表与执行包装（`app/tools/registry.py`）

```python
@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool              # LangChain @tool 产出的对象
    timeout_s: float = 3.0
    max_retries: int = 2        # 首次 + 2 次重试 = 最多 3 次

@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    ok: bool
    content: str                # 回灌给模型的文本（成功是业务结果，失败是错误说明）
    summary: str                # 给前端徽章 tooltip 的一句话
    elapsed_ms: int

class ToolRegistry:
    def as_langchain_tools(self) -> list[BaseTool]          # 给 model.bind_tools()
    def get(self, name: str) -> ToolSpec                    # 名字不存在 → 明确报错
    async def execute(self, name, raw_args, call_id, ctx) -> ToolOutcome
```

`execute()` 的四项职责，逐项对应需求 3：

| 步骤 | 做什么 | 失败时 |
|---|---|---|
| **查注册表** | 名字 → `ToolSpec` | 名字不存在 → 回灌"没有这个工具" |
| **Schema 校验** | `args_schema.model_validate(raw_args)` | 参数不合法 → **不重试**，回灌校验错误原文 |
| **超时** | `asyncio.wait_for(..., timeout_s)` | 超时 → 重试 |
| **重试** | 只对超时/连接类故障重试 | 用尽 → 回灌"查询超时" |

**为什么参数错误不重试**：重试一次得到的还是同样的非法参数，纯浪费 3 秒。

**重试只包住工具函数本身**，不包住模型调用（模型调用有自己的重试，在 `providers.py` 的 `max_retries=2`）。

### 6.3 上下文注入（`app/tools/context.py`）

`create_ticket` 的签名里**没有**会话 id——模型不该知道、也不该能指定往哪个会话建单。

```python
@dataclass(frozen=True)
class ToolContext:
    conversation_id: int

_ctx: ContextVar[ToolContext] = ContextVar("tool_context")

@contextmanager
def tool_context(conversation_id: int): ...
def current_context() -> ToolContext: ...
```

`registry.execute()` 执行前把当前会话 id 塞进 `ContextVar`，`create_ticket` 从中取出。

**两个好处**：模型**不可能**把工单建到别的会话里（它连这个参数都没有）；签名语义干净——"描述"和"类型"是用户告诉模型的，"哪个会话"是系统知道的。

### 6.4 ★ System Prompt 必须改写（实测发现，原设计漏掉）

**ch01 的 System Prompt 与 ch02 的功能直接冲突。**

ch01 写的是：

> 你**没有**订单系统的访问权限。你看不到任何订单状态、物流信息、金额、收货地址。
> 因此你**绝对不能**说"我帮您查到了""我看到您的订单显示…"这类话。

**实测后果**（带 ch01 的 System Prompt，问「订单 1001 的物流到哪了」）：

```
第一个非空增量: TEXT
正文: '抱歉，我这边看不到订单的物流信息，没法帮您查。您可以在 App 的「我的订单」里点开这笔订单…'
tool_calls: []
```

**模型老实地遵守了 ch01 的约束，拒绝查物流。** 也就是说：**本章的验收标准 1 在不改 Prompt 的情况下 100% 会失败**——不是概率问题。

对照：同一个问题、**不带** System Prompt 时，模型正确调用了 `query_logistics`。诊断确定。

#### 改写方案

把"你没有权限"改为"你能通过工具查，但只能转述工具返回的内容"：

```
## 你的能力边界（最重要）
你可以通过工具查询真实数据：
- 问订单详情（买了什么、何时下单、什么状态）→ 调用 query_order
- 问物流进度、包裹到哪了 → 调用 query_logistics
- 问商品信息（价格、规格、库存）→ 调用 query_product
- 问平台政策（退货、换货、发票、发货时效、尺码等）→ 调用 query_faq
- 用户明确要求转人工，或问题确实无法自助解决 → 调用 create_ticket

**只能转述工具返回的内容。** 工具没给的细节一律不提——不补充、不推测、不编造。
尤其：金额、时间、城市、物流节点，工具没返回就不要说。
如果工具说没查到，就如实告诉用户查不到，并给出下一步（换个说法再问，或转人工）。
```

**保住了 ch01 那条约束的真正意图**：不许凭空断言订单事实。只是把"没有数据来源"换成了"有工具这个数据来源，且只能转述它"。

#### 实测效果（2026-09-20）

用上述 Prompt 测 8 个场景，工具选择 **7/8**：

| 场景 | 期望 | 实际 | 结果 |
|---|---|---|---|
| 订单 1001 的物流到哪了 | `query_logistics` | `query_logistics` | ✅ |
| 1001 这个订单买的什么 | `query_order` | `query_order` | ✅ |
| 退货政策是什么 | `query_faq` | `query_faq` | ✅ |
| 我要投诉，帮我转人工 | `create_ticket` | `create_ticket` | ✅ |
| 你好 / 谢谢 / 帮我写个 Python 脚本 | 不调工具 | 不调工具 | ✅ |
| **邮费是多少** | `query_faq` | **不调，反问了** | ❌ |

**唯一的不符是「邮费是多少」**——模型反问"您指的是哪方面的邮费呢？退货运费、换货运费，还是配送运费？"。这是合理行为（问题确实有歧义），但它意味着**验收标准 3 用这个措辞不会触发工具**。

改用问法即可：**「邮费政策是什么」实测会调 `query_faq`**，而 `faq` 表里确实没有「邮费」条目 → 构成要暴露的漏召回。

> 其余 5 条行为约束（不编造、不承诺赔付、不确定转人工、≤150 字、中文、AI 身份）**全部保留不动**。

## 7. 工具轮时序（`app/chat.py` 改造）

### 7.1 完整时序

```
① 从 MySQL 读回该会话的历史 messages
② 组装：system + 历史 + 本轮用户消息
③ 第一次请求模型（带五个工具），**边流边转发正文，同时累积整条响应**
④ 流结束后看累积结果的 tool_calls：
   ├─ 为空 → 不需要工具，正文已真流式转发完，结束      ← 零额外延迟
   └─ 非空 → 继续
              ↓
⑤ 执行工具（可能多个）→ 推 tool(start) 帧 → 执行 → 推 tool(end) 帧
⑥ 第二次请求模型（★ 这次不绑工具）
⑦ 流式转发最终回答
⑧ 落库：user、assistant(带 tool_calls)、N 条 tool 结果、最终 assistant
```

### 7.2 ★ 为什么是"先转发后决断"，而不是"看第一个增量就分叉"

**这一节是实测推翻原设计的记录，不要改回去。**

本设计初稿的写法是：看模型吐的**第一个非空增量**——是正文就判定"不需要工具"，是工具申请就进入工具轮。

**实测证伪（2026-09-20，无 System Prompt）：**

```
chunk1: content=''    tool_call_chunks=[]
chunk2: content='I'   tool_call_chunks=[]
chunk3: content="'ll" tool_call_chunks=[]
...
第一个非空增量 = TEXT
累积后 tool_calls = [{'name': 'query_logistics', ...}]   ← 模型明明要调工具
```

模型**先流式吐了正文，然后才发工具申请**。按"看第一个增量"的逻辑，这一轮会走"不需要工具"分支——**用户会看到机器人说"我去查一下"，然后工具不执行、没有下文**。

原设计把这种情况写在 §7.7 当"几乎不会发生的边界，不加逻辑（YAGNI）"。**它不是边界，它是这个模型的正常行为。**

**改成"先转发后决断"**：不管模型先说什么，正文一律实时转发，同时累积；流结束后再根据最终 `tool_calls` 是否为决定走向。好处：

- **不需要工具时仍是真流式**（零额外延迟）——ch01 的验收标准 1 保住了
- 需要工具时，用户会先看到模型说的半句（如"我帮您查一下"），再看到工具徽章，再看到最终回答——**这本来就是自然的对话节奏**
- 不再依赖"模型不会先说后调"这个未经证实的假设

**代价**：模型若有较长的"思考式前言"，用户要等它说完才看到工具徽章。实测带 System Prompt 时不会发生（见 §7.7）。

### 7.3 为什么不用非流式探测

另一种做法是第一次请求用非流式（`ainvoke`）拿完整响应再判断。**否掉的理由**：那会让"不需要工具"的每一轮都退化成**假流式**——先静默 1 秒多，然后文字一次性刷出来。ch01 的验收标准 1 就是"能看到流式输出"，这样做等于把 ch01 交付的东西弄丢。

### 7.4 单轮收敛的结构性保证

**第 ⑥ 步的第二次模型调用不绑工具。**

没有工具可用，模型就**不可能**再申请工具。于是"只做单轮"不是靠计数器拦出来的，而是**调用方式本身决定的**。这比 `max_iterations=1` 可靠——那条路漏判一次就变成 Agent Loop 了。

### 7.5 领域事件新增

沿用 ch01 的边界（`chat.py` 出领域事件，`api.py` 封 SSE 帧）：

```python
@dataclass(frozen=True)
class ToolCallEvent:      # 工具开始执行
    name: str
    arguments: dict
    call_id: str

@dataclass(frozen=True)
class ToolResultEvent:    # 工具执行结束
    call_id: str
    ok: bool
    summary: str
    elapsed_ms: int
```

### 7.6 并行工具调用

模型一轮里要几个工具就执行几个，结果全部回灌，然后收敛。仍然只回灌一次，不多轮循环——符合"单轮"的定义。

### 7.7 模型会不会"先正文后工具"——实测数据

**这个模型的正常行为是：有时会。** 与 System Prompt 强相关：

| 场景 | 第一个非空增量 | 是否调工具 |
|---|---|---|
| **无 System Prompt**，问「订单 1001 的物流到哪了」 | `TEXT`（先吐 "I'll look up the logistics information..."） | **是**（28 个 chunk，正文在前、工具申请在后） |
| 带**改写后**的 System Prompt，同一问题 | `TOOL` | 是 |
| 带改写后的 System Prompt，寒暄/道谢/越界 | `TEXT` | 否 |

**结论**：不能假设"先正文后工具"不会发生——它取决于 Prompt，而 Prompt 是会改的。所以 §7.1 的"先转发后决断"是**结构性的正确做法**，不是权宜之计。

**用户可见的后果**（当它发生时）：先看到模型说的半句（如"我帮您查一下"），再看到工具徽章，再看到最终回答。**这是可接受的对话节奏，不需要特殊处理。**

**落库注意**：当"先正文后工具"发生时，前言的正文与最终回答**分属两条 assistant 消息**（第一条带 `tool_calls`，第二条是最终回答）。`messages` 表按 `id` 顺序原样存两条即可，不做合并。

## 8. SSE 协议扩展与前端契约

### 8.1 新增事件

```
event: meta
data: {"session_id":"99b98538-..."}

event: tool                                     ← 新增：工具开始
data: {"phase":"start","call_id":"call_abc","name":"query_logistics",
       "arguments":{"order_id":"1001"}}

event: tool                                     ← 新增：工具结束
data: {"phase":"end","call_id":"call_abc","ok":true,
       "summary":"查询到 3 条物流轨迹，最新一条：已到达杭州中转站","elapsed_ms":1204}

event: delta
data: {"text":"您的包裹"}

event: done
data: {"finish_reason":"stop","usage":{...}}
```

**完整序列**：`meta? → tool(start)* → tool(end)* → delta* → done`

不需要工具的对话是 `meta? → delta* → done`——**与 ch01 完全一致，前端零改动即可兼容**。

### 8.2 为什么 start / end 分两帧

工具执行是唯一会让用户干等的地方（实测约 1 秒）。分两帧前端才能显示"**查询中…**"的真实等待状态，而不是等工具跑完才突然冒出一个徽章。

### 8.3 工具中文标签放在前端

徽章要显示「查物流」而不是 `query_logistics`。映射放**前端**一个小字典：

```js
const TOOL_LABELS = {
  query_order: "查订单", query_product: "查商品",
  query_logistics: "查物流", query_faq: "查常见问题",
  create_ticket: "建工单",
};
// 未知工具名 → 直接显示原始 name（后端加了新工具前端不会崩）
```

**理由**：标签是展示问题。让后端发 `label` 会把 UI 词汇渗进领域事件，而后端本该不知道"徽章"这种东西。代价是后端加工具时前端要补一行——但有兜底，不补也只是显示英文名。

### 8.4 前端形态

`web/index.html`，单文件、原生 JS、**零构建**。挂载：`app.mount("/chat", StaticFiles(directory="web", html=True))`，浏览器打开 `http://127.0.0.1:8000/chat`。

功能四件事：输入框、消息列表、SSE 解析、工具徽章。

**按用户要求走 Vibe Coding**：先做能跑的最小版本（输入框 + 气泡 + 徽章 + SSE 解析），用户看效果描述要改哪里再改。**不套 brainstorm / TDD / code review。**

## 9. 会话存储切换（对 ch01 的破坏性变更）

### 9.1 变更内容

`app/sessions.py` 新增 `MySQLSessionStore(SessionStore)`：

```python
class MySQLSessionStore(SessionStore):
    async def get(self, session_uuid: str) -> list[BaseMessage]: ...
    async def append(self, session_uuid: str, *messages: BaseMessage) -> None: ...
    async def exists(self, session_uuid: str) -> bool: ...
```

### 9.2 ★ `SessionStore` ABC 要从同步改成 async

ch01 的 ABC 是同步方法（`def get`），而 DB 访问要 `await`。**这是本章唯一会动到 ch01 已有接口的地方。**

受影响的文件清单（实施时必须逐个核对）：

| 文件 | 要改什么 |
|---|---|
| `app/sessions.py` | ABC 三个方法加 `async`；`InMemorySessionStore` 同步实现也要跟着改（或删除） |
| `app/api.py` | `store.get(...)` / `store.append(...)` 调用点加 `await` |
| `tests/test_api.py` | 桩与调用点跟着改 |
| `tests/test_sessions.py` | 全部测试改成 async |

### 9.3 消息的数据库往返

新增 `app/db/messages_io.py`——**唯一同时知道 LangChain 消息类型和表结构的地方**：

| LangChain | 表 |
|---|---|
| `HumanMessage` | `role='user'`, `content` |
| `AIMessage`（纯文本） | `role='assistant'`, `content` |
| `AIMessage`（带 tool_calls） | `role='assistant'`, `tool_calls` JSON |
| `ToolMessage` | `role='tool'`, `content`, `tool_call_id` |

> **注**：`messages.role` 的 ENUM 只有三个值，**不含 `system`**。System Prompt 不进表——它由 `app/prompts.py` 每次从模板构造（ch01 既有设计，不变）。

## 10. 配置

```ini
APP_DB_URL=mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service?charset=utf8mb4
APP_TOOL_TIMEOUT_S=3.0
APP_TOOL_MAX_RETRIES=2
```

沿用 ch01 的 `APP_` 前缀与 `pydantic-settings` 读取方式。

## 11. 错误处理

### 11.1 一条贯穿全节的原则

**"查不到"不是错误，"坏了"才是。**

```
query_order("9999")  → 正常返回："没有找到订单 9999"        ← 业务结果，模型照着回答
query_order 超时      → 异常 → 重试 → 仍失败 → 包装成工具结果  ← 故障，回灌让模型说人话
```

这条线划清楚，"错误处理"才不会变成把业务逻辑塞进 `try/except`。

### 11.2 工具失败：回灌而非中止

重试耗尽后，把错误当作一条 `ToolMessage` 回灌给模型（内容如"查询超时，暂时查不到订单信息"），让模型用自然语言告诉用户并给下一步。

**理由**：用户得到的是人话而不是 500；错误也落 `messages` 表，可追溯。

### 11.3 HTTP 错误边界不变

ch01 的规则不变：**流开始之前用 HTTP 状态码，流开始之后用 `event: error`**。工具失败走回灌，**不产生 error 帧**，所以这条边界一个字都不用改。

### 11.4 `create_ticket` 建单后更新会话状态

`create_ticket` 成功写入 `tickets` 后，**在同一事务内**把该会话的 `conversations.status` 改成 `'已转人工'`。

**为什么**：`status` 这个字段的含义就是"这通会话现在处于什么状态"。建了工单意味着转人工了，状态就该改——这不是权衡，是字段本身的语义。表数量不是职责边界的判据。

> `status` 的第三个值 `'已结束'` 在本章**没有触发它的动作**——本章没有"结束会话"这个操作。这不是"字段没人写"，是那一章还没做。

## 12. 测试与验证策略

按用户要求：可单测的代码走 TDD；纯 Prompt / 数据类任务用评估集替代 TDD。

### 12.1 TDD 覆盖

| 模块 | 测什么 |
|---|---|
| `repositories/` | 四个表的读写；`query_faq` 的 LIKE 命中与不命中 |
| `tools/registry.py` | 参数校验拒绝非法输入；超时触发重试；重试耗尽后包装成 `ok=False`；**参数错误不重试**；名字不存在时的行为 |
| `tools/catalog.py` | **确定性**：同 key 两次调用结果完全一致；不同 key 结果不同 |
| `db/messages_io.py` | 往返转换：含 tool_calls 的 AIMessage → 表 → 还原后相等 |
| `chat.py` | 打桩模型：不需要工具时直通流式；需要工具时事件序列为 `tool(start) → tool(end) → delta* → done`；**第二次调用不绑工具** |
| `sessions.py` | `MySQLSessionStore` 的 get/append/exists |

**测试数据库**：**不能用内存 SQLite**——DDL 用了 `ENUM`、`JSON`、`utf8mb4`，SQLite 不认。测试连同一个 MySQL 实例上的独立库 `customer_service_test`，每个用例在事务里跑完回滚。

### 12.2 评估集（替代 TDD 的质量验证）

工具选择的质量**不可单测**——它取决于模型，是纯 Prompt/数据类任务。新增 `evals/tool_cases.yaml`：

| 用例 | 期望 |
|---|---|
| 订单 1001 的物流到哪了 | 调 `query_logistics`（可能连带 `query_order`） |
| 退货政策是什么 | 调 `query_faq`，且参数含「退货」 |
| 邮费是多少 | 调 `query_faq`，**但查不到** → 验收 3 要记录的漏召回 |
| 你好 | **不调任何工具** |
| 我要投诉，帮我转人工 | 调 `create_ticket`，`ticket_type="投诉"` |
| 1001 这个订单买的什么 | 调 `query_order` |

**判据是"模型选了哪些工具 + 传给工具的参数"，不是最终回复的措辞**——后者由 ch01 的 `evals/cases.yaml` 管，两套不重叠。

## 13. 验收标准对应

| 验收 | 复现方式 |
|---|---|
| 1. 订单 1001 的物流到哪了 → 看到模型选中工具（气泡带徽章）并按返回结果作答 | 浏览器开 `/chat`；徽章显示「查物流」；回答内容与 `query_logistics("1001")` 的返回值一致（**确定性数据，可当场对账**） |
| 2. 退货政策是什么 → `query_faq` 查得到并作答 | 徽章显示「查常见问题」；回答引用 `faq` 表里的条目 |
| 3. **邮费政策是什么** → **查不到（预期结果）** | 徽章仍出现（工具被调了），但工具返回"没有找到相关问答"；**把模型如何应对这个空结果记进 `dev-notes/ch02.md`，留给下一步升级** |

> **措辞按实测调整过**：原始措辞「邮费是多少」实测**不会触发工具**（模型会反问澄清，见 §6.4）。「邮费政策是什么」实测会调 `query_faq`，且 `faq` 表无「邮费」条目 → 才能构成要暴露的漏召回。**这个改动是实测驱动的，不是把用例改到能过。**

## 14. 已知限制与后续升级路径

### 14.1 刷新页面丢失当前对话显示

**现象**：前端只把 `session_id` 存在 JS 变量里，刷新即丢；且本章没做"读历史"的接口。

**性质**：这是"缺一个读接口"，**不是缺数据**——`messages` 表存的是完整流水（含 `tool_calls` 和 tool 结果），所以刷新后连**工具徽章都能精确还原**。

**升级路径**（留给后续章节）：

```
后端：GET /api/conversations/{session_id}/messages     约 30 行
前端：页面加载时调一次，渲染历史 + 还原徽章            约 40 行
```

**但有一个真正需要先想清楚的产品问题**：前端得先记住 `session_id` 才能查历史，而存储方式对应三种产品形态：

| 方案 | 特点 |
|---|---|
| `localStorage` | 关掉浏览器再打开还在；同一浏览器只能记住一个会话 |
| URL 带 `?s=xxx` | 可分享链接、能开多个会话；地址栏难看 |
| 两者都要 | "会话列表 + 当前选中"，是完整产品形态 |

**这是产品选择而非技术选择，所以单独留一章来想，不在 ch02 顺手做掉。**

### 14.2 其他限制

- `query_faq` 用 SQL `LIKE`，**没有语义泛化能力**——这正是验收 3 要暴露的问题，ch03 走向量检索
- `query_order` / `query_product` / `query_logistics` 返回假数据，不接真实系统
- 单 worker 运行；多 worker 下 MySQL 会话存储虽然共享，但 ch01 既有的其它进程内状态未验证
- **所有会话的 `user_id` 都是 `'anonymous'`**（§5.4）——本章无鉴权，系统无法得知"用户是谁"。因此 `conversations.idx_user_id` 这个索引在本章没有任何区分度，它保留是为有了真实用户之后准备的。接登录时只需改 `APP_DEFAULT_USER_ID` 与一处赋值，不动表结构。

## 15. 未决项与风险

| 项 | 性质 | 处置 |
|---|---|---|
| `SessionStore` ABC 改 async 会波及 ch01 的 `api.py` 与两个测试文件 | **已知破坏性变更** | §9.2 已列受影响文件清单，实施时逐个核对 |
| ~~模型可能"先吐正文再申请工具"~~ | ~~已知边界~~ | ✅ **已解决**：实测确认是正常行为（§7.7），设计改为"先转发后决断"（§7.2），结构上不再依赖这个假设 |
| **ch01 的 System Prompt 与工具能力冲突** | **阻塞验收 1** | ✅ **已解决**：§6.4 给出改写方案并实测 7/8。**这是原设计完全漏掉的一块交付物**——设计阶段只盯着工具和数据库，没检查既有 Prompt 是否会与新能力打架 |
| 验收 3 的原始措辞不触发工具 | 中 | ✅ **已解决**：实测「邮费是多少」模型会反问，「邮费政策是什么」才触发（§6.4）。验收措辞已按实测调整 |
| `ticket_no` 并发撞号 | 低概率 | 主键约束兜底，撞了就重试一次；不引入分布式 ID 生成器 |
| `query_faq` 的关键词抽取完全依赖模型 | 中 | 由 §12.2 评估集测量；漏召回是预期要暴露的结果 |
| Docker 依赖用户手动开启 WSL 集成 | 前置条件 | 已确认可用（§3） |
