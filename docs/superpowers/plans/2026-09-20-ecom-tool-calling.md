# ch02 Function Calling 工具链 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch01 的客服后端装上 Function Calling 工具链：模型自主决定调哪个工具，工具结果回灌后仍逐 token 流式作答，聊天页用徽章显示工具轨迹。

**Architecture:** 四层切分 —— `db/`（表与连接）→ `repositories/`（读写）→ `tools/`（业务能力）→ `chat.py`（编排模型与工具）。工具轮用**流式探测**：判模型吐的第一个增量是正文还是工具申请；**第二次模型调用不绑工具**，从调用方式上结构性保证单轮收敛。

**Tech Stack:** Python 3.12.14 / uv / FastAPI 0.141.1 / SQLAlchemy 2.0.54 (async) / aiomysql 0.3.2 / MySQL 8.4 (Docker) / LangChain 1.4.1 / langchain-core 1.6.3 / langchain-openai 1.6.2

**Spec:** `docs/superpowers/specs/2026-09-20-ecom-tool-calling-design.md`

## Global Constraints

以下约束适用于**每一个**任务，不再逐条重复：

- **Python 3.12.14**，依赖管理用 `uv`（本机无 `pip3`）。所有命令在仓库根 `/home/zzx/custermerHelp` 执行。
- **Docker 必须可用**（MySQL 8.4 容器）。`docker compose up -d` 起库。
- **数据库 URL**：`mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service?charset=utf8mb4`
- **全程 async**：接口、工具、仓储、会话存储都是 `async def`；DB 访问一律 `await`。**绝不能**在 async 代码里调用同步 SQLAlchemy。
- **`Chunk + Chunk` 可累加**：LangChain 1.x 的 `AIMessageChunk` 支持 `+`，用于累积流式分片。官方文档确认的写法是 `gathered = chunk if gathered is None else gathered + chunk`。
- **ch01 的三条上游约束不得回退**（任何"顺手简化"都会破坏）：
  - `thinking` 必须关闭，走 `extra_body`
  - 输出上限必须走 `extra_body`，**不能**走 `ChatOpenAI(max_tokens=...)`
  - `with_structured_output` 必须显式 `method="function_calling"`
- **脚本一律用 `uv run python -m 包.模块`**，不要用 `uv run python 路径/文件.py`（后者 `sys.path[0]` 变成脚本目录，`import app` 失败）。
- **中文 token 计数用 `app/context.py` 的 `_count_tokens`**，不用 `count_tokens_approximately`（对中文低估 58%）。
- **`.env` 绝不进 git。**
- **提交信息**末尾必须带：`Co-Authored-By: Claude Code <noreply@anthropic.com>`
- **每完成一个任务**，向 `dev-notes/ch02.md` 追记一段（关键原话 / 关键产出 / 被纠偏 / 翻车返工）。

---

### Task 1: Docker + MySQL + 建表 + 种子数据

**Files:**
- Create: `docker-compose.yml`, `scripts/schema.sql`, `scripts/seed.py`
- Modify: `.env.example`, `.gitignore`（加 `mysql_data` 无需，volume 由 docker 管）

**Interfaces:**
- Consumes: 无
- Produces: 一个跑着的 MySQL 8.4（库 `customer_service`，另有测试库 `customer_service_test`），四张表已建，`faq` 表有种子数据

- [ ] **Step 1: 写 `scripts/schema.sql`**

**这是用户提供的 DDL，除 `session_uuid` 一列外逐字采用。不要"顺手优化"表结构。**

```sql
-- ch02 · Function Calling 工具链 · 建表 DDL
-- 商品、订单、物流走工具内 mock，不建表
-- 建表顺序：先 conversations，再依赖它的 messages / tickets

-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE IF NOT EXISTS conversations (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  session_uuid VARCHAR(36)     NOT NULL                COMMENT '对外会话标识(ch02 新增,对应 API 的 session_id)',
  user_id      VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status       ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_session_uuid (session_uuid),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE IF NOT EXISTS messages (
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

-- FAQ 问答对:query_faq 的数据源
CREATE TABLE IF NOT EXISTS faq (
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
CREATE TABLE IF NOT EXISTS tickets (
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

- [ ] **Step 2: 写 `docker-compose.yml`**

```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: custermerhelp-mysql
    environment:
      MYSQL_ROOT_PASSWORD: rootpw
      MYSQL_DATABASE: customer_service
      MYSQL_USER: app
      MYSQL_PASSWORD: apppw
      # ★ 不是装饰：镜像默认 POSIX locale 时，initdb 会把 DDL 里的中文按
      # latin1 双重编码落库（enum('进行中') 存成乱码字节），且中文条件查询
      # 恒返回 0——不报错，只是静默失效。Task 1 实测踩到并修复。
      LANG: C.UTF-8
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
    ports:
      # ★ 3307 而非 3306：宿主 3306 被 Windows 侧 MySQL80 服务 + VMware NAT
      # 转发规则占用（用户确认）。实测绑 0.0.0.0:3306 与 127.0.0.1:3306 都失败。
      # 所有连接串必须用 3307。
      - "3307:3306"
    volumes:
      - ./scripts/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
      # 官方 entrypoint 只给 app 授权 MYSQL_DATABASE；测试库需显式建库+授权，
      # 否则 seed 报 1044，后续测试也连不上。
      - ./scripts/init-test-db.sql:/docker-entrypoint-initdb.d/02-init-test-db.sql:ro
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "127.0.0.1", "-uroot", "-prootpw"]
      interval: 5s
      timeout: 5s
      retries: 20

volumes:
  mysql_data:
```

> **改了 `docker-compose.yml` 之后必须 `docker compose down -v` 再 `up -d`。** `LANG` 只在**首次 initdb** 时起作用；不删数据卷重建的话，库里已有的乱码字节不会被修正。验证方式是对着真实数据看十六进制：
>
> ```bash
> docker compose exec -T mysql mysql -uapp -papppw customer_service --default-character-set=utf8mb4 \
>   -N -e "SELECT HEX('进行中');"     # 必须是 E8BF9BE8A18C，不是 C3A8C2BF...
> docker compose exec -T mysql mysql -uapp -papppw customer_service --default-character-set=utf8mb4 \
>   -N -e "SELECT COUNT(*) FROM faq WHERE category='售后';"   # 必须 > 0
> ```

- [ ] **Step 3: 起库并验证四张表**

```bash
docker compose up -d
# 等到 healthy（首次拉镜像可能要几分钟）
until [ "$(docker inspect -f '{{.State.Health.Status}}' custermerhelp-mysql 2>/dev/null)" = "healthy" ]; do sleep 2; done
echo "mysql healthy"

docker compose exec -T mysql mysql -uapp -papppw customer_service -e "SHOW TABLES;"
```
Expected: 四行 —— `conversations` / `faq` / `messages` / `tickets`

```bash
docker compose exec -T mysql mysql -uapp -papppw customer_service -e "DESC conversations;"
```
Expected: 含 `session_uuid` 列，且 `uk_session_uuid` 唯一键存在

- [ ] **Step 4: 写 `scripts/seed.py`**

```python
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
    await test_engine.dispose()
    print("customer_service_test 已建表")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 加配置项并跑 seeding**

`.env.example` 追加：

```ini
APP_DB_URL=mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service?charset=utf8mb4
APP_DEFAULT_USER_ID=anonymous
APP_TOOL_TIMEOUT_S=3.0
APP_TOOL_MAX_RETRIES=2
```

`app/config.py` 的 `Settings` 追加字段（保持既有字段不动）：

```python
    db_url: str = "mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service?charset=utf8mb4"
    default_user_id: str = "anonymous"
    tool_timeout_s: float = 3.0
    tool_max_retries: int = 2
```

同步写入真实 `.env`（`APP_DB_URL` 等四项）。

```bash
uv run python -m scripts.seed
```
Expected: `customer_service.faq 已灌入 12 条` + `customer_service_test 已建表`

- [ ] **Step 6: 验证 `faq` 的命中与不命中（这是验收 2/3 的数据前提）**

```bash
docker compose exec -T mysql mysql -uapp -papppw customer_service \
  -e "SELECT COUNT(*) FROM faq WHERE question LIKE '%退货%';
      SELECT COUNT(*) FROM faq WHERE question LIKE '%邮费%';"
```
Expected: 第一条 ≥ 3，**第二条必须是 0**

> ⚠️ **第二条为 0 是设计要求，不是遗漏。** 验收标准 3 要求「邮费是多少」查不到，把这个漏召回记录下来作为下一步升级的输入。**不要为了让验收 3 通过而补数据。**

- [ ] **Step 7: 提交**

```bash
git add docker-compose.yml scripts/schema.sql scripts/seed.py .env.example app/config.py
git commit -m "feat: MySQL 容器、四张表 DDL 与种子数据

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: `db/` 数据层

**Files:**
- Create: `app/db/__init__.py`, `app/db/base.py`, `app/db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Consumes: `app.config.get_settings()`（Task 1 加了 `db_url`）
- Produces:
  - `app.db.base.Base`（`DeclarativeBase` 子类）
  - `app.db.base.engine`（`AsyncEngine`）、`app.db.base.AsyncSessionLocal`（`async_sessionmaker[AsyncSession]`）
  - `app.db.base.get_db() -> AsyncIterator[AsyncSession]`（FastAPI 依赖）
  - `app.db.models`：`Conversation`、`Message`、`Faq`、`Ticket` 四个 ORM 类

- [ ] **Step 1: 写失败的测试 `tests/test_db_models.py`**

```python
from app.db.base import Base
from app.db import models  # noqa: F401  确保模型被注册进 metadata


def test_four_tables_registered():
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversations_has_session_uuid_unique():
    t = Base.metadata.tables["conversations"]
    assert "session_uuid" in t.columns
    assert t.columns["session_uuid"].unique is True


def test_messages_columns():
    t = Base.metadata.tables["messages"]
    # 注意是 t.columns.keys()：SQLAlchemy 2.0 里 set(t.columns) 迭代出的是
    # Column 对象而不是列名，直接和字符串集合比较永远为假。Task 2 实测踩到。
    assert {"conversation_id", "role", "content", "tool_calls", "tool_call_id"} <= set(t.columns.keys())


def test_tickets_primary_key_is_ticket_no():
    t = Base.metadata.tables["tickets"]
    assert [c.name for c in t.primary_key.columns] == ["ticket_no"]


def test_messages_has_fk_to_conversations():
    t = Base.metadata.tables["messages"]
    fks = {fk.target_fullname for fk in t.foreign_keys}
    assert "conversations.id" in fks
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_db_models.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: 依赖加进 `pyproject.toml`**

```toml
dependencies = [
    # ... 既有依赖不动 ...
    "sqlalchemy[asyncio]>=2.0.54",
    "aiomysql>=0.3.2",
]
```

```bash
uv sync
```

- [ ] **Step 4: 写 `app/db/__init__.py`（空文件）、`app/db/base.py`**

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(get_settings().db_url, echo=False, pool_pre_ping=True)

# expire_on_commit=False：提交后仍可读对象属性，避免 async 下的隐式懒加载报错
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每个请求一个 session。

    测试通过 app.dependency_overrides[get_db] 替换成测试 session。
    """
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] **Step 5: 写 `app/db/models.py`**

```python
import datetime

from sqlalchemy import BigInteger, Enum, ForeignKey, Index, String, Text, func
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
```

> **注意**：`messages` 与 `tickets` 各有名为 `idx_conversation_id` 的索引，但分属不同表，**MySQL 允许重名**。这里给 tickets 的索引换名只是为了在 SQLAlchemy 的 metadata 里更好区分，不改 DDL 语义。

- [ ] **Step 6: 跑测试确认通过**

```bash
uv run pytest tests/test_db_models.py -v
```
Expected: 5 passed

- [ ] **Step 7: 验证 ORM 与真实表结构一致**

```bash
uv run python -c "
import asyncio
from sqlalchemy import text
from app.db.base import engine

async def main():
    async with engine.begin() as conn:
        r = await conn.execute(text('DESC conversations'))
        for row in r:
            print(row[0], row[1])
    await engine.dispose()

asyncio.run(main())
"
```
Expected: 打印出 `id / session_uuid / user_id / status / created_at / updated_at` 六列（证明 ORM 映射的表在库里真实存在）

- [ ] **Step 8: 提交**

```bash
git add app/db/ tests/test_db_models.py pyproject.toml uv.lock
git commit -m "feat: SQLAlchemy async 数据层与四张表 ORM 模型

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: `repositories/` 数据访问层

**Files:**
- Create: `app/repositories/__init__.py`, `app/repositories/faq.py`, `app/repositories/conversation.py`, `app/repositories/message.py`, `app/repositories/ticket.py`
- Create: `tests/conftest.py`（改：加测试 DB fixture）
- Test: `tests/test_repositories.py`

**Interfaces:**
- Consumes: `app.db.base.AsyncSessionLocal`、`app.db.models.*`
- Produces:
  - `faq.search_by_keyword(session, keyword, limit=3) -> list[Faq]`
  - `conversation.get_or_create(session, session_uuid, user_id) -> Conversation`
  - `conversation.set_status(session, conversation_id, status) -> None`
  - `message.append(session, conversation_id, role, content, tool_calls=None, tool_call_id=None) -> None`
  - `message.list_by_conversation(session, conversation_id) -> list[Message]`
  - `ticket.create(session, ticket_no, conversation_id, description, ticket_type) -> Ticket`
  - `ticket.count_on_date(session, date) -> int`

- [ ] **Step 1: 给 `tests/conftest.py` 加测试 DB fixture**

**整份替换** `tests/conftest.py` 为下面的内容。文件里原有的 `APP_LLM_API_KEY` 设置与 `anyio_backend` fixture **必须保留**，所以下面这份代码块把它们一并写全了——不要在这份之外再"追加"旧内容，那样会产生重复定义。

```python
import os

import pytest

# 测试期间不读真实 .env，避免测试依赖本机密钥
os.environ.setdefault("APP_LLM_API_KEY", "test-key")
# ★ 必须在任何 app.db.base 被 import 之前设置：base.py 在 import 时就用
# get_settings().db_url 建 engine，晚了就会指向生产库。
os.environ.setdefault(
    "APP_DB_URL",
    "mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service_test?charset=utf8mb4",
)

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_session():
    """每个用例一个 session，跑完回滚。

    外层的 engine 每次测试重建，避免事件循环切换导致的连接池串用。
    """
    from app.config import get_settings

    get_settings.cache_clear()
    engine = create_async_engine(get_settings().db_url, echo=False)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as session:
        trans = await session.begin()
        try:
            yield session
        finally:
            await trans.rollback()
    await engine.dispose()
```

> **注意**：`APP_DB_URL` 必须在**任何 `app.db.base` 被 import 之前**设置好，否则 `create_async_engine` 会在 import 时拿到生产库 URL。上面的 `os.environ.setdefault` 在文件顶部，符合这个顺序。

- [ ] **Step 2: 写失败的测试 `tests/test_repositories.py`**

```python
import datetime

import pytest
from sqlalchemy import text

from app.repositories import conversation, faq, message, ticket


async def test_faq_hit(db_session):
    rows = await faq.search_by_keyword(db_session, "退货")
    assert len(rows) >= 3
    assert all("退货" in r.question for r in rows)


async def test_faq_miss(db_session):
    """★ 验收标准 3 的数据前提：『邮费』在 faq 表里查不到。

    种子数据故意不含邮费条目。这条测试锁住这个事实——
    如果哪天有人为了"让验收 3 通过"补了数据，这条测试会红。
    """
    rows = await faq.search_by_keyword(db_session, "邮费")
    assert rows == []


async def test_faq_limit(db_session):
    rows = await faq.search_by_keyword(db_session, "退", limit=2)
    assert len(rows) <= 2


async def test_conversation_get_or_create_is_idempotent(db_session):
    a = await conversation.get_or_create(db_session, "uuid-1", "anonymous")
    b = await conversation.get_or_create(db_session, "uuid-1", "anonymous")
    assert a.id == b.id


async def test_conversation_set_status(db_session):
    c = await conversation.get_or_create(db_session, "uuid-2", "anonymous")
    assert c.status == "进行中"
    await conversation.set_status(db_session, c.id, "已转人工")
    await db_session.refresh(c)
    assert c.status == "已转人工"


async def test_message_append_and_list_in_order(db_session):
    c = await conversation.get_or_create(db_session, "uuid-3", "anonymous")
    await message.append(db_session, c.id, "user", "第一条")
    await message.append(db_session, c.id, "assistant", "第二条")
    rows = await message.list_by_conversation(db_session, c.id)
    assert [r.content for r in rows] == ["第一条", "第二条"]


async def test_message_stores_tool_calls_json(db_session):
    c = await conversation.get_or_create(db_session, "uuid-4", "anonymous")
    calls = [{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}]
    await message.append(db_session, c.id, "assistant", None, tool_calls=calls)
    rows = await message.list_by_conversation(db_session, c.id)
    assert rows[0].tool_calls == calls


async def test_message_stores_tool_call_id(db_session):
    c = await conversation.get_or_create(db_session, "uuid-5", "anonymous")
    await message.append(db_session, c.id, "tool", "查询结果", tool_call_id="call_1")
    rows = await message.list_by_conversation(db_session, c.id)
    assert rows[0].tool_call_id == "call_1"


async def test_ticket_create(db_session):
    c = await conversation.get_or_create(db_session, "uuid-6", "anonymous")
    t = await ticket.create(db_session, "T20260920001", c.id, "我要投诉", "投诉")
    assert t.ticket_no == "T20260920001"
    assert t.status == "待处理"


async def test_ticket_count_on_date(db_session):
    c = await conversation.get_or_create(db_session, "uuid-7", "anonymous")
    today = datetime.date.today()
    before = await ticket.count_on_date(db_session, today)
    await ticket.create(db_session, "T20260920999", c.id, "咨询", "咨询")
    after = await ticket.count_on_date(db_session, today)
    assert after == before + 1
```

- [ ] **Step 3: 跑测试确认失败**

```bash
uv run pytest tests/test_repositories.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.repositories'`

- [ ] **Step 4: 写四个 repository**

`app/repositories/__init__.py`（空文件）

`app/repositories/faq.py`：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Faq


async def search_by_keyword(session: AsyncSession, keyword: str, limit: int = 3) -> list[Faq]:
    """按关键词检索 faq。

    ★ 只搜 question 列，不搜 answer。若连 answer 一起搜，某条答案里顺带
    提到"邮费自理"就会命中，会让验收标准 3 的结论依赖种子数据的措辞巧合，
    测的就不是检索能力而是数据运气（spec §5.5）。
    """
    stmt = (
        select(Faq)
        .where(Faq.question.like(f"%{keyword}%"))
        .order_by(Faq.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())
```

`app/repositories/conversation.py`：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation


async def get_or_create(session: AsyncSession, session_uuid: str, user_id: str) -> Conversation:
    stmt = select(Conversation).where(Conversation.session_uuid == session_uuid)
    found = (await session.execute(stmt)).scalar_one_or_none()
    if found is not None:
        return found

    conv = Conversation(session_uuid=session_uuid, user_id=user_id, status="进行中")
    session.add(conv)
    await session.flush()          # 拿到自增 id，但不提交（事务由调用方管）
    return conv


async def set_status(session: AsyncSession, conversation_id: int, status: str) -> None:
    conv = await session.get(Conversation, conversation_id)
    if conv is not None:
        conv.status = status
        await session.flush()
```

`app/repositories/message.py`：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message


async def append(
    session: AsyncSession,
    conversation_id: int,
    role: str,
    content: str | None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> None:
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
    )
    await session.flush()


async def list_by_conversation(session: AsyncSession, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars())
```

`app/repositories/ticket.py`：

```python
import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Ticket


async def create(
    session: AsyncSession,
    ticket_no: str,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Ticket:
    t = Ticket(
        ticket_no=ticket_no,
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
        status="待处理",
    )
    session.add(t)
    await session.flush()
    return t


async def count_on_date(session: AsyncSession, day: datetime.date) -> int:
    stmt = select(func.count()).select_from(Ticket).where(
        func.date(Ticket.created_at) == day
    )
    return (await session.execute(stmt)).scalar_one()


async def get_by_no(session: AsyncSession, ticket_no: str) -> Ticket | None:
    return await session.get(Ticket, ticket_no)
```

> **`flush()` 而不是 `commit()`**：事务边界由调用方（`chat.py` 的整轮写入）控制，repository 只负责把语句发出去并拿到自增 id。这样"一轮对话的所有写入要么全成要么全不成"才是可能的。

- [ ] **Step 5: 跑测试确认通过**

```bash
uv run pytest tests/test_repositories.py -v
```
Expected: 10 passed

- [ ] **Step 6: 跑全量测试确认没破坏 ch01**

```bash
uv run pytest -q
```
Expected: 74 + 15（新）= 89 passed

- [ ] **Step 7: 提交**

```bash
git add app/repositories/ tests/conftest.py tests/test_repositories.py
git commit -m "feat: 四个仓储模块（faq/conversation/message/ticket）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: `db/messages_io.py` 消息往返转换

**Files:**
- Create: `app/db/messages_io.py`
- Test: `tests/test_messages_io.py`

**Interfaces:**
- Consumes: `app.db.models.Message`、`langchain_core.messages.*`
- Produces:
  - `to_rows(messages: list[BaseMessage]) -> list[dict]`（每项是可直接喂给 `message.append` 的 kwargs）
  - `from_rows(rows: list[Message]) -> list[BaseMessage]`

**这是唯一同时知道 LangChain 消息类型和表结构的地方。** 其他模块都不该做这个转换。

- [ ] **Step 1: 写失败的测试 `tests/test_messages_io.py`**

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.db.messages_io import from_rows, to_rows


class FakeRow:
    """模拟 app.db.models.Message 的行对象。"""

    def __init__(self, role, content, tool_calls=None, tool_call_id=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


def test_human_message_roundtrip():
    rows = to_rows([HumanMessage("你好")])
    assert rows == [{"role": "user", "content": "你好", "tool_calls": None, "tool_call_id": None}]
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], HumanMessage)
    assert back[0].content == "你好"


def test_plain_ai_message_roundtrip():
    rows = to_rows([AIMessage("您好")])
    assert rows[0]["role"] == "assistant"
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], AIMessage)
    assert back[0].content == "您好"
    assert back[0].tool_calls == []


def test_ai_message_with_tool_calls_roundtrip():
    calls = [{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}]
    msg = AIMessage(content="", tool_calls=calls)
    rows = to_rows([msg])
    assert rows[0]["role"] == "assistant"
    assert rows[0]["tool_calls"] == calls

    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], AIMessage)
    assert back[0].tool_calls == calls


def test_tool_message_roundtrip():
    msg = ToolMessage(content="查到 3 条轨迹", tool_call_id="call_1")
    rows = to_rows([msg])
    assert rows[0] == {
        "role": "tool", "content": "查到 3 条轨迹",
        "tool_calls": None, "tool_call_id": "call_1",
    }
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], ToolMessage)
    assert back[0].content == "查到 3 条轨迹"
    assert back[0].tool_call_id == "call_1"


def test_full_conversation_order_preserved():
    msgs = [
        HumanMessage("订单 1001 的物流到哪了"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]),
        ToolMessage(content="已到达杭州中转站", tool_call_id="c1"),
        AIMessage("您的包裹已到达杭州中转站。"),
    ]
    rows = to_rows(msgs)
    back = from_rows([FakeRow(**r) for r in rows])
    assert [type(m).__name__ for m in back] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert back[2].tool_call_id == "c1"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_messages_io.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db.messages_io'`

- [ ] **Step 3: 写 `app/db/messages_io.py`**

```python
"""LangChain 消息 ↔ messages 表行 的双向转换。

★ 本模块是全项目唯一同时知道 LangChain 消息类型和表结构的地方。
其他模块都不该做这个转换（spec §9.3）。
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.db.models import Message

_ROLE_TO_CLASS = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


def to_rows(messages: list[BaseMessage]) -> list[dict]:
    """转成可直接喂给 repositories.message.append 的 kwargs 列表。"""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content,
                        "tool_calls": None, "tool_call_id": None})
        elif isinstance(m, AIMessage):
            calls = m.tool_calls or None
            out.append({"role": "assistant", "content": m.content or None,
                        "tool_calls": calls, "tool_call_id": None})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content,
                        "tool_calls": None, "tool_call_id": m.tool_call_id})
        else:
            raise TypeError(f"无法落库的消息类型: {type(m).__name__}")
    return out


def from_rows(rows: list[Message]) -> list[BaseMessage]:
    """表行还原成 LangChain 消息。"""
    out: list[BaseMessage] = []
    for r in rows:
        if r.role == "user":
            out.append(HumanMessage(r.content or ""))
        elif r.role == "assistant":
            out.append(AIMessage(content=r.content or "", tool_calls=r.tool_calls or []))
        elif r.role == "tool":
            out.append(ToolMessage(content=r.content or "", tool_call_id=r.tool_call_id or ""))
        else:
            raise ValueError(f"未知的 role: {r.role}")
    return out
```

> **`messages.role` 的 ENUM 只有三个值，不含 `system`** —— System Prompt 不进表，它由 `app/prompts.py` 每次从模板构造（ch01 既有设计）。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_messages_io.py -v
```
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add app/db/messages_io.py tests/test_messages_io.py
git commit -m "feat: LangChain 消息与 messages 表的双向转换

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: `SessionStore` 异步化 + `MySQLSessionStore`

> **⚠️ 这是本章唯一的破坏性变更**，会动到 ch01 的既有接口。受影响文件：`app/sessions.py`、`app/api.py`、`tests/test_sessions.py`、`tests/test_api.py`。

**Files:**
- Modify: `app/sessions.py`（ABC 加 `async`；`InMemorySessionStore` 删掉或改 async）
- Modify: `app/api.py`（`store.get` / `store.append` 调用点加 `await`）
- Modify: `tests/test_sessions.py`（全部改 async）
- Modify: `tests/test_api.py`（桩跟着改）
- Test: `tests/test_sessions.py`

**Interfaces:**
- Consumes: `app.db.base.AsyncSessionLocal`、`app.repositories.conversation/message`、`app.db.messages_io`
- Produces:
  - `SessionStore` 的三个方法全部改成 `async def`
  - `MySQLSessionStore(store_session_factory=None)`，实现 `get/append/exists`

- [ ] **Step 1: 写失败的测试（改写 `tests/test_sessions.py`）**

**整份替换**为：

```python
import pytest

from app.sessions import MySQLSessionStore


async def test_get_unknown_session_returns_empty(db_session):
    store = MySQLSessionStore(session=db_session)
    assert await store.get("从没见过的-uuid") == []
    assert await store.exists("从没见过的-uuid") is False


async def test_append_then_get_roundtrip(db_session):
    from langchain_core.messages import AIMessage, HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("你好"), AIMessage("您好"))
    msgs = await store.get("u1")
    assert [m.content for m in msgs] == ["你好", "您好"]
    assert await store.exists("u1") is True


async def test_sessions_are_isolated(db_session):
    from langchain_core.messages import HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("A"))
    await store.append("u2", HumanMessage("B"))
    assert [m.content for m in await store.get("u1")] == ["A"]
    assert [m.content for m in await store.get("u2")] == ["B"]


async def test_get_returns_new_list_not_shared_state(db_session):
    """返回的列表被调用方改动，不应影响后续读取。"""
    from langchain_core.messages import HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("A"))
    got = await store.get("u1")
    got.append(HumanMessage("偷偷加的"))
    assert len(await store.get("u1")) == 1


async def test_tool_messages_survive_roundtrip(db_session):
    """带 tool_calls 的 assistant 与 tool 结果都要能还原——工具徽章依赖这个。"""
    from langchain_core.messages import AIMessage, ToolMessage

    store = MySQLSessionStore(session=db_session)
    calls = [{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}]
    await store.append(
        "u1",
        AIMessage(content="", tool_calls=calls),
        ToolMessage(content="查到订单", tool_call_id="c1"),
    )
    msgs = await store.get("u1")
    assert msgs[0].tool_calls == calls
    assert msgs[1].tool_call_id == "c1"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_sessions.py -v
```
Expected: FAIL — `ImportError: cannot import name 'MySQLSessionStore'`

- [ ] **Step 3: 改写 `app/sessions.py`**

**整份替换**为：

```python
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.messages_io import from_rows, to_rows
from app.db.models import Conversation
from app.repositories import conversation as conversation_repo
from app.repositories import message as message_repo


class SessionStore(ABC):
    """会话历史存储抽象。

    ★ ch02 起三个方法都是 async —— 因为 DB 访问要 await。
    这相对 ch01 是破坏性变更，调用方（api.py）必须跟着加 await。
    """

    @abstractmethod
    async def get(self, session_id: str) -> list[BaseMessage]: ...

    @abstractmethod
    async def append(self, session_id: str, *messages: BaseMessage) -> None: ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool: ...


class MySQLSessionStore(SessionStore):
    """基于 MySQL 的实现：上下文从 messages 表读回，重启不丢。

    每通会话在 conversations 表里有一行，messages 通过 conversation_id 挂上去。
    对外仍用 UUID 形式的 session_id（= conversations.session_uuid）。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _conversation_id(self, session_id: str) -> int | None:
        conv = await self._find(session_id)
        return None if conv is None else conv.id

    async def _find(self, session_id: str) -> Conversation | None:
        from sqlalchemy import select

        stmt = select(Conversation).where(Conversation.session_uuid == session_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get(self, session_id: str) -> list[BaseMessage]:
        cid = await self._conversation_id(session_id)
        if cid is None:
            return []
        rows = await message_repo.list_by_conversation(self._session, cid)
        return from_rows(rows)

    async def append(self, session_id: str, *messages: BaseMessage) -> None:
        conv = await conversation_repo.get_or_create(
            self._session, session_id, get_settings().default_user_id
        )
        for row in to_rows(list(messages)):
            await message_repo.append(self._session, conv.id, **row)

    async def exists(self, session_id: str) -> bool:
        return await self._find(session_id) is not None
```

> **`InMemorySessionStore` 被删除**。ch02 起上下文以 MySQL 为准，保留一个不再有人用的内存实现只会造成"两个真相"。

- [ ] **Step 4: 改 `app/api.py` 的调用点**

找到 `store.get(session_id)` 与 `store.append(...)` 两处，各加 `await`：

```python
    history = await store.get(session_id)
```

```python
            if completed:
                await store.append(
                    session_id,
                    HumanMessage(req.message),
                    AIMessage("".join(collected)),
                )
```

同时把 `get_store()` 依赖改成**按请求从 `get_db` 拿 session**：

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.sessions import MySQLSessionStore, SessionStore


def get_store(db: AsyncSession = Depends(get_db)) -> SessionStore:
    """依赖注入点：测试通过 app.dependency_overrides 替换它。"""
    return MySQLSessionStore(session=db)
```

- [ ] **Step 5: 改 `tests/test_api.py` 的桩**

把 `InMemorySessionStore(...)` 换成 fake：

```python
class FakeStore:
    """测试用的内存桩，接口与 SessionStore 一致但全是 async。"""

    def __init__(self):
        self._data: dict[str, list] = {}

    async def get(self, session_id: str):
        return list(self._data.get(session_id, []))

    async def append(self, session_id: str, *messages):
        self._data.setdefault(session_id, []).extend(messages)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._data
```

fixture 里 `app.dependency_overrides[get_store] = lambda: store`，并把 `store.get(sid)` 断言改成 `await store.get(sid)`（测试函数改成 `async def`）。

**同时**：因为路由现在依赖 `get_db`，测试必须覆盖它，否则会真连数据库。fixture 里加：

```python
    app.dependency_overrides[get_db] = lambda: None   # get_store 已被覆盖，get_db 不会被真正使用
```

- [ ] **Step 6: 跑测试确认通过**

```bash
uv run pytest tests/test_sessions.py -v
uv run pytest tests/test_api.py -v
uv run pytest -q
```
Expected: `test_sessions.py` 5 passed；`test_api.py` 全部通过；全量 **94 passed** 左右

- [ ] **Step 7: 提交**

```bash
git add app/sessions.py app/api.py tests/test_sessions.py tests/test_api.py
git commit -m "refactor!: SessionStore 异步化并新增 MySQL 实现

BREAKING: SessionStore 三个方法改 async，api.py 调用点与两个测试文件
跟着改。InMemorySessionStore 删除——ch02 起上下文以 MySQL 为准。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: `tools/context.py` + `tools/registry.py`（工具基础设施）

**Files:**
- Create: `app/tools/__init__.py`, `app/tools/context.py`, `app/tools/registry.py`
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `app.config.get_settings()`
- Produces:
  - `app.tools.context.ToolContext`（frozen dataclass，字段 `conversation_id: int`）
  - `app.tools.context.tool_context(conversation_id)` 上下文管理器
  - `app.tools.context.current_context() -> ToolContext`
  - `app.tools.registry.ToolSpec`（`tool: BaseTool`、`timeout_s: float`、`max_retries: int`）
  - `app.tools.registry.ToolOutcome`（`call_id: str`、`ok: bool`、`content: str`、`summary: str`、`elapsed_ms: int`）
  - `app.tools.registry.ToolRegistry`：`as_langchain_tools()`、`get(name)`、`async execute(name, raw_args, call_id) -> ToolOutcome`

- [ ] **Step 1: 写失败的测试 `tests/test_tool_registry.py`**

```python
import asyncio

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools.context import ToolContext, current_context, tool_context
from app.tools.registry import ToolOutcome, ToolRegistry, ToolSpec


class EchoArgs(BaseModel):
    text: str = Field(description="要回显的文本")


@tool(args_schema=EchoArgs)
async def echo(text: str) -> str:
    """回显输入。"""
    return f"echo:{text}"


@tool(args_schema=EchoArgs)
async def slow(text: str) -> str:
    """故意超时。"""
    await asyncio.sleep(10)
    return "never"


_calls = {"n": 0}


@tool(args_schema=EchoArgs)
async def flaky(text: str) -> str:
    """前两次超时，第三次成功——用来验证重试确实发生了。"""
    _calls["n"] += 1
    if _calls["n"] < 3:
        await asyncio.sleep(10)
    return f"ok after {_calls['n']}"


def make_registry(*tools, timeout_s=0.1, max_retries=2):
    return ToolRegistry([ToolSpec(tool=t, timeout_s=timeout_s, max_retries=max_retries)
                         for t in tools])


async def test_registry_lists_tools_for_binding():
    reg = make_registry(echo)
    names = [t.name for t in reg.as_langchain_tools()]
    assert names == ["echo"]


async def test_execute_success():
    reg = make_registry(echo)
    out = await reg.execute("echo", {"text": "hi"}, "call_1")
    assert isinstance(out, ToolOutcome)
    assert out.ok is True
    assert "echo:hi" in out.content
    assert out.call_id == "call_1"
    assert out.elapsed_ms >= 0


async def test_execute_unknown_tool():
    reg = make_registry(echo)
    out = await reg.execute("不存在的工具", {}, "call_2")
    assert out.ok is False
    assert "没有" in out.content or "不存在" in out.content


async def test_invalid_args_rejected_without_retry():
    """参数不合法不重试——重试一次得到的还是同样的非法参数。"""
    calls_before = _calls["n"]
    reg = make_registry(flaky, timeout_s=5.0)
    out = await reg.execute("flaky", {"完全没有的参数": 1}, "call_3")
    assert out.ok is False
    assert _calls["n"] == calls_before, "参数校验失败时不该调用工具函数"


async def test_timeout_is_retried_then_reported():
    reg = make_registry(slow, timeout_s=0.05, max_retries=2)
    out = await reg.execute("slow", {"text": "x"}, "call_4")
    assert out.ok is False
    assert "超时" in out.content
    assert out.elapsed_ms >= 50


async def test_retry_can_succeed():
    _calls["n"] = 0
    reg = make_registry(flaky, timeout_s=0.05, max_retries=2)
    out = await reg.execute("flaky", {"text": "x"}, "call_5")
    assert out.ok is True
    assert _calls["n"] == 3, "应该试满 3 次才成功"


async def test_zero_retries_means_single_attempt():
    _calls["n"] = 0
    reg = make_registry(flaky, timeout_s=0.05, max_retries=0)
    out = await reg.execute("flaky", {"text": "x"}, "call_6")
    assert out.ok is False
    assert _calls["n"] == 1


async def test_context_is_visible_inside_tool():
    seen = {}

    @tool(args_schema=EchoArgs)
    async def spy(text: str) -> str:
        """窥探上下文。"""
        seen["cid"] = current_context().conversation_id
        return "ok"

    reg = make_registry(spy)
    with tool_context(42):
        await reg.execute("spy", {"text": "x"}, "call_7")
    assert seen["cid"] == 42


async def test_context_raises_outside_tool_context():
    with pytest.raises(LookupError):
        current_context()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_tool_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.tools'`

- [ ] **Step 3: 写 `app/tools/__init__.py`、`app/tools/context.py`**

`__init__.py` 先留空（Task 7 再补 `build_registry()`）。

`context.py`：

```python
"""工具上下文：把模型看不到的信息（如当前会话 id）传给工具。

为什么需要它：create_ticket 的签名里没有会话 id —— 模型不该知道、
也不该能指定往哪个会话建单（spec §6.3）。会话 id 由 registry 在
执行前注入，工具用 current_context() 取。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContext:
    conversation_id: int


_ctx: ContextVar[ToolContext] = ContextVar("tool_context")


@contextmanager
def tool_context(conversation_id: int) -> Iterator[None]:
    token = _ctx.set(ToolContext(conversation_id=conversation_id))
    try:
        yield
    finally:
        _ctx.reset(token)


def current_context() -> ToolContext:
    """取当前工具上下文。不在工具执行期间调用会抛 LookupError。"""
    try:
        return _ctx.get()
    except LookupError as exc:
        raise LookupError(
            "当前不在工具执行上下文里；工具只能由 ToolRegistry.execute() 调用"
        ) from exc
```

- [ ] **Step 4: 写 `app/tools/registry.py`**

```python
"""工具基础设施：注册管理、参数 Schema 校验、执行错误处理、超时重试。

设计原则（spec §11.1）：「查不到」不是错误，「坏了」才是。
业务空结果由工具自己正常返回；异常只留给真故障（超时、连接断开、代码 bug）。
"""

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.tools.context import tool_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    timeout_s: float = 3.0
    max_retries: int = 2


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    ok: bool
    content: str      # 回灌给模型的文本
    summary: str      # 给前端徽章 tooltip 的一句话
    elapsed_ms: int


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {s.tool.name: s for s in specs}

    def as_langchain_tools(self) -> list[BaseTool]:
        """给 model.bind_tools() 用。"""
        return [s.tool for s in self._specs.values()]

    def get(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            known = "、".join(sorted(self._specs))
            raise KeyError(f"没有名为 {name} 的工具。可用工具：{known}")
        return spec

    async def execute(
        self, name: str, raw_args: dict, call_id: str, *, conversation_id: int | None = None
    ) -> ToolOutcome:
        started = time.monotonic()

        def elapsed() -> int:
            return int((time.monotonic() - started) * 1000)

        # ① 查注册表
        try:
            spec = self.get(name)
        except KeyError as exc:
            return ToolOutcome(call_id, False, str(exc), "工具不存在", elapsed())

        # ② Schema 校验 —— 失败不重试（重试一次得到的还是同样的非法参数）
        try:
            args = spec.tool.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            msg = f"参数不合法：{exc.errors()}"
            logger.warning("工具 %s 参数校验失败 call_id=%s: %s", name, call_id, exc.errors())
            return ToolOutcome(call_id, False, msg, "参数不合法", elapsed())

        # ③④ 执行 + 超时重试（只重试超时/连接类故障）
        attempts = spec.max_retries + 1
        last_error: Exception | None = None

        with tool_context(conversation_id if conversation_id is not None else 0):
            for attempt in range(1, attempts + 1):
                try:
                    result = await asyncio.wait_for(
                        spec.tool.ainvoke(args.model_dump()), timeout=spec.timeout_s
                    )
                    text = result if isinstance(result, str) else str(result)
                    return ToolOutcome(call_id, True, text, self._summarize(text), elapsed())
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    logger.warning(
                        "工具 %s 第 %d/%d 次超时（%.1fs）call_id=%s",
                        name, attempt, attempts, spec.timeout_s, call_id,
                    )
                except (ConnectionError, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "工具 %s 第 %d/%d 次连接故障 call_id=%s: %s",
                        name, attempt, attempts, call_id, exc,
                    )
                except Exception as exc:  # noqa: BLE001 — 代码 bug 也归为工具失败，回灌而非中断对话
                    last_error = exc
                    logger.exception("工具 %s 执行异常 call_id=%s", name, call_id)
                    break  # 非瞬态故障不重试

        kind = "超时" if isinstance(last_error, (TimeoutError, asyncio.TimeoutError)) else "执行失败"
        content = f"工具 {name} {kind}（已尝试 {attempts} 次）：{type(last_error).__name__}。暂时查不到相关信息。"
        return ToolOutcome(call_id, False, content, f"{kind}", elapsed())

    @staticmethod
    def _summarize(text: str, limit: int = 60) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"
```

> **`timeout_s` 与 `max_retries` 的默认值**来自 `ToolSpec`，但生产构造时应从 `get_settings()` 读 `tool_timeout_s` / `tool_max_retries`（Task 7 的 `build_registry()` 负责）。

- [ ] **Step 5: 跑测试确认通过**

```bash
uv run pytest tests/test_tool_registry.py -v
```
Expected: 9 passed

- [ ] **Step 6: 提交**

```bash
git add app/tools/ tests/test_tool_registry.py
git commit -m "feat: 工具基础设施（注册、Schema 校验、超时重试、错误包装）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 五个业务工具 + `build_registry()`

**Files:**
- Create: `app/tools/catalog.py`, `app/tools/knowledge.py`, `app/tools/ticketing.py`
- Modify: `app/tools/__init__.py`（加 `build_registry()`）
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `app.tools.registry`、`app.tools.context.current_context`、`app.repositories.*`、`app.config.get_settings()`
- Produces:
  - `catalog.query_order` / `catalog.query_product` / `catalog.query_logistics`（`BaseTool`）
  - `knowledge.query_faq`
  - `ticketing.create_ticket`
  - `app.tools.build_registry() -> ToolRegistry`

- [ ] **Step 1: 写失败的测试 `tests/test_tools.py`**

```python
import pytest

from app.tools import build_registry
from app.tools.catalog import query_logistics, query_order, query_product


async def test_registry_has_five_tools():
    reg = build_registry()
    names = sorted(t.name for t in reg.as_langchain_tools())
    assert names == [
        "create_ticket", "query_faq", "query_logistics", "query_order", "query_product",
    ]


@pytest.mark.parametrize("tool", [query_order, query_product, query_logistics])
async def test_fake_data_is_deterministic(tool):
    """★ 同一个 key 两次调用必须返回完全相同的数据。

    这是验收标准 1 能做对账的前提——模型说"已到杭州中转站"，
    你去工具结果里能查到同一句。纯随机会让验收退化成"看着像对"。
    """
    arg = {"order_id": "1001"} if tool is not query_product else {"product_id": "P1001"}
    a = await tool.ainvoke(arg)
    b = await tool.ainvoke(arg)
    assert a == b


@pytest.mark.parametrize("tool", [query_order, query_product, query_logistics])
async def test_different_keys_give_different_data(tool):
    if tool is query_product:
        a, b = await tool.ainvoke({"product_id": "P1"}), await tool.ainvoke({"product_id": "P2"})
    else:
        a, b = await tool.ainvoke({"order_id": "1"}), await tool.ainvoke({"order_id": "2"})
    assert a != b


async def test_query_faq_hit_via_registry(db_session):
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "退货"}, "c1", conversation_id=1)
    assert out.ok is True
    assert "退货" in out.content


async def test_query_faq_miss_via_registry(db_session):
    """★ 验收标准 3：『邮费』查不到。工具正常返回"没找到"，不是异常。"""
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c2", conversation_id=1)
    assert out.ok is True, "查不到是业务结果，不是工具故障"
    assert "没有找到" in out.content or "未找到" in out.content


async def test_create_ticket_writes_row_and_updates_conversation(db_session):
    from app.repositories import conversation as conv_repo

    conv = await conv_repo.get_or_create(db_session, "ticket-test-uuid", "anonymous")
    assert conv.status == "进行中"

    reg = build_registry(session=db_session)
    out = await reg.execute(
        "create_ticket",
        {"description": "我要投诉物流太慢", "ticket_type": "投诉"},
        "c3",
        conversation_id=conv.id,
    )
    assert out.ok is True

    await db_session.refresh(conv)
    assert conv.status == "已转人工", "建单即转人工，状态要跟着改"

    from app.db.models import Ticket
    from sqlalchemy import select

    rows = list((await db_session.execute(
        select(Ticket).where(Ticket.conversation_id == conv.id)
    )).scalars())
    assert len(rows) == 1
    assert rows[0].ticket_type == "投诉"
    assert rows[0].ticket_no.startswith("T")


async def test_create_ticket_rejects_bad_type(db_session):
    reg = build_registry(session=db_session)
    out = await reg.execute(
        "create_ticket", {"description": "x", "ticket_type": "退款"}, "c4", conversation_id=1
    )
    assert out.ok is False
    assert "参数不合法" in out.content
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_tools.py -v
```
Expected: FAIL — `ImportError: cannot import name 'build_registry'`

- [ ] **Step 3: 写 `app/tools/catalog.py`**

```python
"""三个返回确定性假数据的工具。

ch02 不接真实的电商/物流系统，但数据必须**可重复**：
同一个订单号两次调用返回完全相同的内容，这样验收时能对着工具结果
逐字核对模型的回答（spec §6.1）。
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
    """由 key 派生的确定性随机源。同一个 key 永远得到同一串数据。"""
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
    lines = [f"- 2026-09-{r.randint(10, 19):02d} {r.randint(8, 22):02d}:{r.randint(0, 59):02d} "
             f"{c}转运中心：{r.choice(['已揽收', '已发出', '已到达', '运输中'])}" for c in cities]
    latest = f"- 2026-09-20 {r.randint(8, 22):02d}:{r.randint(0, 59):02d} " \
             f"{r.choice(_CITIES)}：{r.choice(['派送中', '已签收', '待取件'])}"
    return (
        f"订单 {order_id} 的物流信息：\n"
        f"- 承运商：{carrier}\n"
        f"- 运单号：{waybill}\n"
        f"- 轨迹：\n" + "\n".join(lines) + "\n" + latest
    )
```

- [ ] **Step 4: 写 `app/tools/knowledge.py`**

```python
"""FAQ 检索工具。本章用 SQL LIKE，没有语义泛化能力——这正是验收标准 3 要暴露的问题。"""

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import faq as faq_repo

_session: AsyncSession | None = None


def bind_session(session: AsyncSession | None) -> None:
    """由 build_registry() 调用，把当前请求的 DB session 交给工具。

    用模块级变量而非函数参数，因为工具签名要暴露给模型——
    模型不该看到、也不该传一个 Session 对象。
    """
    global _session
    _session = session


@tool
async def query_faq(keyword: str) -> str:
    """按关键词检索常见问答库（退货、换货、发票、发货时效、尺码等）。

    keyword 请传「核心业务词」——例如用户问「退货政策是什么」就传「退货」，
    问「怎么开发票」就传「发票」。不要传整句话。
    """
    if _session is None:
        return "FAQ 查询暂时不可用（数据库未绑定）。"

    rows = await faq_repo.search_by_keyword(_session, keyword)
    if not rows:
        return f"没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"

    return "\n\n".join(f"问：{r.question}\n答：{r.answer}" for r in rows)
```

- [ ] **Step 5: 写 `app/tools/ticketing.py`**

```python
"""人工工单工具。成功建单后把会话状态改成「已转人工」——建单即转人工（spec §11.4）。"""

import datetime
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import conversation as conv_repo
from app.repositories import ticket as ticket_repo
from app.tools.context import current_context

_session: AsyncSession | None = None


def bind_session(session: AsyncSession | None) -> None:
    global _session
    _session = session


async def _next_ticket_no(session: AsyncSession) -> str:
    today = datetime.date.today()
    n = await ticket_repo.count_on_date(session, today) + 1
    return f"T{today.strftime('%Y%m%d')}{n:03d}"


@tool
async def create_ticket(
    description: str, ticket_type: Literal["售后", "投诉", "咨询"]
) -> str:
    """创建人工工单并转接人工客服。仅当用户明确要求转人工、或问题无法自助解决时调用。"""
    if _session is None:
        return "工单系统暂时不可用（数据库未绑定）。"

    # 会话 id 从上下文取，不在函数签名里——模型不该能指定往哪个会话建单
    conversation_id = current_context().conversation_id

    ticket_no = await _next_ticket_no(_session)
    await ticket_repo.create(
        _session, ticket_no, conversation_id, description, ticket_type
    )
    await conv_repo.set_status(_session, conversation_id, "已转人工")

    return f"已为您创建工单，工单号 {ticket_no}（类型：{ticket_type}）。人工客服会尽快跟进。"
```

- [ ] **Step 6: 写 `app/tools/__init__.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.tools import knowledge, ticketing
from app.tools.catalog import query_logistics, query_order, query_product
from app.tools.knowledge import query_faq
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.ticketing import create_ticket

ALL_TOOLS = [query_order, query_product, query_logistics, query_faq, create_ticket]


def build_registry(session: AsyncSession | None = None) -> ToolRegistry:
    """构造工具注册表，并把当前请求的 DB session 绑给需要它的工具。

    超时与重试次数从配置读，不写死在 ToolSpec 的默认值里。
    """
    s = get_settings()
    knowledge.bind_session(session)
    ticketing.bind_session(session)
    return ToolRegistry(
        [ToolSpec(tool=t, timeout_s=s.tool_timeout_s, max_retries=s.tool_max_retries)
         for t in ALL_TOOLS]
    )
```

- [ ] **Step 7: 跑测试确认通过**

```bash
uv run pytest tests/test_tools.py -v
uv run pytest -q
```
Expected: `test_tools.py` 全部通过；全量约 **104 passed**

- [ ] **Step 8: 提交**

```bash
git add app/tools/ tests/test_tools.py
git commit -m "feat: 五个业务工具与 build_registry()

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 改写 System Prompt + 工具选择评估集

> **本任务在计划执行中途新增。** 原因是实测发现 ch01 的 System Prompt 与本章功能**直接冲突**：它写着"你**没有**订单系统的访问权限"，模型老实遵守，问「订单 1001 的物流到哪了」时**拒绝查物流**——验收标准 1 在不改 Prompt 的情况下 100% 会失败。详见 spec §6.4。

**Files:**
- Modify: `app/prompts.py`（只改 `SYSTEM_PROMPT` 的能力边界段）
- Create: `evals/tool_cases.yaml`, `evals/run_tool_evals.py`
- Modify: `README.md`（加运行命令）

**Interfaces:**
- Consumes: `app.tools.build_registry`、`app.providers.get_chat_model`、`app.prompts.build_chat_prompt`
- Produces: 改写后的 `SYSTEM_PROMPT`；`evals/run_tool_evals.py` 可执行

- [ ] **Step 1: 改写 `app/prompts.py` 的能力边界段**

**只替换 `SYSTEM_PROMPT` 里"## 你的能力边界（最重要）"那一整段。** 后面「行为约束」六条与「回复风格」**一字不动**。

原来那段是：

```
## 你的能力边界（最重要）
你**没有**订单系统的访问权限。你看不到任何订单状态、物流信息、金额、收货地址。
因此你**绝对不能**说"我帮您查到了""我看到您的订单显示…"这类话。
需要订单信息时，引导用户自己在 App 的「我的订单」页查看，或转人工客服。
```

替换为：

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

**为什么这样改而不是删掉约束**：ch01 那条约束的**真正意图**是"不许凭空断言订单事实"。改后变成了"有工具这个数据来源，且只能转述它"——意图保住了，与新能力的冲突解除了。

- [ ] **Step 2: 写 `evals/tool_cases.yaml`**

```yaml
# ch02 工具选择评估集
# 判据是「模型选了哪些工具 + 传给工具的参数」，不是最终回复的措辞——
# 后者由 evals/cases.yaml（ch01）管，两套不重叠。
- id: logistics-1
  input: 订单 1001 的物流到哪了
  expect_tools: [query_logistics]
  expect_args_contain: {query_logistics: "1001"}
- id: order-1
  input: 1001 这个订单买的什么
  expect_tools: [query_order]
  expect_args_contain: {query_order: "1001"}
- id: faq-hit-1
  input: 退货政策是什么
  expect_tools: [query_faq]
  expect_args_contain: {query_faq: "退货"}
- id: faq-hit-2
  input: 怎么开发票
  expect_tools: [query_faq]
  expect_args_contain: {query_faq: "发票"}
- id: faq-miss-1
  input: 邮费政策是什么
  expect_tools: [query_faq]
  expect_args_contain: {query_faq: "邮费"}
- id: chitchat-1
  input: 你好
  expect_tools: []
- id: chitchat-2
  input: 谢谢
  expect_tools: []
- id: chitchat-3
  input: 帮我写个 Python 脚本
  expect_tools: []
- id: ticket-1
  input: 我要投诉，帮我转人工
  expect_tools: [create_ticket]
- id: ticket-2
  input: 这个问题解决不了，转人工客服
  expect_tools: [create_ticket]
```

> **`faq-miss-1` 用「邮费政策是什么」而不是「邮费是多少」**——实测前者会调 `query_faq`、后者模型会反问澄清（spec §6.4）。这不是把用例改到能过，是按实测选定**能触发工具**的措辞，才能暴露要记录的漏召回。

- [ ] **Step 3: 写 `evals/run_tool_evals.py`**

```python
"""ch02 工具选择评估集。

用法：uv run python -m evals.run_tool_evals
判据是「模型选了哪些工具 + 传给工具的参数」，不是最终回复的措辞。
"""

import asyncio
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from app.prompts import build_chat_prompt
from app.providers import get_chat_model
from app.tools import build_registry

ROOT = Path(__file__).parent


async def main() -> int:
    cases = yaml.safe_load((ROOT / "tool_cases.yaml").read_text(encoding="utf-8"))
    model = get_chat_model(temperature=0)
    # session=None：本评估只测「选不选工具、传什么参数」，不测工具的执行结果
    bound = model.bind_tools(build_registry(session=None).as_langchain_tools())

    results = []
    for c in cases:
        rendered = build_chat_prompt().format_messages(
            history=[HumanMessage(c["input"])]
        )
        reply = await bound.ainvoke(rendered)
        called = [tc["name"] for tc in (reply.tool_calls or [])]

        problems = []
        if sorted(called) != sorted(c["expect_tools"]):
            problems.append(f"工具不符：期望 {c['expect_tools']}，实际 {called}")

        for tool_name, must_contain in (c.get("expect_args_contain") or {}).items():
            call = next((tc for tc in (reply.tool_calls or []) if tc["name"] == tool_name), None)
            if call is None:
                problems.append(f"没调 {tool_name}，无法检查参数")
            elif must_contain not in str(call["args"]):
                problems.append(f"{tool_name} 参数 {call['args']} 不含「{must_contain}」")

        results.append((c["id"], not problems, problems, called))
        mark = "PASS" if not problems else "FAIL"
        detail = "" if not problems else "  " + "；".join(problems)
        print(f"  [{mark}] {c['id']:<14} 调用 {called}{detail}")
        if problems and reply.content:
            print(f"           模型改为回复：{reply.content[:60]}")

    ok = sum(1 for _, p, _, _ in results if p)
    print(f"\n工具选择通过率：{ok}/{len(results)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: 跑评估集**

```bash
uv run python -m evals.run_tool_evals
```
Expected: 一张通过率表。

**预期起点约 8~10/10**。如果 `logistics-1` / `order-1` 失败（模型仍拒绝查），说明 Step 1 的 Prompt 改动没生效——回去核对 `app/prompts.py` 是否真的替换了能力边界段。

- [ ] **Step 5: 若失败，调优 Prompt 而不是调用例**

改 `SYSTEM_PROMPT` 的措辞（**不是**改 `tool_cases.yaml` 的期望值），重跑。

**特别关注 `chitchat-3`（帮我写个 Python 脚本）**——模型最容易对越界请求也去调工具。如果它调了，说明 `create_ticket` 的描述诱导性太强（"问题无法自助解决时调用"这句太宽），收紧它。

**每次改动都要在 `dev-notes/ch02.md` 里记"改了什么 → 哪条从 FAIL 变 PASS"。** 不要凭单次跑下结论——上游 `temperature=0` 仍有抖动（ch01 实测：两次相同代码 15 条回复有 14 条不同），关键归因要做重复验证。

- [ ] **Step 6: `README.md` 加命令**

```markdown
uv run python -m evals.run_tool_evals    # ch02 工具选择评估集
```

- [ ] **Step 7: 提交**

```bash
git add app/prompts.py evals/tool_cases.yaml evals/run_tool_evals.py README.md
git commit -m "feat: 改写 System Prompt 以启用工具，并加工具选择评估集

ch01 的 SYSTEM_PROMPT 写着「你没有订单系统的访问权限」，与本章的
查订单/查物流能力直接冲突——实测模型会拒绝查物流，验收 1 必然失败。
改为「可以查，但只能转述工具返回的内容」，保住原约束的真正意图。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: `chat.py` 工具轮（本章核心）

**Files:**
- Modify: `app/chat.py`
- Test: `tests/test_chat_tools.py`（新建，不动 ch01 的 `tests/test_chat.py`）

**Interfaces:**
- Consumes: `app.tools.registry.ToolRegistry`、`app.tools.registry.ToolOutcome`、`app.prompts.build_chat_prompt`、`app.context.prepare_messages`
- Produces:
  - 新事件 `ToolCallEvent(name, arguments, call_id)`、`ToolResultEvent(call_id, ok, summary, elapsed_ms)`
  - `DomainEvent` 联合扩展为五个类型
  - `stream_reply_with_tools(message, history, *, model, registry, token_budget, conversation_id) -> AsyncIterator[DomainEvent]`
  - **`stream_reply` 保持原签名与行为不变**（ch01 的 `tests/test_chat.py` 必须继续通过）

- [ ] **Step 1: 写失败的测试 `tests/test_chat_tools.py`**

```python
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.chat import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    stream_reply_with_tools,
)
from app.tools.registry import ToolOutcome, ToolSpec
from app.tools.registry import ToolRegistry


class ScriptedTool:
    """一个假的 LangChain 工具，只为让 registry 能注册它。"""

    def __init__(self, name):
        self.name = name
        from pydantic import BaseModel

        class Args(BaseModel):
            order_id: str = "1001"

        self.args_schema = Args

    async def ainvoke(self, args):
        return f"{self.name} 的结果：已到达杭州中转站"


def make_registry(names=("query_logistics",)):
    return ToolRegistry([ToolSpec(tool=ScriptedTool(n), timeout_s=1.0, max_retries=0)
                         for n in names])


class NoToolModel:
    """第一个增量就是正文 —— 模型不需要工具。"""

    def __init__(self):
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def astream(self, messages):
        self.calls += 1
        for t in ["您", "好"]:
            yield AIMessageChunk(content=t)


class ToolThenAnswerModel:
    """第一次吐工具申请，第二次吐正文。记录每次调用绑了哪些工具。"""

    def __init__(self):
        self.bound_history: list[list] = []
        self.seen_messages: list[list] = []
        self._bound: list = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        self.bound_history.append(list(self._bound))
        self.seen_messages.append(list(messages))
        if len(self.bound_history) == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_logistics", "args": '{"order_id": "1001"}',
                     "id": "call_1", "index": 0, "type": "tool_call_chunk"}
                ],
            )
        else:
            for t in ["已", "到", "杭", "州"]:
                yield AIMessageChunk(content=t)


async def collect(gen):
    return [e async for e in gen]


async def test_no_tool_path_streams_directly():
    model = NoToolModel()
    events = await collect(stream_reply_with_tools(
        "你好", [], model=model, registry=make_registry(), token_budget=4096, conversation_id=1
    ))
    assert [type(e).__name__ for e in events] == ["DeltaEvent", "DeltaEvent", "DoneEvent"]
    assert model.calls == 1, "不需要工具时只调一次模型"


async def test_tool_path_emits_tool_events_then_deltas():
    model = ToolThenAnswerModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    names = [type(e).__name__ for e in events]
    assert names == [
        "ToolCallEvent", "ToolResultEvent", "DeltaEvent", "DeltaEvent", "DeltaEvent",
        "DeltaEvent", "DoneEvent",
    ]
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    assert call.name == "query_logistics"
    assert call.arguments == {"order_id": "1001"}
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is True
    assert "杭州" in result.summary


async def test_second_call_has_no_tools_bound():
    """★ 单轮收敛的结构性保证：第二次调用不绑工具，模型不可能再申请工具。"""
    model = ToolThenAnswerModel()
    await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    assert len(model.bound_history) == 2
    assert len(model.bound_history[0]) == 1, "第一次要绑工具"
    assert model.bound_history[1] == [], "第二次绝不能绑工具"


async def test_second_call_receives_tool_result_message():
    from langchain_core.messages import ToolMessage

    model = ToolThenAnswerModel()
    await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    second = model.seen_messages[1]
    assert isinstance(second[-1], ToolMessage)
    assert second[-1].tool_call_id == "call_1"
    assert "杭州" in second[-1].content


async def test_failing_tool_is_fed_back_not_raised():
    """工具失败回灌成 tool 结果，让模型说人话——不产生 ErrorEvent。"""

    class Boom:
        name = "query_logistics"

        def __init__(self):
            from pydantic import BaseModel

            class Args(BaseModel):
                order_id: str = "1001"

            self.args_schema = Args

        async def ainvoke(self, args):
            raise RuntimeError("上游炸了")

    reg = ToolRegistry([ToolSpec(tool=Boom(), timeout_s=1.0, max_retries=0)])
    model = ToolThenAnswerModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=reg,
        token_budget=4096, conversation_id=1,
    ))
    assert not any(isinstance(e, ErrorEvent) for e in events)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is False
    assert any(isinstance(e, DeltaEvent) for e in events), "回灌后模型仍要作答"


class PreambleThenToolModel:
    """★ 实测出来的真实行为：模型先吐一段正文，然后才发工具申请。

    第一版设计按「第一个增量是正文 → 判定不需要工具」来分叉，遇到这个模型
    会走错分支——工具根本不执行，用户只看到"我帮您查一下"然后没有下文。
    这条测试就是防那个回归的。
    """

    def __init__(self):
        self.bound_history: list[list] = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        self.bound_history.append(list(self._bound))
        if len(self.bound_history) == 1:
            yield AIMessageChunk(content="好的，我帮您查一下")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_logistics", "args": '{"order_id": "1001"}',
                    "id": "call_p", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            yield AIMessageChunk(content="您的包裹已到达杭州中转站。")


async def test_preamble_then_tool_still_executes_tool():
    """★ 回归测试：先说后调，工具也必须执行。

    这是本章设计改动的原因（spec §7.2/§7.7）。如果哪天有人把实现改回
    「看第一个增量就分叉」，这条测试会红。
    """
    model = PreambleThenToolModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    kinds = [type(e).__name__ for e in events]

    # 前言要转发出去（用户看得到），工具要执行，最终回答要有
    assert kinds[0] == "DeltaEvent", "前言应该被实时转发"
    assert "ToolCallEvent" in kinds, "先说后调时工具仍必须执行"
    assert "ToolResultEvent" in kinds
    assert kinds[-1] == "DoneEvent"

    # 工具在第二次请求之前执行完
    assert kinds.index("ToolResultEvent") < kinds.index("DeltaEvent", kinds.index("ToolResultEvent"))

    # 第二次仍然不绑工具
    assert model.bound_history[1] == []


async def test_no_tool_round_means_no_tool_events():
    """不需要工具时不该出现任何 tool 事件。"""
    model = NoToolModel()
    events = await collect(stream_reply_with_tools(
        "你好", [], model=model, registry=make_registry(), token_budget=4096, conversation_id=1
    ))
    kinds = [type(e).__name__ for e in events]
    assert "ToolCallEvent" not in kinds
    assert "ToolResultEvent" not in kinds


async def test_current_message_is_last_in_rendered_prompt():
    """当前用户消息必须排在渲染结果最后，历史在它前面。"""

    class RecordingModel:
        def __init__(self):
            self.seen = None

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            self.seen = list(messages)
            yield AIMessageChunk(content="好")

    model = RecordingModel()
    await collect(stream_reply_with_tools(
        "最新提问", [HumanMessage("之前的话")], model=model,
        registry=make_registry(), token_budget=4096, conversation_id=1,
    ))
    contents = [m.content for m in model.seen]
    assert contents[0].startswith("你是「小助手」"), "第一条应是 System Prompt"
    assert contents[1:] == ["之前的话", "最新提问"]
```

> **`test_preamble_then_tool_still_executes_tool` 是本章最重要的回归测试**——它锁住的是"设计被实测推翻后改对"这件事。删掉它，实现就可能在下次重构时退回错误分支，而症状（用户看到机器人说"我去查一下"然后没下文）不会让任何其他测试变红。

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_chat_tools.py -v
```
Expected: FAIL — `ImportError: cannot import name 'ToolCallEvent'`

- [ ] **Step 3: 写 `app/chat.py`（追加，不改动 `stream_reply`）**

在 ch01 的 `app/chat.py` 基础上**追加**以下内容（`DeltaEvent` / `DoneEvent` / `ErrorEvent` / `classify_exception` / `stream_reply` **保持不动**）：

```python
@dataclass(frozen=True)
class ToolCallEvent:
    name: str
    arguments: dict
    call_id: str


@dataclass(frozen=True)
class ToolResultEvent:
    call_id: str
    ok: bool
    summary: str
    content: str          # 回灌给模型的完整文本；落库时用它构造 ToolMessage
    elapsed_ms: int


DomainEventWithTools = DeltaEvent | DoneEvent | ErrorEvent | ToolCallEvent | ToolResultEvent


async def stream_reply_with_tools(
    message: str,
    history: list[BaseMessage],
    *,
    model=None,
    registry,
    token_budget: int,
    conversation_id: int,
) -> AsyncIterator[DomainEventWithTools]:
    """带工具轮的一轮对话。

    时序（spec §7.1）：
      ① 组装消息（system + 裁剪后历史 + 当前消息）
      ② 第一次请求模型，带工具 —— 正文实时转发，同时累积整条响应
      ③ 流结束后看累积结果的 tool_calls：
         为空 → 不需要工具，正文已转发完，结束（零额外延迟）
         非空 → 执行工具，再发第二次请求
      ④ 推 ToolCallEvent / ToolResultEvent
      ⑤ 第二次请求模型 —— ★ 不绑工具，结构性保证单轮收敛
      ⑥ 流式转发最终回答

    ★ 为什么不是「看第一个增量就分叉」（spec §7.2）：实测这个模型会
    **先吐正文再申请工具**。按第一个增量分叉会走错分支——用户看到机器人
    说"我去查一下"然后工具不执行、没有下文。改为流结束后看最终结果。
    """
    from langchain_core.messages import ToolMessage

    model = model or get_chat_model()
    messages = prepare_messages(history, HumanMessage(message), token_budget=token_budget)
    rendered = build_chat_prompt().format_messages(history=messages)

    # ---- ② 第一次请求：带工具。正文实时转发，同时累积整条响应 ----
    bound = model.bind_tools(registry.as_langchain_tools())
    gathered = None
    try:
        async for chunk in bound.astream(rendered):
            gathered = chunk if gathered is None else gathered + chunk
            if chunk.text:
                yield DeltaEvent(text=chunk.text)
    except Exception as exc:  # noqa: BLE001
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    tool_calls = list(getattr(gathered, "tool_calls", None) or [])

    if not tool_calls:
        # 不需要工具：正文已逐块转发完毕
        yield DoneEvent(finish_reason="stop", usage=None)
        return

    # ---- ④ 执行工具（可能多个）----
    tool_messages: list[ToolMessage] = []
    for call in tool_calls:
        yield ToolCallEvent(name=call["name"], arguments=call["args"], call_id=call["id"])

        outcome = await registry.execute(
            call["name"], call["args"], call["id"], conversation_id=conversation_id
        )

        yield ToolResultEvent(
            call_id=outcome.call_id, ok=outcome.ok,
            summary=outcome.summary, content=outcome.content,
            elapsed_ms=outcome.elapsed_ms,
        )
        tool_messages.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))

    # ---- ⑤⑥ 第二次请求：不绑工具 ----
    # followup 里放的是累积出来的 gathered 本身（AIMessageChunk），
    # 而不是新构造的 AIMessage——这样模型若说过前言，前言也如实带上。
    followup = [*rendered, gathered, *tool_messages]

    try:
        async for chunk in model.astream(followup):   # ← 注意：model，不是 bound
            if chunk.text:
                yield DeltaEvent(text=chunk.text)
    except Exception as exc:  # noqa: BLE001
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    yield DoneEvent(finish_reason="stop", usage=None)
```

> **`ToolResultEvent` 为什么要带 `content`**：`api.py` 落库时需要用它构造 `ToolMessage`，而 `summary` 是截断过的、不能拿来落库。SSE 帧里只发 `summary`，不发 `content`（前端不需要全文）。

同时把 `app/chat.py` 顶部的 import 补上需要的符号：`AIMessage`、`ToolMessage`、`build_chat_prompt`、`prepare_messages`、`get_chat_model`（ch01 已有后三者）。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_chat_tools.py -v
uv run pytest -q
```
Expected: 新测试全过；**ch01 的 `tests/test_chat.py` 12 条必须继续通过**

- [ ] **Step 5: 用真实模型做一次端到端手测**

```bash
uv run python -c "
import asyncio
from langchain_core.messages import HumanMessage
from app.chat import stream_reply_with_tools
from app.tools import build_registry

async def main():
    reg = build_registry()
    async for e in stream_reply_with_tools(
        '订单 1001 的物流到哪了', [], registry=reg,
        token_budget=4096, conversation_id=1,
    ):
        print(type(e).__name__, e)

asyncio.run(main())
"
```
Expected: 打印出 `ToolCallEvent(name='query_logistics', ...)` → `ToolResultEvent(ok=True, ...)` → 若干 `DeltaEvent` → `DoneEvent`。**把原始输出贴进 `dev-notes/ch02.md`。**

- [ ] **Step 6: 提交**

```bash
git add app/chat.py tests/test_chat_tools.py
git commit -m "feat: 工具轮——流式探测 + 执行 + 回灌，第二次调用不绑工具

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: `api.py` SSE 扩展与工具上下文接线

**Files:**
- Modify: `app/api.py`
- Test: `tests/test_api_tools.py`（新建）

**Interfaces:**
- Consumes: `app.chat.stream_reply_with_tools` 及五个事件类型、`app.tools.build_registry`
- Produces: `POST /api/chat` 的分支逻辑；`event: tool` SSE 帧

- [ ] **Step 1: 写失败的测试 `tests/test_api_tools.py`**

```python
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.main import app
from app.providers import get_default_chat_model
from app.sessions import SessionStore
from app.api import get_store
from app.db.base import get_db


class FakeStore:
    def __init__(self):
        self._data = {}

    async def get(self, session_id):
        return list(self._data.get(session_id, []))

    async def append(self, session_id, *messages):
        self._data.setdefault(session_id, []).extend(messages)

    async def exists(self, session_id):
        return session_id in self._data


class ToolCallingModel:
    def __init__(self):
        self._bound = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if len(self._bound) > 0:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq", "args": '{"keyword": "退货"}',
                    "id": "call_x", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            yield AIMessageChunk(content="退货政策是七天无理由。")


@pytest.fixture
def client(db_session):
    """★ 必须注入真实的 db session：路由要用它做 get_or_create 与 build_registry。
    覆盖成 None 会让 get_or_create(None, ...) 直接崩。
    """
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_default_chat_model] = lambda: ToolCallingModel()
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def parse_sse(raw: str):
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((name, data))
    return out


def test_tool_events_appear_before_deltas(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert "tool" in names
    assert names.index("tool") < names.index("delta")
    assert names[-1] == "done"


def test_tool_start_and_end_frames(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    tools = [d for n, d in parse_sse(r.text) if n == "tool"]
    assert tools[0]["phase"] == "start"
    assert tools[0]["name"] == "query_faq"
    assert tools[0]["arguments"] == {"keyword": "退货"}
    assert tools[1]["phase"] == "end"
    assert tools[1]["call_id"] == tools[0]["call_id"]
    assert "ok" in tools[1]
    assert "summary" in tools[1]
    assert "content" not in tools[1], "SSE 帧只发 summary，不发回灌全文"


def test_full_message_sequence_is_stored(client):
    """★ 落库必须是**完整序列**：user → assistant(带 tool_calls) → tool → 最终 assistant。

    只存 user + 最终 assistant 的话，下一轮回放历史时 tool_call_id 会指向
    一条不存在的申请单，上游会直接报错。
    """
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert msgs[0].content == "退货政策是什么"
    assert msgs[1].tool_calls and msgs[1].tool_calls[0]["name"] == "query_faq"
    assert msgs[2].tool_call_id == msgs[1].tool_calls[0]["id"], "tool_call_id 必须对得上"
    assert msgs[3].content, "最终回答不能为空"


def test_no_tool_turn_stores_two_messages(client):
    """不需要工具的轮次：只存 user + assistant，没有 tool 消息。"""

    class PlainModel:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="您好，有什么可以帮您？")

    app.dependency_overrides[get_default_chat_model] = lambda: PlainModel()
    r = client.post("/api/chat", json={"message": "你好"})
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert msgs[1].content == "您好，有什么可以帮您？"
    assert msgs[1].tool_calls == []
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_api_tools.py -v
```
Expected: FAIL — `tool` 帧不存在

- [ ] **Step 3: 改 `app/api.py`**

在 `chat` 路由里，把 `stream_reply` 换成 `stream_reply_with_tools`，并处理两个新事件。关键改动：

```python
from app.chat import (
    DeltaEvent, DoneEvent, ErrorEvent, ToolCallEvent, ToolResultEvent,
    stream_reply_with_tools,
)
from app.tools import build_registry


@router.post("/chat")
async def chat(
    req: ChatRequest,
    store: SessionStore = Depends(get_store),
    model=Depends(get_default_chat_model),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    is_new_session = req.session_id is None
    session_id = req.session_id or str(uuid.uuid4())
    settings = get_settings()

    history = await store.get(session_id)

    # 会话必须存在，工具才拿得到 conversation_id（create_ticket 要用）
    from app.db.models import Conversation
    from sqlalchemy import select
    from app.repositories import conversation as conv_repo

    conv = await conv_repo.get_or_create(db, session_id, settings.default_user_id)

    registry = build_registry(session=db)

    async def event_stream() -> AsyncIterator[str]:
        if is_new_session:
            yield sse_frame("meta", {"session_id": session_id})

        # 落库用的累积器（spec §7.1 步骤 ⑧）
        # 工具轮之前的正文 → preamble；工具轮之后的正文 → final_answer。
        # 分开放是因为它们分属两条 assistant 消息：带 tool_calls 的那条，
        # 和最终回答那条。混成一条会让历史回放失真。
        preamble: list[str] = []
        final_answer: list[str] = []
        tool_calls_seen: list[dict] = []
        tool_messages: list[ToolMessage] = []
        in_tool_round = False
        completed = False

        try:
            async for event in stream_reply_with_tools(
                req.message, history, model=model, registry=registry,
                token_budget=settings.history_token_budget,
                conversation_id=conv.id,
            ):
                if isinstance(event, ToolCallEvent):
                    in_tool_round = True
                    tool_calls_seen.append({
                        "id": event.call_id, "name": event.name,
                        "args": event.arguments, "type": "tool_call",
                    })
                    yield sse_frame("tool", {
                        "phase": "start", "call_id": event.call_id,
                        "name": event.name, "arguments": event.arguments,
                    })
                elif isinstance(event, ToolResultEvent):
                    tool_messages.append(
                        ToolMessage(content=event.content, tool_call_id=event.call_id)
                    )
                    yield sse_frame("tool", {
                        "phase": "end", "call_id": event.call_id,
                        "ok": event.ok, "summary": event.summary,
                        "elapsed_ms": event.elapsed_ms,
                    })
                elif isinstance(event, DeltaEvent):
                    (final_answer if in_tool_round else preamble).append(event.text)
                    yield sse_frame("delta", {"text": event.text})
                elif isinstance(event, DoneEvent):
                    completed = True
                    yield sse_frame("done", {
                        "finish_reason": event.finish_reason, "usage": event.usage,
                    })
                elif isinstance(event, ErrorEvent):
                    yield sse_frame("error", {"code": event.code, "message": event.message})
        except asyncio.CancelledError:
            logger.info("客户端断开，丢弃本轮会话 %s 的写入", session_id)
            raise
        finally:
            if completed:
                # ★ 完整序列：user → assistant(带 tool_calls) → tool 结果* → 最终 assistant
                # 只存 user + 最终 assistant 会让下一轮回放历史时 tool_call_id
                # 指向一条不存在的申请单，上游会直接报错。
                to_store: list[BaseMessage] = [HumanMessage(req.message)]
                if tool_calls_seen:
                    to_store.append(AIMessage(
                        content="".join(preamble), tool_calls=tool_calls_seen
                    ))
                    to_store.extend(tool_messages)
                    to_store.append(AIMessage("".join(final_answer)))
                else:
                    to_store.append(AIMessage("".join(preamble)))
                await store.append(session_id, *to_store)
            await db.commit()

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={...})
```

> **三处需要注意**：
> 1. `build_registry(session=db)` 必须在**请求处理函数里**构造，不能放模块级——它要绑定当前请求的 session。
> 2. **落库的是完整消息序列，不是"user + 一句回复"**。缺了 assistant(tool_calls) 或 tool 消息，下一轮回放历史时 `tool_call_id` 会指向不存在的申请单，上游直接报错。
> 3. `db.commit()` 在 `finally` 里无条件执行：`get_or_create` 产生的**会话行即使这一轮失败也要保留**（否则下次请求又要重建），而业务消息只在 `completed` 时才写。两件事分开，不要用一句含糊的 `commit()` 带过。

> **`asyncio.CancelledError` 必须 `raise` 出去**——它是 `BaseException`，`except Exception` 捕不到，但显式写出来是为了记日志；`finally` 里的 `db.commit()` 仍会执行（保留会话行），而 `completed=False` 保证不写半截消息。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_api_tools.py -v
uv run pytest -q
```
Expected: 新测试 2 passed；全量约 **106 passed**

- [ ] **Step 5: 起服务用 curl 验证真实 SSE**

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
sleep 6
curl -sN -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"订单 1001 的物流到哪了"}' | head -20
kill %1
```
Expected: 前几帧是 `event: meta`，然后 `event: tool`（phase=start 与 end），然后 `event: delta`。**把原始输出贴进 `dev-notes/ch02.md`。**

- [ ] **Step 6: 提交**

```bash
git add app/api.py tests/test_api_tools.py
git commit -m "feat: /api/chat 接入工具轮，新增 event: tool 帧

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: 聊天页（Vibe Coding）

> **本任务不套 brainstorm / TDD / code review 流程**（用户明确要求）。做法是：先交付一个能跑的最小版本，用户看效果描述要改哪里，再改。

**Files:**
- Create: `web/index.html`
- Modify: `app/main.py`（挂载静态目录）

**Interfaces:**
- Consumes: `POST /api/chat` 的 SSE（`meta` / `tool` / `delta` / `done` / `error`）
- Produces: `GET /chat` 可访问的聊天页

- [ ] **Step 1: 挂载静态目录**

`app/main.py` 追加：

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_web = Path(__file__).resolve().parent.parent / "web"
if _web.is_dir():
    app.mount("/chat", StaticFiles(directory=_web, html=True), name="chat")
```

- [ ] **Step 2: 写 `web/index.html`（最小可用版）**

单文件、原生 JS、零构建。要点：

- 顶部标题，中间消息列表，底部输入框 + 发送按钮
- 用户消息靠右气泡，助手消息靠左气泡
- **工具徽章**：收到 `phase=start` 时插入一个"⏳ 查物流…"的灰色小徽章；收到 `phase=end` 时改成"🔧 查物流"，`ok=false` 时用警示色
- 中文标签映射表 + 未知工具名兜底显示原始 name
- 用 `fetch` + `ReadableStream` 读 SSE（不用 `EventSource`——它只支持 GET）
- 首帧 `meta` 里拿 `session_id`，存在变量里供后续轮次使用

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>智能客服</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 16px; }
  #list { display: flex; flex-direction: column; gap: 10px; min-height: 60vh; }
  .row { display: flex; }
  .row.user { justify-content: flex-end; }
  .bubble { max-width: 76%; padding: 8px 12px; border-radius: 12px; white-space: pre-wrap; }
  .user .bubble { background: #1677ff; color: #fff; }
  .bot .bubble { background: #f1f1f1; }
  .badge { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 10px;
           background: #e8e8e8; color: #666; margin-bottom: 4px; }
  .badge.err { background: #fff1f0; color: #cf1322; }
  form { display: flex; gap: 8px; margin-top: 16px; }
  input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }
  button { padding: 10px 18px; border: 0; border-radius: 8px; background: #1677ff; color: #fff; }
</style>
</head>
<body>
<h3>智能客服</h3>
<div id="list"></div>
<form id="form">
  <input id="input" placeholder="问点什么…" autocomplete="off">
  <button>发送</button>
</form>
<script>
const TOOL_LABELS = {
  query_order: "查订单", query_product: "查商品",
  query_logistics: "查物流", query_faq: "查常见问题",
  create_ticket: "建工单",
};
const labelOf = (n) => TOOL_LABELS[n] || n;

let sessionId = null;
const list = document.getElementById("list");

function addRow(who) {
  const row = document.createElement("div");
  row.className = "row " + who;
  const wrap = document.createElement("div");
  row.appendChild(wrap);
  list.appendChild(row);
  return wrap;
}

function addBadge(wrap, text, bad) {
  const b = document.createElement("div");
  b.className = "badge" + (bad ? " err" : "");
  b.textContent = text;
  wrap.appendChild(b);
  return b;
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  addRow("user").innerHTML = '<div class="bubble"></div>';
  list.lastChild.firstChild.textContent = text;

  const wrap = addRow("bot");
  let badge = null;
  let bubble = null;
  const badges = {};

  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sessionId ? { session_id: sessionId, message: text } : { message: text }),
  });

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop();
    for (const block of blocks) {
      let name = null, data = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
      }
      if (name === "meta") sessionId = data.session_id;
      else if (name === "tool") {
        if (data.phase === "start") {
          badges[data.call_id] = addBadge(wrap, "⏳ " + labelOf(data.name) + "…");
        } else if (badges[data.call_id]) {
          badges[data.call_id].textContent = "🔧 " + labelOf(data.name) + " · " + data.summary;
          if (!data.ok) badges[data.call_id].classList.add("err");
        }
      } else if (name === "delta") {
        if (!bubble) { bubble = document.createElement("div"); bubble.className = "bubble"; wrap.appendChild(bubble); }
        bubble.textContent += data.text;
      } else if (name === "error") {
        addBadge(wrap, "出错：" + data.message, true);
      }
    }
  }
});
</script>
</body>
</html>
```

- [ ] **Step 3: 起服务，浏览器验证三条验收标准**

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/chat`，依次问：

1. 「订单 1001 的物流到哪了」→ 应看到 `⏳ 查物流…` → `🔧 查物流 · <摘要>` 徽章 + 按结果作答
2. 「退货政策是什么」→ `🔧 查常见问题` 徽章
3. 「邮费是多少」→ `🔧 查常见问题` 徽章，但内容说查不到（**这是预期结果**）

> 徽章具体长什么样、颜色、位置，用户会提意见。**这一步之后再按用户描述迭代**，不要提前过度设计。

- [ ] **Step 4: 截图或记录实际效果到 `dev-notes/ch02.md`**

- [ ] **Step 5: 提交**

```bash
git add web/ app/main.py
git commit -m "feat: 聊天页（含工具轨迹徽章）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 12: 验收、README 与收尾

**Files:**
- Modify: `README.md`, `dev-notes/ch02.md`

**Interfaces:**
- Consumes: 全部
- Produces: 三条验收标准的实测证据

- [ ] **Step 1: README 补 ch02 段落**

````markdown
## ch02 · Function Calling 工具链

### 启动

```bash
docker compose up -d          # 起 MySQL 8.4
uv run python -m scripts.seed # 灌种子数据 + 建测试库
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

聊天页：<http://127.0.0.1:8000/chat>

### 五个工具

| 工具 | 作用 | 数据来源 |
|---|---|---|
| `query_order` | 查订单 | 工具内确定性生成（不接真实系统） |
| `query_product` | 查商品 | 同上 |
| `query_logistics` | 查物流 | 同上 |
| `query_faq` | 查常见问题 | `faq` 表，SQL `LIKE` |
| `create_ticket` | 建人工工单 | 写 `tickets` 表 + 会话转「已转人工」 |

### 已知限制

- `query_faq` 只有关键词匹配、无语义泛化——问「邮费是多少」查不到，这是**预期结果**，留给下一步升级
- 刷新页面会丢当前对话显示（数据已落库，缺的是读接口）
- 所有会话的 `user_id` 都是 `anonymous`（本章无鉴权）
````

- [ ] **Step 2: 跑三条验收，把原始输出贴进 `dev-notes/ch02.md`**

```bash
docker compose up -d && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
```

浏览器 `http://127.0.0.1:8000/chat`，逐条问、逐条记录：

| 验收 | 记录什么 |
|---|---|
| 1 | 徽章文字；回答内容；**并与 `query_logistics("1001")` 的返回值逐字对照**（确定性数据，必须对得上） |
| 2 | 徽章文字；回答是否引用了 `faq` 表里的条目 |
| 3 | 徽章仍出现（工具被调了）；工具返回什么；**模型如何应对这个空结果**——这是要留给下一步升级的关键观察 |

- [ ] **Step 3: 全量测试**

```bash
uv run pytest -q
```
Expected: 全部通过。把数字记进 dev-notes。

- [ ] **Step 4: 提交**

```bash
git add README.md dev-notes/ch02.md
git commit -m "docs: ch02 验收记录与 README

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

- [ ] **Step 5: 请求 code review**

```
使用 superpowers:requesting-code-review 对 ch02 全量改动做评审
```

把评审结论（含被驳回的建议）记进 `dev-notes/ch02.md`。

---

## 自查记录

**1. Spec 覆盖**

| Spec 节 | 覆盖任务 |
|---|---|
| §3 前置条件（Docker） | Task 1 |
| §4 四层架构 | Task 2/3/4/6/7 |
| §5 数据库（DDL/改动/种子） | Task 1 |
| §5.4 `user_id` 定 anonymous | Task 1（配置）+ Task 5（`append` 里用） |
| §5.5 `query_faq` 只搜 question | Task 3 + Task 7 |
| §5.6 测试库 | Task 1（seed 建库）+ Task 3（fixture） |
| §5.7 种子数据（含故意无邮费） | Task 1 |
| **§6.4 System Prompt 必须改写** | **Task 8** |
| §6.1~6.3 五个工具 + 基础设施 + 上下文注入 | Task 6/7 |
| §7.1 工具轮时序 | Task 9 |
| §7.2 先转发后决断（实测推翻原设计） | Task 9（有专门回归测试） |
| §7.4 第二次不绑工具 | Task 9（有专门测试） |
| §7.6 并行工具调用 | Task 9（循环执行全部 tool_calls） |
| §7.7 先正文后工具（实测行为） | Task 9 的 `test_preamble_then_tool_still_executes_tool` |
| §8 SSE 扩展 + 前端契约 | Task 10 + Task 11 |
| §9 会话存储切换（含破坏性） | Task 5 |
| §9.3 消息往返 | Task 4 |
| §10 配置 | Task 1 |
| §11 错误处理（回灌/边界/建单改状态） | Task 6/7/9/10 |
| §12.1 TDD 覆盖 | Task 2~10 各自的测试 |
| **§12.2 工具选择评估集** | **Task 8** |
| §13 三条验收 | Task 11 + Task 12 |
| §14 已知限制 | Task 12（README）|
| §15 未决项 | 分散在对应任务 |

**2. 占位符扫描**：无 TBD / TODO / "类似 Task N"。所有代码步骤含可运行代码。

> **已清除一处**：Task 9 初稿的 `test_empty_history_current_message_is_last` 是个空断言（它的桩模型不记录 messages，只断言"不抛异常"）。已换成 `test_current_message_is_last_in_rendered_prompt`，用会记录 messages 的桩做真实断言。

**3. 类型一致性**

- `ToolOutcome` 字段 `(call_id, ok, content, summary, elapsed_ms)` 在 Task 6 定义，Task 9 消费，一致
- `ToolResultEvent` 字段 `(call_id, ok, summary, content, elapsed_ms)` 在 Task 9 定义，Task 10 消费（帧里取 `summary`、落库取 `content`），一致
- `ToolSpec` 字段 `(tool, timeout_s, max_retries)` 在 Task 6 定义，Task 7 的 `build_registry()` 与 Task 9 的测试构造，一致
- `stream_reply_with_tools(message, history, *, model, registry, token_budget, conversation_id)` 在 Task 9 定义、Task 10 调用，一致
- `build_registry(session=None)` 在 Task 7 定义、Task 8/10 调用，一致
- `SessionStore.get/append/exists` 全 async 在 Task 5 定义，Task 5 改的 `api.py` 与 Task 10 的 `api.py` 都 `await`，一致
- `MySQLSessionStore(session=...)` 构造参数在 Task 5 定义、Task 5 的测试使用，一致
- `get_db` 作为 FastAPI 依赖在 Task 2 定义，Task 5 起被 `get_store` 与 chat 路由使用，Task 10 的测试 fixture 用真实 `db_session` 覆盖它，一致

**4. 明确标注的风险与例外**

- **Task 5 是破坏性变更**，计划里单独加了 ⚠️ 标注与受影响文件清单
- **Task 8 是计划执行中途新增的**，因为实测发现 ch01 的 System Prompt 与本章功能冲突（spec §6.4）
- **Task 11 是 Vibe Coding**，不套 brainstorm/TDD/code review，计划里明确写了"先做最小版本，用户提意见再改"
- **Task 9 的实现方式被实测推翻过一次**（"看第一个增量"→"先转发后决断"）。回归测试 `test_preamble_then_tool_still_executes_tool` 锁住这个改动；删掉它，实现可能在下次重构时退回错误分支，而症状（用户看到机器人说"我去查一下"然后没下文）不会让任何其他测试变红
