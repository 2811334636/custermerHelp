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
  metadata        JSON            NULL                     COMMENT '消息附加数据(ch04: citations/retrieval_strategy/top_score)',
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

-- ch03 · 知识库

-- 知识块:MySQL 为权威源,Milvus 只按 chunk_id 存向量
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  chunk_id      VARCHAR(64)  NOT NULL COMMENT '确定性主键：source:ordinal',
  category      VARCHAR(255) NOT NULL,
  questions     VARCHAR(512) NOT NULL,
  answer        TEXT         NOT NULL,
  section_path  VARCHAR(512) NOT NULL,
  content_type  ENUM('policy','faq','mined') NOT NULL,
  is_key_clause TINYINT(1)   NOT NULL DEFAULT 0,
  prev_chunk_id VARCHAR(64)  NULL,
  next_chunk_id VARCHAR(64)  NULL,
  status        ENUM('待向量化','已向量化') NOT NULL DEFAULT '待向量化',
  vector_id     VARCHAR(64)  NULL COMMENT '回填的 Milvus 主键',
  content_hash  CHAR(64)     NOT NULL COMMENT 'embedding 文本的 sha256',
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id),
  KEY idx_status (status),
  KEY idx_content_type (content_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ch04 · 低置信度问题池
-- 与 scripts/add_low_confidence_table.sql（历史迁移）保持同一份 DDL；
-- 新装库由本文件建表，已有库用 scripts/alter_ch04.sql 补齐。
CREATE TABLE IF NOT EXISTS low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    reason VARCHAR(50) NOT NULL COMMENT 'no_results/low_score/model_insufficient',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 挖掘暂存区:LLM 从会话中抽出的问答候选,人审/去重后进 knowledge_chunks
CREATE TABLE IF NOT EXISTS mined_qa_staging (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question        VARCHAR(512) NOT NULL,
  answer          TEXT         NOT NULL,
  category        VARCHAR(64)  NOT NULL,
  conversation_id BIGINT UNSIGNED NULL,
  status          ENUM('待入库','已入库','已丢弃') NOT NULL DEFAULT '待入库',
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
