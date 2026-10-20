-- ch04 · 从 ch03 升级已有数据库（幂等，可重复执行）
--
-- scripts/schema.sql 只对全新安装（首次初始化数据卷）生效；已有 ch03 库
-- 不会自动获得下面两项结构。执行方式：
--   docker exec -i custermerhelp-mysql mysql -uapp -papppw customer_service \
--     < scripts/alter_ch04.sql
-- 测试库同理（库名 customer_service_test）。

-- 1) messages.metadata：引用溯源 / 检索策略 / 分数（ch04 §3.5）
SET @c := (SELECT COUNT(*) FROM information_schema.columns
           WHERE table_schema = DATABASE() AND table_name='messages' AND column_name='metadata');
SET @s := IF(@c = 0,
  'ALTER TABLE messages ADD COLUMN metadata JSON NULL AFTER tool_call_id', 'SELECT 1');
PREPARE stmt FROM @s; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 2) 低置信度问题池：拒答（no_results/low_score/model_insufficient）入池
CREATE TABLE IF NOT EXISTS low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    reason VARCHAR(50) NOT NULL COMMENT 'no_results/low_score/model_insufficient',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
