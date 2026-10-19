CREATE TABLE IF NOT EXISTS low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    reason VARCHAR(50) NOT NULL COMMENT 'no_results/low_score/model_insufficient',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
