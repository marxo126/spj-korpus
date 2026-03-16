-- Error logging for admin panel
CREATE TABLE IF NOT EXISTS error_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    level ENUM('error','warning','info') DEFAULT 'error',
    source ENUM('php','js','api') DEFAULT 'php',
    message TEXT NOT NULL,
    url VARCHAR(500) DEFAULT NULL,
    user_id INT DEFAULT NULL,
    user_agent VARCHAR(500) DEFAULT NULL,
    extra JSON DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created (created_at),
    INDEX idx_level (level),
    INDEX idx_source (source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
