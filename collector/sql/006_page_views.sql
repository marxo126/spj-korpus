-- Page view tracking & server metrics
CREATE TABLE IF NOT EXISTS page_views (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(500) NOT NULL,
    page VARCHAR(100) DEFAULT NULL,       -- extracted page name (e.g. 'record', 'themes')
    user_id INT DEFAULT NULL,
    session_id VARCHAR(64) DEFAULT NULL,  -- hashed session ID for unique visitor counting
    ip_hash VARCHAR(64) DEFAULT NULL,     -- SHA-256 of IP (GDPR-safe, no raw IPs)
    user_agent VARCHAR(500) DEFAULT NULL,
    referrer VARCHAR(500) DEFAULT NULL,
    device_type ENUM('desktop','mobile','tablet') DEFAULT 'desktop',
    response_time_ms INT DEFAULT NULL,    -- server-side response time
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created (created_at),
    INDEX idx_page (page),
    INDEX idx_session (session_id),
    INDEX idx_ip_hash (ip_hash, created_at),
    INDEX idx_created_page (created_at, page),
    INDEX idx_created_device (created_at, device_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Aggregated daily stats (populated by cron or on-demand)
CREATE TABLE IF NOT EXISTS daily_metrics (
    date DATE NOT NULL PRIMARY KEY,
    page_views INT DEFAULT 0,
    unique_visitors INT DEFAULT 0,       -- distinct ip_hash
    unique_sessions INT DEFAULT 0,       -- distinct session_id
    recordings_count INT DEFAULT 0,
    new_users INT DEFAULT 0,
    avg_response_ms INT DEFAULT NULL,
    top_pages JSON DEFAULT NULL,         -- [{page, count}, ...]
    top_referrers JSON DEFAULT NULL,     -- [{referrer, count}, ...]
    device_breakdown JSON DEFAULT NULL,  -- {desktop: N, mobile: N, tablet: N}
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
