-- SPJ Collector — MySQL Schema
-- Run this on your WebSupport.sk MySQL database
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    google_id VARCHAR(255) UNIQUE DEFAULT NULL,
    display_name VARCHAR(100) DEFAULT '',
    public_name VARCHAR(200) DEFAULT '',
    show_public_name TINYINT(1) DEFAULT 0,
    school VARCHAR(200) DEFAULT '',
    location VARCHAR(200) DEFAULT '',
    age_range ENUM('under_18','18-25','26-35','36-50','50+') DEFAULT NULL,
    gender ENUM('woman','man','neutral') DEFAULT NULL,
    is_admin TINYINT(1) DEFAULT 0,
    is_researcher TINYINT(1) DEFAULT 0,
    dominant_hand ENUM('right','left') DEFAULT 'right',
    total_recordings INT DEFAULT 0,
    password_reset_token VARCHAR(64) DEFAULT NULL,
    password_reset_expires DATETIME DEFAULT NULL,
    consent_service TINYINT(1) DEFAULT 0,
    consent_biometric TINYINT(1) DEFAULT 0,
    consent_retention TINYINT(1) DEFAULT 0,
    consent_date DATETIME DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATE DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS themes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    emoji VARCHAR(50) DEFAULT '',
    sort_order INT DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS signs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gloss_id VARCHAR(100) UNIQUE NOT NULL,
    word_sk VARCHAR(200) NOT NULL,
    link_posunky VARCHAR(500) DEFAULT NULL,
    link_dictio VARCHAR(500) DEFAULT NULL,
    category VARCHAR(100) DEFAULT NULL,
    theme_id INT DEFAULT NULL,
    sort_order_in_theme INT DEFAULT 0,
    total_recordings INT DEFAULT 0,
    target_recordings INT DEFAULT 50,
    FOREIGN KEY (theme_id) REFERENCES themes(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS recordings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT DEFAULT NULL,
    sign_id INT NOT NULL,
    video_filename VARCHAR(255) NOT NULL,
    duration_ms INT DEFAULT NULL,
    status ENUM('pending','approved','rejected') DEFAULT 'pending',
    validations_up INT DEFAULT 0,
    validations_down INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (sign_id) REFERENCES signs(id) ON DELETE CASCADE,
    INDEX idx_status (status),
    INDEX idx_sign_id (sign_id),
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS validations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    recording_id INT NOT NULL,
    validator_id INT DEFAULT NULL,
    vote TINYINT(1) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_vote (recording_id, validator_id),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY (validator_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_theme_progress (
    user_id INT NOT NULL,
    theme_id INT NOT NULL,
    recordings_count INT DEFAULT 0,
    completed_at DATETIME DEFAULT NULL,
    PRIMARY KEY (user_id, theme_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (theme_id) REFERENCES themes(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
