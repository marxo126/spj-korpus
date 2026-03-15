<?php
/**
 * SPJ Collector — Configuration
 */

// Database — load from local secrets file if exists, else env vars (Docker), else defaults
$_secrets_file = __DIR__ . '/secrets.php';
if (file_exists($_secrets_file)) {
    require_once $_secrets_file; // defines DB_HOST, DB_NAME, DB_USER, DB_PASS
} else {
    define('DB_HOST', getenv('DB_HOST') ?: 'localhost');
    define('DB_NAME', getenv('DB_NAME') ?: 'spj_collector');
    define('DB_USER', getenv('DB_USER') ?: 'your_db_user');
    define('DB_PASS', getenv('DB_PASS') ?: 'your_db_pass');
}

// Paths
define('UPLOAD_DIR', __DIR__ . '/../uploads');
define('UPLOAD_PENDING', UPLOAD_DIR . '/pending');
define('UPLOAD_APPROVED', UPLOAD_DIR . '/approved');

// Limits
define('MAX_VIDEO_SIZE', 10 * 1024 * 1024); // 10 MB
define('MAX_VIDEO_DURATION_MS', 8000);       // 8 seconds
define('MIN_VIDEO_DURATION_MS', 1000);       // 1 second
define('DAILY_GOAL', 20);
define('TARGET_RECORDINGS_PER_SIGN', 50);

// Validation
define('VOTES_TO_APPROVE', 3);
define('VOTES_TO_REJECT', 3);
define('MIN_RECORDINGS_TO_VALIDATE', 20);

// Storage monitoring
define('STORAGE_LIMIT_GB', 18); // leaves 2 GB headroom for app files and DB

// Google OAuth (fill in your credentials)
define('GOOGLE_CLIENT_ID', '');
define('GOOGLE_CLIENT_SECRET', '');
define('GOOGLE_REDIRECT_URI', '');

// Site
define('SITE_NAME', 'SPJ Collector');
define('SITE_URL', 'https://zber.spj.sk'); // production domain

// Session (must be set before session_start)
if (session_status() === PHP_SESSION_NONE) {
    ini_set('session.cookie_httponly', 1);
    ini_set('session.cookie_secure', isset($_SERVER['HTTPS']) ? 1 : 0);
    ini_set('session.use_strict_mode', 1);
}
