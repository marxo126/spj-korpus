<?php
/**
 * SPJ Collector — Public health check endpoint
 *
 * GET /api/status.php — returns JSON with basic health info.
 * No auth required (for uptime monitoring services).
 * Detailed checks only shown to authenticated admins.
 */

header('Content-Type: application/json');
header('Cache-Control: no-store');

require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/analytics.php';

$status = 'ok';
$checks = [];

// DB connectivity
try {
    $pdo = get_db();
    $pdo->query('SELECT 1');
    $checks['database'] = 'ok';
} catch (\Throwable $e) {
    $checks['database'] = 'error';
    $status = 'degraded';
}

// Disk space (quota-based, not shared server disk)
$disk = get_disk_usage();
if ($disk['used_pct'] !== null) {
    $checks['disk'] = $disk['used_pct'] < 90 ? 'ok' : ($disk['used_pct'] < 95 ? 'warning' : 'critical');
    if ($disk['used_pct'] >= 95) $status = 'degraded';
} else {
    $checks['disk'] = 'unknown';
}

// Upload dirs writable
$checks['uploads_writable'] = is_writable(UPLOAD_PENDING) ? 'ok' : 'error';
if ($checks['uploads_writable'] !== 'ok') $status = 'degraded';

// Public response: only status + timestamp (no server internals)
$response = [
    'status' => $status,
    'timestamp' => gmdate('c'),
];

// Detailed checks only for authenticated admin requests
session_start();
if (!empty($_SESSION['user_id'])) {
    require_once __DIR__ . '/../includes/admin_auth.php';
    if (is_admin()) {
        $checks['disk_used_pct'] = $disk['used_pct'];
        $checks['storage_used_mb'] = round($disk['used'] / (1024 * 1024));
        $checks['storage_limit_mb'] = STORAGE_LIMIT_GB * 1024;
        $response['checks'] = $checks;
    }
}

echo json_encode($response, JSON_PRETTY_PRINT);
