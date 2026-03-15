<?php
/**
 * Admin — Storage usage API (JSON)
 *
 * Currently unused — admin/stats.php computes storage inline.
 * Available for AJAX polling (e.g. real-time storage bar updates).
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';

if (!is_logged_in() || !is_researcher()) {
    http_response_code(403);
    echo json_encode(['error' => 'Forbidden'], JSON_UNESCAPED_UNICODE);
    exit;
}

$upload_bytes = 0;
$upload_dir = UPLOAD_DIR;
if (is_dir($upload_dir)) {
    $output = shell_exec("du -sb " . escapeshellarg($upload_dir) . " 2>/dev/null");
    if ($output) $upload_bytes = (int) explode("\t", $output)[0];
}

$limit_bytes = STORAGE_LIMIT_GB * 1024 * 1024 * 1024;
$pct = $limit_bytes > 0 ? round(($upload_bytes / $limit_bytes) * 100, 1) : 0;

echo json_encode([
    'used_bytes' => $upload_bytes,
    'limit_bytes' => $limit_bytes,
    'used_gb' => round($upload_bytes / (1024*1024*1024), 2),
    'limit_gb' => STORAGE_LIMIT_GB,
    'percent' => $pct,
    'warning' => $pct >= 80,
    'critical' => $pct >= 90,
], JSON_UNESCAPED_UNICODE);
