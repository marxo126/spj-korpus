<?php
/**
 * SPJ Collector — JS Error Logging API
 * POST: { message, source, url, extra }
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';
require_once __DIR__ . '/../includes/error_logger.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

// Rate limit: max 20 error reports per minute per session
if (!check_rate_limit('js_error', 20, 60)) {
    http_response_code(429);
    echo json_encode(['error' => 'Too many errors']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true);
if (!$input || empty($input['message'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Missing message']);
    exit;
}

$message = mb_substr($input['message'] ?? '', 0, 2000);
$level = in_array($input['level'] ?? '', ['error', 'warning', 'info']) ? $input['level'] : 'error';
$extra = [];
if (!empty($input['stack'])) $extra['stack'] = mb_substr($input['stack'], 0, 2000);
if (!empty($input['filename'])) $extra['filename'] = mb_substr($input['filename'], 0, 500);
if (!empty($input['lineno'])) $extra['lineno'] = (int) $input['lineno'];
if (!empty($input['colno'])) $extra['colno'] = (int) $input['colno'];

// Override URL from client if provided
if (!empty($input['url'])) {
    $_SERVER['REQUEST_URI'] = mb_substr($input['url'], 0, 500);
}

log_error($message, $level, 'js', $extra ?: null);

echo json_encode(['ok' => true]);
