<?php
/**
 * SPJ Collector — Authenticated video proxy
 * GET /api/video.php?file=rec_xxx.mp4&dir=pending|approved
 */

require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

if (!is_logged_in()) {
    http_response_code(401);
    exit;
}

// NOTE: Any authenticated user can view any video. This is intentional —
// the validation workflow (/validate.php) requires users to view and vote
// on other users' recordings. If per-user access control is needed later,
// add ownership check here: verify $_SESSION['user_id'] owns the recording.

$filename = basename($_GET['file'] ?? '');
$dir = $_GET['dir'] ?? 'pending';

if (!$filename || !preg_match('/^rec_[a-f0-9]+\.[a-f0-9]+\.(mp4|webm)$/', $filename)) {
    http_response_code(400);
    exit;
}

$base = $dir === 'approved' ? UPLOAD_APPROVED : UPLOAD_PENDING;
$path = $base . '/' . $filename;

if (!file_exists($path)) {
    http_response_code(404);
    exit;
}

$ext = pathinfo($filename, PATHINFO_EXTENSION);
$mime = $ext === 'mp4' ? 'video/mp4' : 'video/webm';

header('Content-Type: ' . $mime);
header('Content-Length: ' . filesize($path));
header('Cache-Control: private, max-age=3600');
readfile($path);
