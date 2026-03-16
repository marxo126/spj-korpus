<?php
/**
 * SPJ Collector — Authenticated video proxy
 * GET /api/video.php?file=rec_xxx.mp4&dir=pending|approved
 */

require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/admin_auth.php';

if (!is_logged_in()) {
    http_response_code(401);
    exit;
}

$filename = basename($_GET['file'] ?? '');
$dir = $_GET['dir'] ?? 'pending';

if (!$filename || !preg_match('/^rec_[a-f0-9]+\.[a-f0-9]+\.(mp4|webm)$/', $filename)) {
    http_response_code(400);
    exit;
}

// GDPR: regular users can only view their own videos; researchers/admins can view all
if (!is_researcher()) {
    start_session();
    $authorized = $_SESSION['authorized_videos'] ?? [];
    if (!in_array($filename, $authorized, true)) {
        $pdo = get_db();
        $stmt = $pdo->prepare('SELECT id FROM recordings WHERE video_filename = ? AND user_id = ?');
        $stmt->execute([$filename, get_user_id()]);
        if (!$stmt->fetch()) {
            http_response_code(403);
            exit;
        }
        $authorized[] = $filename;
        $_SESSION['authorized_videos'] = $authorized;
    }
}

$base = $dir === 'approved' ? UPLOAD_APPROVED : UPLOAD_PENDING;
$path = $base . '/' . $filename;

if (!file_exists($path)) {
    http_response_code(404);
    exit;
}

$ext = pathinfo($filename, PATHINFO_EXTENSION);
$mime = $ext === 'mp4' ? 'video/mp4' : 'video/webm';
$size = filesize($path);

// Safari REQUIRES Range request support for video playback
header('Content-Type: ' . $mime);
header('Accept-Ranges: bytes');
header('Cache-Control: private, max-age=3600');

if (isset($_SERVER['HTTP_RANGE'])) {
    // Parse Range header: bytes=START-END
    preg_match('/bytes=(\d+)-(\d*)/', $_SERVER['HTTP_RANGE'], $m);
    $start = (int) $m[1];
    $end = $m[2] !== '' ? (int) $m[2] : $size - 1;
    if ($start > $end || $start >= $size) {
        http_response_code(416);
        header("Content-Range: bytes */$size");
        exit;
    }
    $length = $end - $start + 1;
    http_response_code(206);
    header("Content-Range: bytes $start-$end/$size");
    header("Content-Length: $length");
    $fp = fopen($path, 'rb');
    fseek($fp, $start);
    echo fread($fp, $length);
    fclose($fp);
} else {
    header('Content-Length: ' . $size);
    readfile($path);
}
