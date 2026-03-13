<?php
/**
 * SPJ Collector — Video Upload API
 * POST /api/upload.php
 * Form data: video (file), sign_id (int)
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

// Must be logged in
if (!is_logged_in()) {
    http_response_code(401);
    echo json_encode(['error' => 'Nie ste prihlásený'], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed'], JSON_UNESCAPED_UNICODE);
    exit;
}

require_csrf();

if (!check_rate_limit('upload', 100, 3600)) {
    http_response_code(429);
    echo json_encode(['error' => 'Príliš veľa nahrávok. Skúste to neskôr.'], JSON_UNESCAPED_UNICODE);
    exit;
}

$user_id = get_user_id();
$sign_id = (int) ($_POST['sign_id'] ?? 0);
$file = $_FILES['video'] ?? null;

// Validate
if (!$file || $file['error'] !== UPLOAD_ERR_OK) {
    http_response_code(400);
    echo json_encode(['error' => 'Žiadny súbor'], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($sign_id <= 0) {
    http_response_code(400);
    echo json_encode(['error' => 'Neplatný znak'], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($file['size'] > MAX_VIDEO_SIZE) {
    http_response_code(400);
    echo json_encode(['error' => 'Súbor je príliš veľký (max 10 MB)'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Validate file type
$allowed_types = ['video/mp4', 'video/webm', 'video/quicktime'];
$finfo = finfo_open(FILEINFO_MIME_TYPE);
$mime = finfo_file($finfo, $file['tmp_name']);
finfo_close($finfo);

if (!in_array($mime, $allowed_types)) {
    http_response_code(400);
    echo json_encode(['error' => 'Neplatný formát videa'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Validate duration (client-sent, but also check server-side via ffprobe if available)
$duration_ms = (int) ($_POST['duration_ms'] ?? 3000);
if ($duration_ms < MIN_VIDEO_DURATION_MS || $duration_ms > MAX_VIDEO_DURATION_MS) {
    http_response_code(400);
    echo json_encode(['error' => 'Neplatná dĺžka videa'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Server-side duration check with ffprobe (if available)
$ffprobe = trim(shell_exec('which ffprobe 2>/dev/null') ?: '');
if ($ffprobe && is_executable($ffprobe)) {
    $tmp = escapeshellarg($file['tmp_name']);
    $probe_duration = (float) trim(shell_exec("$ffprobe -v quiet -show_entries format=duration -of csv=p=0 $tmp 2>/dev/null") ?: '0');
    if ($probe_duration > 0) {
        $probe_ms = (int) ($probe_duration * 1000);
        if ($probe_ms < MIN_VIDEO_DURATION_MS || $probe_ms > MAX_VIDEO_DURATION_MS) {
            http_response_code(400);
            echo json_encode(['error' => 'Video je príliš krátke alebo dlhé'], JSON_UNESCAPED_UNICODE);
            exit;
        }
        $duration_ms = $probe_ms; // use server-verified duration
    }
}

// Generate unique filename
$ext = $mime === 'video/mp4' || $mime === 'video/quicktime' ? 'mp4' : 'webm';
$filename = uniqid('rec_', true) . '.' . $ext;
$dest = UPLOAD_PENDING . '/' . $filename;

// Ensure upload directory exists
if (!is_dir(UPLOAD_PENDING)) {
    mkdir(UPLOAD_PENDING, 0755, true);
}

// Move file
if (!move_uploaded_file($file['tmp_name'], $dest)) {
    http_response_code(500);
    echo json_encode(['error' => 'Nepodarilo sa uložiť súbor'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Insert DB row
try {
    $pdo = get_db();

    $stmt = $pdo->prepare('
        INSERT INTO recordings (user_id, sign_id, video_filename, duration_ms)
        VALUES (?, ?, ?, ?)
    ');
    $stmt->execute([
        $user_id,
        $sign_id,
        $filename,
        $duration_ms,
    ]);

    // Update counters
    $pdo->prepare('UPDATE users SET total_recordings = total_recordings + 1, last_active = CURDATE() WHERE id = ?')
        ->execute([$user_id]);
    $pdo->prepare('UPDATE signs SET total_recordings = total_recordings + 1 WHERE id = ?')
        ->execute([$sign_id]);

    // Update user_theme_progress
    $stmt = $pdo->prepare('SELECT theme_id FROM signs WHERE id = ?');
    $stmt->execute([$sign_id]);
    $sign_row = $stmt->fetch();
    if ($sign_row && $sign_row['theme_id']) {
        $tid = (int) $sign_row['theme_id'];
        $pdo->prepare("
            INSERT INTO user_theme_progress (user_id, theme_id, recordings_count)
            VALUES (?, ?, 1)
            ON DUPLICATE KEY UPDATE recordings_count = recordings_count + 1
        ")->execute([$user_id, $tid]);

        // Check if theme completed
        $stmt = $pdo->prepare("
            SELECT utp.recordings_count, COUNT(s.id) as word_count
            FROM user_theme_progress utp
            JOIN signs s ON s.theme_id = utp.theme_id
            WHERE utp.user_id = ? AND utp.theme_id = ?
            GROUP BY utp.theme_id
        ");
        $stmt->execute([$user_id, $tid]);
        $tp = $stmt->fetch();
        if ($tp && $tp['recordings_count'] >= $tp['word_count'] && $tp['word_count'] > 0) {
            $pdo->prepare("
                UPDATE user_theme_progress SET completed_at = NOW()
                WHERE user_id = ? AND theme_id = ? AND completed_at IS NULL
            ")->execute([$user_id, $tid]);
        }
    }

    echo json_encode(['ok' => true, 'id' => $pdo->lastInsertId()], JSON_UNESCAPED_UNICODE);

} catch (PDOException $e) {
    // Remove uploaded file if DB insert fails
    @unlink($dest);
    http_response_code(500);
    echo json_encode(['error' => 'Chyba databázy'], JSON_UNESCAPED_UNICODE);
}
