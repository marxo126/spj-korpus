<?php
/**
 * SPJ Collector — User theme progress API
 * GET /api/theme_progress.php
 * Returns per-theme recording counts for the current user.
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

if (!is_logged_in()) {
    http_response_code(401);
    echo json_encode(['error' => 'Nie ste prihlásený'], JSON_UNESCAPED_UNICODE);
    exit;
}

$user_id = get_user_id();
$pdo = get_db();

// Get all themes with word counts and user progress
$stmt = $pdo->prepare("
    SELECT t.id, t.name, t.emoji,
           COUNT(s.id) as word_count,
           COALESCE(MAX(utp.recordings_count), 0) as user_count,
           MAX(utp.completed_at) as completed_at
    FROM themes t
    LEFT JOIN signs s ON s.theme_id = t.id
    LEFT JOIN user_theme_progress utp ON utp.theme_id = t.id AND utp.user_id = ?
    GROUP BY t.id, t.name, t.emoji, t.sort_order
    ORDER BY t.sort_order ASC
");
$stmt->execute([$user_id]);
$themes = $stmt->fetchAll();

echo json_encode(['themes' => $themes], JSON_UNESCAPED_UNICODE);
