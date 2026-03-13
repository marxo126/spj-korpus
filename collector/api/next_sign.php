<?php
/**
 * SPJ Collector — Get next sign to record
 * GET /api/next_sign.php?theme_id=X&sign_id=X
 * Returns sign with fewest recordings that user hasn't done.
 * Supports theme filtering and direct sign selection.
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

// Direct sign selection (e.g. from a link)
$sign_id = (int) ($_GET['sign_id'] ?? 0);
if ($sign_id > 0) {
    $stmt = $pdo->prepare("
        SELECT s.id, s.gloss_id, s.word_sk, s.link_posunky, s.link_dictio,
               s.category, s.total_recordings, s.target_recordings
        FROM signs s
        WHERE s.id = ?
    ");
    $stmt->execute([$sign_id]);
    $sign = $stmt->fetch();
    if ($sign) {
        echo json_encode($sign, JSON_UNESCAPED_UNICODE);
        exit;
    }
}

// Theme filtering
$theme_id = isset($_GET['theme_id']) ? (int) $_GET['theme_id'] : null;

$where_theme = '';
$params = [$user_id];

if ($theme_id !== null) {
    if ($theme_id === 0) {
        // Uncategorized signs
        $where_theme = 'AND s.theme_id IS NULL';
    } else {
        $where_theme = 'AND s.theme_id = ?';
        $params[] = $theme_id;
    }
}

// Select from top 5 least-recorded signs that user hasn't done
$stmt = $pdo->prepare("
    SELECT s.id, s.gloss_id, s.word_sk, s.link_posunky, s.link_dictio,
           s.category, s.total_recordings, s.target_recordings
    FROM signs s
    WHERE s.id NOT IN (
        SELECT sign_id FROM recordings WHERE user_id = ?
    )
    $where_theme
    ORDER BY s.total_recordings ASC
    LIMIT 5
");
$stmt->execute($params);
$signs = $stmt->fetchAll();

if (empty($signs)) {
    echo json_encode(['error' => 'all_done', 'message' => 'Všetky znaky sú nahrané!'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Pick randomly from top 5 (avoids everyone doing same sign)
$next = $signs[array_rand($signs)];
echo json_encode($next, JSON_UNESCAPED_UNICODE);
