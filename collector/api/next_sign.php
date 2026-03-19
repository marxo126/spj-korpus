<?php
/**
 * SPJ Collector — Get next sign to record
 * GET /api/next_sign.php?theme_id=X&sign_id=X
 * Returns sign with fewest recordings that user hasn't done.
 * When all signs are done once, offers signs below target for variant recordings.
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

// Theme filtering (shared by both queries)
$theme_id = isset($_GET['theme_id']) ? (int) $_GET['theme_id'] : null;
$where_theme = '';
$theme_params = [];
if ($theme_id !== null) {
    if ($theme_id === 0) {
        $where_theme = 'AND s.theme_id IS NULL';
    } else {
        $where_theme = 'AND s.theme_id = ?';
        $theme_params = [$theme_id];
    }
}

// 1) First: signs the user hasn't recorded yet (priority)
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
$stmt->execute(array_merge([$user_id], $theme_params));
$signs = $stmt->fetchAll();

if (!empty($signs)) {
    $next = $signs[array_rand($signs)];
    echo json_encode($next, JSON_UNESCAPED_UNICODE);
    exit;
}

// 2) Fallback: signs below target that user can add variants to
//    Sorted by fewest user recordings (so they record different signs, not same one 10x)
$stmt = $pdo->prepare("
    SELECT s.id, s.gloss_id, s.word_sk, s.link_posunky, s.link_dictio,
           s.category, s.total_recordings, s.target_recordings,
           COUNT(r.id) as my_count
    FROM signs s
    LEFT JOIN recordings r ON r.sign_id = s.id AND r.user_id = ?
    WHERE s.total_recordings < s.target_recordings
    $where_theme
    GROUP BY s.id
    ORDER BY my_count ASC, s.total_recordings ASC
    LIMIT 5
");
$stmt->execute(array_merge([$user_id], $theme_params));
$signs = $stmt->fetchAll();

if (!empty($signs)) {
    $next = $signs[array_rand($signs)];
    $next['variant'] = true; // signal to UI that this is a variant recording
    echo json_encode($next, JSON_UNESCAPED_UNICODE);
    exit;
}

// 3) Truly all done — all signs at target
echo json_encode(['error' => 'all_done', 'message' => 'Všetky posunky sú nahrané!'], JSON_UNESCAPED_UNICODE);
