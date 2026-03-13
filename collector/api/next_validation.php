<?php
/**
 * SPJ Collector — Get next recording to validate
 * GET /api/next_validation.php
 * Returns a pending recording that user hasn't voted on and didn't create.
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/auth.php';

if (!is_logged_in()) {
    http_response_code(401);
    echo json_encode(['error' => 'Nie ste prihlásený'], JSON_UNESCAPED_UNICODE);
    exit;
}

$user_id = get_user_id();
$pdo = get_db();

$stmt = $pdo->prepare("
    SELECT r.id, r.video_filename, r.duration_ms,
           s.word_sk, s.gloss_id, s.link_posunky, s.link_dictio
    FROM recordings r
    JOIN signs s ON r.sign_id = s.id
    WHERE r.status = 'pending'
      AND r.user_id != ?
      AND r.id NOT IN (
          SELECT recording_id FROM validations WHERE validator_id = ?
      )
    ORDER BY r.validations_up + r.validations_down ASC, RAND()
    LIMIT 1
");
$stmt->execute([$user_id, $user_id]);
$recording = $stmt->fetch();

if (!$recording) {
    echo json_encode(['error' => 'no_more'], JSON_UNESCAPED_UNICODE);
    exit;
}

echo json_encode($recording, JSON_UNESCAPED_UNICODE);
