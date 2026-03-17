<?php
/**
 * SPJ Collector — Get next recording to validate
 * GET /api/next_validation.php
 * Returns a pending recording that user hasn't voted on and didn't create.
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/admin_auth.php';

if (!is_logged_in()) {
    http_response_code(401);
    echo json_encode(['error' => 'Nie ste prihlásený'], JSON_UNESCAPED_UNICODE);
    exit;
}

if (!is_researcher()) {
    http_response_code(403);
    echo json_encode(['error' => 'Prístup len pre výskumníkov'], JSON_UNESCAPED_UNICODE);
    exit;
}

$user_id = get_user_id();
$pdo = get_db();

$stmt = $pdo->prepare("
    SELECT r.id, r.video_filename, r.duration_ms,
           r.validations_up, r.validations_down,
           s.word_sk, s.gloss_id, s.link_posunky, s.link_dictio,
           u.display_name AS contributor_name, u.dominant_hand,
           (SELECT COUNT(*) FROM recordings r2 WHERE r2.sign_id = r.sign_id) AS variants_total,
           (SELECT SUM(r2.status = 'approved') FROM recordings r2 WHERE r2.sign_id = r.sign_id) AS variants_approved
    FROM recordings r
    JOIN signs s ON r.sign_id = s.id
    LEFT JOIN users u ON r.user_id = u.id
    WHERE r.status = 'pending'
      AND (r.user_id IS NULL OR r.user_id != ?)
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

$recording['variants_total'] = (int) $recording['variants_total'];
$recording['variants_approved'] = (int) ($recording['variants_approved'] ?? 0);

echo json_encode($recording, JSON_UNESCAPED_UNICODE);
