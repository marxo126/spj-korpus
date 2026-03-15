<?php
/**
 * SPJ Collector — GDPR Data Export (Art. 20)
 * Returns JSON with all user's personal data.
 */
require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: /progress.php');
    exit;
}
if (!is_logged_in()) {
    header('Location: /index.php');
    exit;
}
require_csrf();

if (!check_rate_limit('export', 5, 3600)) {
    http_response_code(429);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Príliš veľa exportov. Skúste to neskôr.'], JSON_UNESCAPED_UNICODE);
    exit;
}

$user_id = get_user_id();
$pdo = get_db();

// Get user data
$stmt = $pdo->prepare('SELECT email, display_name, public_name, show_public_name,
    school, location, age_range, gender, dominant_hand,
    consent_service, consent_service_date, consent_biometric, consent_biometric_date,
    consent_retention, consent_retention_date, consent_date,
    created_at, last_active, total_recordings FROM users WHERE id = ?');
$stmt->execute([$user_id]);
$user_data = $stmt->fetch(PDO::FETCH_ASSOC);

// Get recordings list (metadata only, not video files)
$stmt = $pdo->prepare('SELECT r.id, s.word_sk, s.gloss_id, r.status,
    r.duration_ms, r.created_at FROM recordings r
    JOIN signs s ON r.sign_id = s.id WHERE r.user_id = ?
    ORDER BY r.created_at DESC');
$stmt->execute([$user_id]);
$recordings = $stmt->fetchAll(PDO::FETCH_ASSOC);

// Get validations given
$stmt = $pdo->prepare('SELECT recording_id, vote, created_at
    FROM validations WHERE validator_id = ? ORDER BY created_at DESC');
$stmt->execute([$user_id]);
$validations = $stmt->fetchAll(PDO::FETCH_ASSOC);

$export = [
    'export_date' => date('c'),
    'user' => $user_data,
    'recordings' => $recordings,
    'validations' => $validations,
];

header('Content-Type: application/json; charset=utf-8');
header('Content-Disposition: attachment; filename="spj-export-' . date('Y-m-d') . '.json"');
echo json_encode($export, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
exit;
