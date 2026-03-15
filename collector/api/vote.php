<?php
/**
 * SPJ Collector — Validation vote API
 * POST /api/vote.php
 * Body: recording_id (int), vote (1=good, 0=bad)
 */

header('Content-Type: application/json');
require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

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

$user_id = get_user_id();
$recording_id = (int) ($_POST['recording_id'] ?? 0);
$vote = (int) ($_POST['vote'] ?? -1);

if ($recording_id <= 0 || ($vote !== 0 && $vote !== 1)) {
    http_response_code(400);
    echo json_encode(['error' => 'Neplatné údaje'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Check user has enough recordings to validate
$pdo = get_db();
$stmt = $pdo->prepare('SELECT total_recordings FROM users WHERE id = ?');
$stmt->execute([$user_id]);
$user = $stmt->fetch();

if (!$user || $user['total_recordings'] < MIN_RECORDINGS_TO_VALIDATE) {
    http_response_code(403);
    echo json_encode(['error' => "Potrebujete aspoň " . MIN_RECORDINGS_TO_VALIDATE . " nahrávok"], JSON_UNESCAPED_UNICODE);
    exit;
}

// Can't validate own recordings or already-processed ones
$stmt = $pdo->prepare('SELECT user_id, status FROM recordings WHERE id = ?');
$stmt->execute([$recording_id]);
$rec = $stmt->fetch();
if (!$rec || $rec['user_id'] == $user_id || $rec['status'] !== 'pending') {
    http_response_code(400);
    echo json_encode(['error' => 'Nahrávka nie je dostupná na hlasovanie'], JSON_UNESCAPED_UNICODE);
    exit;
}

try {
    $pdo->beginTransaction();

    // Insert vote (unique constraint prevents double voting)
    $stmt = $pdo->prepare('INSERT INTO validations (recording_id, validator_id, vote) VALUES (?, ?, ?)');
    $stmt->execute([$recording_id, $user_id, $vote]);

    // Update vote counts
    if ($vote) {
        $pdo->prepare('UPDATE recordings SET validations_up = validations_up + 1 WHERE id = ?')->execute([$recording_id]);
    } else {
        $pdo->prepare('UPDATE recordings SET validations_down = validations_down + 1 WHERE id = ?')->execute([$recording_id]);
    }

    // Check if auto-approve/reject threshold reached
    $stmt = $pdo->prepare('SELECT validations_up, validations_down, video_filename FROM recordings WHERE id = ?');
    $stmt->execute([$recording_id]);
    $rec = $stmt->fetch();

    $new_status = null;
    if ($rec['validations_up'] >= VOTES_TO_APPROVE) {
        $new_status = 'approved';
    } elseif ($rec['validations_down'] >= VOTES_TO_REJECT) {
        $new_status = 'rejected';

        // Decrement counters
        $stmt2 = $pdo->prepare('SELECT user_id, sign_id FROM recordings WHERE id = ?');
        $stmt2->execute([$recording_id]);
        $rec_info = $stmt2->fetch();
        if ($rec_info) {
            $pdo->prepare('UPDATE signs SET total_recordings = GREATEST(0, total_recordings - 1) WHERE id = ?')
                ->execute([$rec_info['sign_id']]);
            $pdo->prepare('UPDATE users SET total_recordings = GREATEST(0, total_recordings - 1) WHERE id = ?')
                ->execute([$rec_info['user_id']]);

            // Decrement user_theme_progress
            $stmt3 = $pdo->prepare('SELECT theme_id FROM signs WHERE id = ?');
            $stmt3->execute([$rec_info['sign_id']]);
            $sign_row = $stmt3->fetch();
            if ($sign_row && $sign_row['theme_id']) {
                $pdo->prepare("
                    UPDATE user_theme_progress
                    SET recordings_count = GREATEST(0, recordings_count - 1), completed_at = NULL
                    WHERE user_id = ? AND theme_id = ?
                ")->execute([$rec_info['user_id'], $sign_row['theme_id']]);
            }
        }
    }

    if ($new_status) {
        $pdo->prepare('UPDATE recordings SET status = ? WHERE id = ?')->execute([$new_status, $recording_id]);
    }

    $pdo->commit();

    // File operations AFTER commit — filesystem is not transactional
    if ($new_status === 'approved') {
        $src = UPLOAD_PENDING . '/' . $rec['video_filename'];
        $dst = UPLOAD_APPROVED . '/' . $rec['video_filename'];
        if (file_exists($src)) {
            if (!is_dir(UPLOAD_APPROVED)) mkdir(UPLOAD_APPROVED, 0755, true);
            rename($src, $dst);
        }
    } elseif ($new_status === 'rejected') {
        $path = UPLOAD_PENDING . '/' . $rec['video_filename'];
        if (file_exists($path)) unlink($path);
    }

    echo json_encode(['ok' => true, 'new_status' => $new_status], JSON_UNESCAPED_UNICODE);

} catch (PDOException $e) {
    if ($pdo->inTransaction()) {
        $pdo->rollBack();
    }
    if (str_contains($e->getMessage(), 'Duplicate entry')) {
        http_response_code(409);
        echo json_encode(['error' => 'Už ste hlasovali'], JSON_UNESCAPED_UNICODE);
    } else {
        http_response_code(500);
        echo json_encode(['error' => 'Chyba databázy'], JSON_UNESCAPED_UNICODE);
    }
}
