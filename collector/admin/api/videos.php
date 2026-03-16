<?php
/**
 * Admin — Video approve/delete API
 * Logs all actions to error_log for audit trail.
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_once __DIR__ . '/../../includes/error_logger.php';
require_researcher();
require_csrf();

$pdo = get_db();
$action = $_POST['action'] ?? '';
$recording_id = (int) ($_POST['recording_id'] ?? 0);
$admin_user = get_user();
$admin_name = $admin_user['display_name'] ?: $admin_user['email'];

if ($recording_id <= 0) {
    header('Location: /admin/?tab=videos&error=invalid');
    exit;
}

$stmt = $pdo->prepare('SELECT r.*, s.theme_id, s.word_sk FROM recordings r JOIN signs s ON r.sign_id = s.id WHERE r.id = ?');
$stmt->execute([$recording_id]);
$rec = $stmt->fetch();

if (!$rec) {
    header('Location: /admin/?tab=videos&error=not_found');
    exit;
}

// Get recording owner info
$owner_email = '';
if ($rec['user_id']) {
    $stmt2 = $pdo->prepare('SELECT email FROM users WHERE id = ?');
    $stmt2->execute([$rec['user_id']]);
    $owner = $stmt2->fetch();
    $owner_email = $owner ? $owner['email'] : 'deleted user';
}

switch ($action) {
    case 'approve':
        // Move file from pending to approved
        $src = UPLOAD_PENDING . '/' . $rec['video_filename'];
        $dst = UPLOAD_APPROVED . '/' . $rec['video_filename'];
        if (file_exists($src)) {
            if (!is_dir(UPLOAD_APPROVED)) mkdir(UPLOAD_APPROVED, 0755, true);
            rename($src, $dst);
        }
        $pdo->prepare('UPDATE recordings SET status = ? WHERE id = ?')->execute(['approved', $recording_id]);

        log_error(
            "Video #{$recording_id} SCHVÁLENÉ — \"{$rec['word_sk']}\" od {$owner_email} — schválil {$admin_name}",
            'info', 'api',
            ['action' => 'approve', 'recording_id' => $recording_id, 'word' => $rec['word_sk'],
             'owner' => $owner_email, 'admin' => $admin_name, 'filename' => $rec['video_filename']]
        );
        break;

    case 'reject':
        $pdo->prepare('UPDATE recordings SET status = ? WHERE id = ?')->execute(['rejected', $recording_id]);

        log_error(
            "Video #{$recording_id} ZAMIETNUTÉ — \"{$rec['word_sk']}\" od {$owner_email} — zamietol {$admin_name}",
            'info', 'api',
            ['action' => 'reject', 'recording_id' => $recording_id, 'word' => $rec['word_sk'],
             'owner' => $owner_email, 'admin' => $admin_name, 'filename' => $rec['video_filename']]
        );
        break;

    case 'delete':
        // Delete video file
        foreach ([UPLOAD_PENDING, UPLOAD_APPROVED] as $dir) {
            $path = $dir . '/' . $rec['video_filename'];
            if (file_exists($path)) unlink($path);
        }
        // Decrement counters
        $pdo->prepare('UPDATE signs SET total_recordings = GREATEST(0, total_recordings - 1) WHERE id = ?')
            ->execute([$rec['sign_id']]);
        if ($rec['user_id']) {
            $pdo->prepare('UPDATE users SET total_recordings = GREATEST(0, total_recordings - 1) WHERE id = ?')
                ->execute([$rec['user_id']]);
        }
        if ($rec['theme_id'] && $rec['user_id']) {
            $pdo->prepare('UPDATE user_theme_progress SET recordings_count = GREATEST(0, recordings_count - 1), completed_at = NULL WHERE user_id = ? AND theme_id = ?')
                ->execute([$rec['user_id'], $rec['theme_id']]);
        }
        // Delete DB row
        $pdo->prepare('DELETE FROM recordings WHERE id = ?')->execute([$recording_id]);

        log_error(
            "Video #{$recording_id} VYMAZANÉ — \"{$rec['word_sk']}\" od {$owner_email} — vymazal {$admin_name}",
            'warning', 'api',
            ['action' => 'delete', 'recording_id' => $recording_id, 'word' => $rec['word_sk'],
             'owner' => $owner_email, 'admin' => $admin_name, 'filename' => $rec['video_filename']]
        );
        break;
}

header('Location: /admin/?tab=videos');
exit;
