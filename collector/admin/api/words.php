<?php
/**
 * Admin — Word CRUD API
 * POST with action: add, edit, delete
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_admin();
require_csrf();

$pdo = get_db();
$action = $_POST['action'] ?? '';

switch ($action) {
    case 'add':
        $gloss_id = trim($_POST['gloss_id'] ?? '');
        $word_sk = trim($_POST['word_sk'] ?? '');
        $theme_id = (int) ($_POST['theme_id'] ?? 0) ?: null;
        $link_posunky = trim($_POST['link_posunky'] ?? '') ?: null;
        $link_dictio = trim($_POST['link_dictio'] ?? '') ?: null;

        if (!$gloss_id || !$word_sk) {
            header('Location: /admin/?tab=words&error=missing_fields');
            exit;
        }

        try {
            $stmt = $pdo->prepare('INSERT INTO signs (gloss_id, word_sk, theme_id, link_posunky, link_dictio) VALUES (?, ?, ?, ?, ?)');
            $stmt->execute([$gloss_id, $word_sk, $theme_id, $link_posunky, $link_dictio]);
            header('Location: /admin/?tab=words&success=added');
        } catch (PDOException $e) {
            header('Location: /admin/?tab=words&error=duplicate');
        }
        break;

    case 'edit':
        $sign_id = (int) ($_POST['sign_id'] ?? 0);
        $gloss_id = trim($_POST['gloss_id'] ?? '');
        $word_sk = trim($_POST['word_sk'] ?? '');
        $theme_id = (int) ($_POST['theme_id'] ?? 0) ?: null;
        $link_posunky = trim($_POST['link_posunky'] ?? '') ?: null;
        $link_dictio = trim($_POST['link_dictio'] ?? '') ?: null;

        $check = $pdo->prepare('SELECT id FROM signs WHERE id = ?');
        $check->execute([$sign_id]);
        if (!$check->fetch()) {
            header('Location: /admin/?tab=words&error=not_found');
            exit;
        }

        $stmt = $pdo->prepare('UPDATE signs SET gloss_id = ?, word_sk = ?, theme_id = ?, link_posunky = ?, link_dictio = ? WHERE id = ?');
        $stmt->execute([$gloss_id, $word_sk, $theme_id, $link_posunky, $link_dictio, $sign_id]);
        header('Location: /admin/?tab=words&success=updated');
        break;

    case 'delete':
        $sign_id = (int) ($_POST['sign_id'] ?? 0);

        // Get sign's theme_id once (same for all recordings)
        $sign_theme = $pdo->prepare('SELECT theme_id FROM signs WHERE id = ?');
        $sign_theme->execute([$sign_id]);
        $theme_row = $sign_theme->fetch();
        $tid = ($theme_row && $theme_row['theme_id']) ? $theme_row['theme_id'] : null;

        // Delete video files and decrement counters before CASCADE
        $stmt = $pdo->prepare('SELECT r.id, r.video_filename, r.user_id FROM recordings r WHERE r.sign_id = ?');
        $stmt->execute([$sign_id]);
        $recordings = $stmt->fetchAll();

        foreach ($recordings as $rec) {
            // Delete video files
            foreach ([UPLOAD_PENDING, UPLOAD_APPROVED] as $dir) {
                $path = $dir . '/' . $rec['video_filename'];
                if (file_exists($path)) unlink($path);
            }
            // Decrement user counter
            $pdo->prepare('UPDATE users SET total_recordings = GREATEST(0, total_recordings - 1) WHERE id = ?')
                ->execute([$rec['user_id']]);
            // Decrement user_theme_progress + clear completed_at
            if ($tid) {
                $pdo->prepare('UPDATE user_theme_progress SET recordings_count = GREATEST(0, recordings_count - 1), completed_at = NULL WHERE user_id = ? AND theme_id = ?')
                    ->execute([$rec['user_id'], $tid]);
            }
        }

        $pdo->prepare('DELETE FROM signs WHERE id = ?')->execute([$sign_id]);
        header('Location: /admin/?tab=words&success=deleted');
        break;
}
exit;
