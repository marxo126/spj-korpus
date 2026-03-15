<?php
/**
 * Admin — CSV Import
 * Format: gloss_id, word_sk, theme_name, link_posunky, link_dictio
 * Max 500 rows.
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_admin();
require_csrf();

$pdo = get_db();
$file = $_FILES['csv_file'] ?? null;

if (!$file || $file['error'] !== UPLOAD_ERR_OK) {
    header('Location: /admin/?tab=words&error=no_file');
    exit;
}

$handle = fopen($file['tmp_name'], 'r');
if (!$handle) {
    header('Location: /admin/?tab=words&error=read_fail');
    exit;
}

$imported = 0;
$skipped = 0;
$row_num = 0;

// Cache existing themes
$themes_cache = [];
$theme_rows = $pdo->query('SELECT id, name FROM themes')->fetchAll();
foreach ($theme_rows as $t) {
    $themes_cache[mb_strtolower($t['name'])] = $t['id'];
}

while (($row = fgetcsv($handle)) !== false) {
    $row_num++;
    if ($row_num > 500) break;
    if (count($row) < 2) continue;

    $gloss_id = trim($row[0] ?? '');
    $word_sk = trim($row[1] ?? '');
    $theme_name = trim($row[2] ?? '');
    $link_posunky = trim($row[3] ?? '') ?: null;
    $link_dictio = trim($row[4] ?? '') ?: null;

    if (!$gloss_id || !$word_sk) { $skipped++; continue; }

    // Resolve theme
    $theme_id = null;
    if ($theme_name) {
        $key = mb_strtolower($theme_name);
        if (isset($themes_cache[$key])) {
            $theme_id = $themes_cache[$key];
        } else {
            // Create new theme
            $pdo->prepare('INSERT INTO themes (name) VALUES (?)')->execute([$theme_name]);
            $theme_id = (int) $pdo->lastInsertId();
            $themes_cache[$key] = $theme_id;
        }
    }

    // Insert sign (skip duplicates via INSERT IGNORE)
    $stmt = $pdo->prepare('INSERT IGNORE INTO signs (gloss_id, word_sk, theme_id, link_posunky, link_dictio) VALUES (?, ?, ?, ?, ?)');
    $stmt->execute([$gloss_id, $word_sk, $theme_id, $link_posunky, $link_dictio]);
    if ($stmt->rowCount() > 0) {
        $imported++;
    } else {
        $skipped++; // duplicate gloss_id
    }
}

fclose($handle);
header("Location: /admin/?tab=words&success=imported&count=$imported&skipped=$skipped");
exit;
