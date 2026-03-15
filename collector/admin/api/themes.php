<?php
/**
 * Admin — Theme CRUD API
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_admin();
require_csrf();

$pdo = get_db();
$action = $_POST['action'] ?? '';

switch ($action) {
    case 'add':
        $name = trim($_POST['name'] ?? '');
        $emoji = trim($_POST['emoji'] ?? '');
        $sort_order = (int) ($_POST['sort_order'] ?? 0);
        if ($name) {
            $pdo->prepare('INSERT INTO themes (name, emoji, sort_order) VALUES (?, ?, ?)')
                ->execute([$name, $emoji, $sort_order]);
        }
        break;

    case 'delete':
        $theme_id = (int) ($_POST['theme_id'] ?? 0);
        $pdo->prepare('DELETE FROM themes WHERE id = ?')->execute([$theme_id]);
        break;

    case 'move_up':
        $theme_id = (int) ($_POST['theme_id'] ?? 0);
        $pdo->beginTransaction();
        $current = $pdo->prepare('SELECT sort_order FROM themes WHERE id = ?');
        $current->execute([$theme_id]);
        $order = (int) $current->fetchColumn();
        if ($order > 1) {
            $pdo->prepare('UPDATE themes SET sort_order = sort_order + 1 WHERE sort_order = ?')->execute([$order - 1]);
            $pdo->prepare('UPDATE themes SET sort_order = ? WHERE id = ?')->execute([$order - 1, $theme_id]);
        }
        $pdo->commit();
        break;

    case 'move_down':
        $theme_id = (int) ($_POST['theme_id'] ?? 0);
        $pdo->beginTransaction();
        $current = $pdo->prepare('SELECT sort_order FROM themes WHERE id = ?');
        $current->execute([$theme_id]);
        $order = (int) $current->fetchColumn();
        $max = (int) $pdo->query('SELECT MAX(sort_order) FROM themes')->fetchColumn();
        if ($order < $max) {
            $pdo->prepare('UPDATE themes SET sort_order = sort_order - 1 WHERE sort_order = ?')->execute([$order + 1]);
            $pdo->prepare('UPDATE themes SET sort_order = ? WHERE id = ?')->execute([$order + 1, $theme_id]);
        }
        $pdo->commit();
        break;
}

header('Location: /admin/?tab=themes');
exit;
