<?php
/**
 * Admin API — User role management
 * POST: user_id, role (user|researcher|admin)
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_admin();
require_csrf();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: /admin/?tab=users');
    exit;
}

$user_id = (int) ($_POST['user_id'] ?? 0);
$role = $_POST['role'] ?? '';

if ($user_id <= 0 || !in_array($role, ['user', 'researcher', 'admin'])) {
    header('Location: /admin/?tab=users&msg=role_error');
    exit;
}

// Can't change own role
if ($user_id === get_user_id()) {
    header('Location: /admin/?tab=users&msg=self_error');
    exit;
}

$pdo = get_db();

$is_admin = $role === 'admin' ? 1 : 0;
$is_researcher = $role === 'researcher' ? 1 : 0;

$stmt = $pdo->prepare('UPDATE users SET is_admin = ?, is_researcher = ? WHERE id = ?');
$stmt->execute([$is_admin, $is_researcher, $user_id]);

header('Location: /admin/?tab=users&msg=role_updated');
exit;
