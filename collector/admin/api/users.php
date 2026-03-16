<?php
/**
 * Admin API — User role management
 * POST: user_id, role (user|researcher|admin)
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_once __DIR__ . '/../../includes/error_logger.php';
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

// Get target user info for audit log
$stmt = $pdo->prepare('SELECT email, is_admin as was_admin, is_researcher as was_researcher FROM users WHERE id = ?');
$stmt->execute([$user_id]);
$target = $stmt->fetch();

$pdo->prepare('UPDATE users SET is_admin = ?, is_researcher = ? WHERE id = ?')
    ->execute([$is_admin, $is_researcher, $user_id]);

$admin_user = get_user();
$role_labels = ['user' => 'Používateľ', 'researcher' => 'Výskumník', 'admin' => 'Admin'];
$old_role = $target['was_admin'] ? 'admin' : ($target['was_researcher'] ? 'researcher' : 'user');
log_error(
    "Rola zmenená: {$target['email']} — {$role_labels[$old_role]} → {$role_labels[$role]} — zmenil {$admin_user['email']}",
    'info', 'api',
    ['action' => 'role_change', 'target_user_id' => $user_id, 'target_email' => $target['email'],
     'old_role' => $old_role, 'new_role' => $role, 'admin' => $admin_user['email']]
);

header('Location: /admin/?tab=users&msg=role_updated');
exit;
