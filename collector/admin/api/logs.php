<?php
/**
 * Admin API — Log management
 * POST action=clear: Delete all logs
 */

require_once __DIR__ . '/../../includes/config.php';
require_once __DIR__ . '/../../includes/admin_auth.php';
require_admin();
require_csrf();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: /admin/?tab=logs');
    exit;
}

$action = $_POST['action'] ?? '';

if ($action === 'clear') {
    $pdo = get_db();
    $pdo->exec('TRUNCATE TABLE error_log');
    header('Location: /admin/?tab=logs&msg=cleared');
    exit;
}

header('Location: /admin/?tab=logs');
