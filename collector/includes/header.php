<?php
/**
 * SPJ Collector — Shared HTML header + nav
 */

require_once __DIR__ . '/auth.php';
start_session();

$current_page = basename($_SERVER['SCRIPT_NAME'], '.php');
$logged_in = is_logged_in();
$user = $logged_in ? get_user() : null;
?>
<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#111827">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title><?= htmlspecialchars($page_title ?? SITE_NAME) ?></title>
    <link rel="stylesheet" href="/css/style.css">
    <link rel="manifest" href="/manifest.json">
    <link rel="icon" href="/icons/icon-192.png" type="image/png">
    <?php if ($logged_in): ?>
    <meta name="csrf-token" content="<?= htmlspecialchars(csrf_token()) ?>">
    <?php endif; ?>
</head>
<body>
    <!-- Top nav (desktop) -->
    <nav class="top-nav">
        <a href="/" class="logo">🤟 SPJ</a>
        <?php if ($logged_in): ?>
        <div class="nav-links desktop-only">
            <a href="/themes.php" class="<?= $current_page === 'themes' ? 'active' : '' ?>">Témy</a>
            <a href="/record.php" class="<?= $current_page === 'record' ? 'active' : '' ?>">Nahrať</a>
            <a href="/validate.php" class="<?= $current_page === 'validate' ? 'active' : '' ?>">Overiť</a>
            <a href="/thanks.php" class="<?= $current_page === 'thanks' ? 'active' : '' ?>">Ďakujeme</a>
            <a href="/progress.php" class="<?= $current_page === 'progress' ? 'active' : '' ?>">Profil</a>
            <?php if ($user && !empty($user['is_admin'])): ?>
            <a href="/admin.php" class="<?= $current_page === 'admin' ? 'active' : '' ?>" style="color: #FBBF24;">Admin</a>
            <?php endif; ?>
        </div>
        <span class="desktop-only user-name"><?= htmlspecialchars($user['display_name'] ?? $user['email'] ?? '') ?></span>
        <?php endif; ?>
    </nav>

    <main class="page-content">
