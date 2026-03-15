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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, maximum-scale=5">
    <meta name="theme-color" content="#111827">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title><?= htmlspecialchars($page_title ?? SITE_NAME) ?></title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/css/style.css?v=<?= time() ?>">
    <link rel="manifest" href="/manifest.json">
    <link rel="icon" href="/img/icon-192.png" type="image/png">
    <?php if ($logged_in): ?>
    <meta name="csrf-token" content="<?= htmlspecialchars(csrf_token()) ?>">
    <?php endif; ?>
</head>
<body>
    <a href="#main-content" class="skip-link">Preskoč na obsah</a>
    <!-- Top nav (desktop) -->
    <nav class="top-nav" aria-label="Hlavná navigácia">
        <a href="/" class="logo"><img src="/img/spj-logo.png" alt="SPJ" style="height:28px;width:28px;vertical-align:middle;margin-right:6px;">Zber SPJ</a>
        <?php if ($logged_in): ?>
        <div class="nav-links desktop-only">
            <a href="/themes.php" class="<?= $current_page === 'themes' ? 'active' : '' ?>">Témy</a>
            <a href="/record.php" class="<?= $current_page === 'record' ? 'active' : '' ?>">Nahrať</a>
            <a href="/validate.php" class="<?= $current_page === 'validate' ? 'active' : '' ?>">Overiť</a>
            <a href="/thanks.php" class="<?= $current_page === 'thanks' ? 'active' : '' ?>">Ďakujeme</a>
            <a href="/progress.php" class="<?= $current_page === 'progress' ? 'active' : '' ?>">Profil</a>
            <?php if ($user && (!empty($user['is_admin']) || !empty($user['is_researcher']))): ?>
            <a href="/admin/" style="color: #FBBF24;">Admin</a>
            <?php endif; ?>
        </div>
        <a href="/api/auth.php?action=logout" class="desktop-only" style="color:#9CA3AF;font-size:13px;text-decoration:none;margin-left:8px;" title="Odhlásiť sa">↪ Odhlásiť</a>
        <?php endif; ?>
        <button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" aria-label="Prepnúť tmavý/svetlý režim" title="Tmavý/svetlý režim">🌙</button>
    </nav>
    <script>
    function toggleTheme() {
        const html = document.documentElement;
        const isDark = html.classList.toggle('dark');
        localStorage.setItem('spj_theme', isDark ? 'dark' : 'light');
        document.getElementById('theme-toggle').textContent = isDark ? '☀️' : '🌙';
    }
    (function() {
        const saved = localStorage.getItem('spj_theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (saved === 'dark' || (!saved && prefersDark)) {
            document.documentElement.classList.add('dark');
            document.addEventListener('DOMContentLoaded', function() {
                var btn = document.getElementById('theme-toggle');
                if (btn) btn.textContent = '☀️';
            });
        }
    })();
    </script>

    <main class="page-content" id="main-content">
