<?php
/**
 * SPJ Collector — Admin Panel
 */

require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/admin_auth.php';
require_researcher();

$allowed_tabs = is_admin() ? ['words', 'themes', 'videos', 'stats'] : ['videos', 'stats'];
$tab = $_GET['tab'] ?? $allowed_tabs[0];
if (!in_array($tab, $allowed_tabs)) $tab = $allowed_tabs[0];

$page_title = 'Admin — ' . SITE_NAME;
require_once __DIR__ . '/../includes/header.php';
?>

<h1 style="margin-bottom: 16px;">Admin Panel</h1>

<div class="admin-tabs">
    <?php if (is_admin()): ?>
    <a href="/admin/?tab=words" class="admin-tab <?= $tab === 'words' ? 'active' : '' ?>">Slová</a>
    <a href="/admin/?tab=themes" class="admin-tab <?= $tab === 'themes' ? 'active' : '' ?>">Témy</a>
    <?php endif; ?>
    <a href="/admin/?tab=videos" class="admin-tab <?= $tab === 'videos' ? 'active' : '' ?>">Videá</a>
    <a href="/admin/?tab=stats" class="admin-tab <?= $tab === 'stats' ? 'active' : '' ?>">Štatistiky</a>
</div>

<?php
switch ($tab) {
    case 'words':
        require __DIR__ . '/words.php';
        break;
    case 'themes':
        require __DIR__ . '/themes.php';
        break;
    case 'videos':
        require __DIR__ . '/videos.php';
        break;
    case 'stats':
        require __DIR__ . '/stats.php';
        break;
}
?>

<?php require_once __DIR__ . '/../includes/footer.php'; ?>
