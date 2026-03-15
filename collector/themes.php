<?php
/**
 * SPJ Collector — Theme Selection (main entry after login)
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
require_login();

$user = get_user();
$user_id = $user['id'];
$pdo = get_db();

// Get all themes with word counts
$themes = $pdo->query("
    SELECT t.id, t.name, t.emoji, t.sort_order,
           COUNT(s.id) as word_count,
           COALESCE(SUM(s.total_recordings), 0) as total_recordings
    FROM themes t
    LEFT JOIN signs s ON s.theme_id = t.id
    GROUP BY t.id, t.name, t.emoji, t.sort_order
    ORDER BY t.sort_order ASC
")->fetchAll();

// Get user progress per theme
$stmt = $pdo->prepare("
    SELECT theme_id, recordings_count, completed_at
    FROM user_theme_progress
    WHERE user_id = ?
");
$stmt->execute([$user_id]);
$progress_rows = $stmt->fetchAll();
$progress = [];
foreach ($progress_rows as $row) {
    $progress[$row['theme_id']] = $row;
}

// Find suggested theme (fewest total recordings globally)
$suggested_id = null;
$min_recordings = PHP_INT_MAX;
foreach ($themes as $theme) {
    if ($theme['total_recordings'] < $min_recordings && $theme['word_count'] > 0) {
        $min_recordings = $theme['total_recordings'];
        $suggested_id = $theme['id'];
    }
}

// Count uncategorized signs
$uncategorized = $pdo->query("SELECT COUNT(*) FROM signs WHERE theme_id IS NULL")->fetchColumn();

// Today's recordings
$stmt = $pdo->prepare('SELECT COUNT(*) FROM recordings WHERE user_id = ? AND DATE(created_at) = CURDATE()');
$stmt->execute([$user_id]);
$today_count = (int) $stmt->fetchColumn();

$page_title = 'Témy — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<div style="margin-bottom: 16px;">
    <div class="progress-info">
        <span class="label">Dnes</span>
        <span class="count" id="progress-count"><?= $today_count ?> / <?= DAILY_GOAL ?></span>
    </div>
    <div class="progress-bar">
        <div class="fill" style="width: <?= min(100, ($today_count / DAILY_GOAL) * 100) ?>%;"></div>
    </div>
</div>

<h2 style="margin-bottom: 16px;">Vyberte tému</h2>

<div class="theme-grid">
    <?php foreach ($themes as $theme):
        $user_count = $progress[$theme['id']]['recordings_count'] ?? 0;
        $completed = !empty($progress[$theme['id']]['completed_at']);
        $is_suggested = ($theme['id'] == $suggested_id);
        $pct = $theme['word_count'] > 0 ? min(100, ($user_count / $theme['word_count']) * 100) : 0;
    ?>
    <a href="/record.php?theme_id=<?= $theme['id'] ?>" class="theme-card <?= $is_suggested ? 'suggested' : '' ?> <?= $completed ? 'completed' : '' ?>">
        <div class="theme-emoji"><?= htmlspecialchars($theme['emoji']) ?></div>
        <div class="theme-info">
            <div class="theme-name"><?= htmlspecialchars($theme['name']) ?></div>
            <div class="theme-progress-text"><?= $user_count ?>/<?= $theme['word_count'] ?> nahraných</div>
            <div class="progress-bar" style="height: 6px;">
                <div class="fill" style="width: <?= $pct ?>%; background: <?= $completed ? 'var(--green)' : 'var(--blue)' ?>;"></div>
            </div>
        </div>
        <?php if ($completed): ?>
            <span class="theme-check">✅</span>
        <?php elseif ($is_suggested): ?>
            <span class="theme-badge">Odporúčané</span>
        <?php endif; ?>
    </a>
    <?php endforeach; ?>

    <?php if ($uncategorized > 0): ?>
    <a href="/record.php?theme_id=0" class="theme-card">
        <div class="theme-emoji">📋</div>
        <div class="theme-info">
            <div class="theme-name">Bez témy</div>
            <div class="theme-progress-text"><?= $uncategorized ?> slov</div>
        </div>
    </a>
    <?php endif; ?>
</div>

<!-- Overall stats -->
<div class="card" style="text-align: center; margin-top: 16px;">
    <p style="font-size: 13px; color: var(--gray);">Celkom nahrané</p>
    <p style="font-size: 28px; font-weight: 900;"><?= number_format($user['total_recordings']) ?></p>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
