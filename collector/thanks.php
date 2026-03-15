<?php
/**
 * SPJ Collector — Public thanks page
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

$pdo = get_db();
$stats = get_community_stats();

// Get public contributors (opted in)
$contributors = $pdo->query("
    SELECT public_name, total_recordings
    FROM users
    WHERE show_public_name = 1 AND public_name != '' AND total_recordings > 0
    ORDER BY total_recordings DESC
    LIMIT 100
")->fetchAll();

// Count anonymous contributors
$anonymous = $pdo->query("
    SELECT COUNT(*) FROM users
    WHERE (show_public_name = 0 OR public_name = '') AND total_recordings > 0
")->fetchColumn();

$page_title = 'Ďakujeme — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1 style="text-align: center; margin-bottom: 4px;">🤟 Ďakujeme!</h1>
<p style="text-align: center; color: var(--gray); margin-bottom: 20px;">
    Prispievatelia, vďaka ktorým rastie SPJ korpus
</p>

<!-- Community stats -->
<div class="card" style="margin-bottom: 20px;">
    <div class="stats-grid">
        <div>
            <span class="stat-num" style="color: var(--blue);"><?= number_format($stats['total_recordings'] ?? 0) ?></span>
            <span class="stat-label">nahrávok</span>
        </div>
        <div>
            <span class="stat-num" style="color: var(--green);"><?= number_format($stats['total_contributors'] ?? 0) ?></span>
            <span class="stat-label">prispievateľov</span>
        </div>
        <div>
            <span class="stat-num"><?= number_format($stats['total_signs'] ?? 0) ?></span>
            <span class="stat-label">posunkov</span>
        </div>
    </div>
</div>

<!-- Contributors list -->
<?php if (!empty($contributors)): ?>
    <?php foreach ($contributors as $i => $c): ?>
    <div class="thanks-item">
        <span class="name">
            <?php if ($i === 0): ?>🥇
            <?php elseif ($i === 1): ?>🥈
            <?php elseif ($i === 2): ?>🥉
            <?php else: ?><span style="display: inline-block; width: 28px;"></span>
            <?php endif; ?>
            <?= htmlspecialchars($c['public_name']) ?>
        </span>
        <span class="count"><?= number_format($c['total_recordings']) ?> nahrávok</span>
    </div>
    <?php endforeach; ?>
<?php endif; ?>

<?php if ($anonymous > 0): ?>
<div style="text-align: center; margin-top: 16px; padding: 14px; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--light-gray);">
    <p style="color: var(--gray); font-size: 15px;">
        + <?= number_format($anonymous) ?> anonymných prispievateľov
    </p>
</div>
<?php endif; ?>

<a href="<?= is_logged_in() ? '/record.php' : '/' ?>" class="btn btn-blue" style="margin-top: 24px;">
    🤟 Pridajte sa! →
</a>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
