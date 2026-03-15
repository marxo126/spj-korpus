<?php
/**
 * Admin — Statistics tab (included by admin/index.php)
 */

$pdo = get_db();

// Storage
$upload_bytes = 0;
$upload_dir = UPLOAD_DIR;
if (is_dir($upload_dir)) {
    $output = shell_exec("du -sb " . escapeshellarg($upload_dir) . " 2>/dev/null");
    if ($output) $upload_bytes = (int) explode("\t", $output)[0];
}
$storage_limit_bytes = STORAGE_LIMIT_GB * 1024 * 1024 * 1024;
$storage_pct = $storage_limit_bytes > 0 ? ($upload_bytes / $storage_limit_bytes) * 100 : 0;
$storage_class = $storage_pct >= 90 ? 'danger' : ($storage_pct >= 80 ? 'warn' : 'ok');

// Recording counts
$total_recordings = (int) $pdo->query('SELECT COUNT(*) FROM recordings')->fetchColumn();
$today_recordings = (int) $pdo->query("SELECT COUNT(*) FROM recordings WHERE DATE(created_at) = CURDATE()")->fetchColumn();
$week_recordings = (int) $pdo->query("SELECT COUNT(*) FROM recordings WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)")->fetchColumn();

// User counts
$total_users = (int) $pdo->query('SELECT COUNT(*) FROM users')->fetchColumn();
$active_today = (int) $pdo->query("SELECT COUNT(*) FROM users WHERE last_active = CURDATE()")->fetchColumn();
$active_week = (int) $pdo->query("SELECT COUNT(*) FROM users WHERE last_active >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)")->fetchColumn();

// Coverage
$coverage_50 = (int) $pdo->query('SELECT COUNT(*) FROM signs WHERE total_recordings >= 50')->fetchColumn();
$coverage_20 = (int) $pdo->query('SELECT COUNT(*) FROM signs WHERE total_recordings >= 20 AND total_recordings < 50')->fetchColumn();
$coverage_10 = (int) $pdo->query('SELECT COUNT(*) FROM signs WHERE total_recordings >= 10 AND total_recordings < 20')->fetchColumn();
$coverage_lt10 = (int) $pdo->query('SELECT COUNT(*) FROM signs WHERE total_recordings < 10')->fetchColumn();

// Per-theme bar chart
$theme_stats = $pdo->query("
    SELECT t.name, t.emoji, COALESCE(SUM(s.total_recordings), 0) as total
    FROM themes t
    LEFT JOIN signs s ON s.theme_id = t.id
    GROUP BY t.id, t.name, t.emoji
    ORDER BY total DESC
")->fetchAll();
$max_theme = max(array_column($theme_stats, 'total') ?: [1]);

// Top contributors
$top_users = $pdo->query("
    SELECT display_name, email, total_recordings
    FROM users
    WHERE total_recordings > 0
    ORDER BY total_recordings DESC
    LIMIT 10
")->fetchAll();
?>

<!-- Storage -->
<h3 style="margin-bottom:8px;">Úložisko</h3>
<?php if ($storage_pct >= 80): ?>
<div style="background:<?= $storage_pct >= 90 ? '#FEE2E2' : '#FEF3C7' ?>;color:<?= $storage_pct >= 90 ? 'var(--red)' : '#A16207' ?>;padding:10px 14px;border-radius:10px;font-weight:600;font-size:14px;margin-bottom:8px;">
    ⚠️ Pripravte migráciu videí (<?= round($storage_pct) ?>% obsadené)
</div>
<?php endif; ?>
<div class="storage-bar">
    <div class="fill <?= $storage_class ?>" style="width:<?= min(100, $storage_pct) ?>%;"></div>
</div>
<p style="font-size:13px;color:var(--gray);margin-bottom:20px;">
    <?= number_format($upload_bytes / (1024*1024*1024), 2) ?> GB / <?= STORAGE_LIMIT_GB ?> GB
</p>

<!-- Stat cards -->
<div class="stat-cards-grid">
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;color:var(--blue);"><?= number_format($total_recordings) ?></div>
        <div style="font-size:12px;color:var(--gray);">Celkom nahrávok</div>
    </div>
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;color:var(--green);"><?= $today_recordings ?></div>
        <div style="font-size:12px;color:var(--gray);">Dnes</div>
    </div>
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;"><?= $week_recordings ?></div>
        <div style="font-size:12px;color:var(--gray);">Tento týždeň</div>
    </div>
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;"><?= $total_users ?></div>
        <div style="font-size:12px;color:var(--gray);">Používatelia</div>
    </div>
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;color:var(--green);"><?= $active_today ?></div>
        <div style="font-size:12px;color:var(--gray);">Aktívni dnes</div>
    </div>
    <div class="stat-card">
        <div style="font-size:28px;font-weight:900;"><?= $active_week ?></div>
        <div style="font-size:12px;color:var(--gray);">Aktívni tento týždeň</div>
    </div>
</div>

<!-- Coverage -->
<h3 style="margin:20px 0 8px;">Pokrytie posunkov</h3>
<div class="stat-cards-grid">
    <div class="stat-card">
        <div style="font-size:24px;font-weight:900;color:var(--green);"><?= $coverage_50 ?></div>
        <div style="font-size:12px;color:var(--gray);">≥50 nahrávok</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:900;color:var(--blue);"><?= $coverage_20 ?></div>
        <div style="font-size:12px;color:var(--gray);">20–49</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:900;color:#F59E0B;"><?= $coverage_10 ?></div>
        <div style="font-size:12px;color:var(--gray);">10–19</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:900;color:var(--red);"><?= $coverage_lt10 ?></div>
        <div style="font-size:12px;color:var(--gray);"><10</div>
    </div>
</div>

<!-- Per-theme chart -->
<h3 style="margin:20px 0 8px;">Nahrávky podľa témy</h3>
<?php foreach ($theme_stats as $ts): ?>
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="width:140px;font-size:13px;font-weight:600;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        <?= htmlspecialchars($ts['emoji'] . ' ' . $ts['name']) ?>
    </span>
    <div style="flex:1;height:18px;background:var(--light-gray);border-radius:9px;overflow:hidden;">
        <div style="height:100%;width:<?= $max_theme > 0 ? ($ts['total'] / $max_theme * 100) : 0 ?>%;background:var(--blue);border-radius:9px;"></div>
    </div>
    <span style="font-size:13px;color:var(--gray);width:40px;"><?= $ts['total'] ?></span>
</div>
<?php endforeach; ?>

<!-- Top contributors -->
<h3 style="margin:20px 0 8px;">Top prispievatelia</h3>
<table class="admin-table">
    <thead>
        <tr><th>#</th><th>Meno</th><th>Nahrávok</th></tr>
    </thead>
    <tbody>
        <?php foreach ($top_users as $i => $u): ?>
        <tr>
            <td><?= $i + 1 ?></td>
            <td><?= htmlspecialchars($u['display_name'] ?: $u['email']) ?></td>
            <td style="font-weight:700;"><?= $u['total_recordings'] ?></td>
        </tr>
        <?php endforeach; ?>
    </tbody>
</table>
