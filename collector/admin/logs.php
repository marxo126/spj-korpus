<?php
/**
 * Admin — Error logs tab (included by admin/index.php)
 * Shows PHP + JS errors captured by error_logger.
 */

$pdo = get_db();

// Filters
$filter_level = $_GET['log_level'] ?? 'all';
$filter_source = $_GET['log_source'] ?? 'all';
$page = max(1, (int) ($_GET['log_page'] ?? 1));
$per_page = 50;
$offset = ($page - 1) * $per_page;

$where = '1=1';
$params = [];
if ($filter_level !== 'all') {
    $where .= ' AND level = ?';
    $params[] = $filter_level;
}
if ($filter_source !== 'all') {
    $where .= ' AND source = ?';
    $params[] = $filter_source;
}

$count_stmt = $pdo->prepare("SELECT COUNT(*) FROM error_log WHERE $where");
$count_stmt->execute($params);
$total = (int) $count_stmt->fetchColumn();
$total_pages = max(1, ceil($total / $per_page));

$params[] = $per_page;
$params[] = $offset;
$stmt = $pdo->prepare("
    SELECT e.*, u.email as user_email
    FROM error_log e
    LEFT JOIN users u ON e.user_id = u.id
    WHERE $where
    ORDER BY e.created_at DESC
    LIMIT ? OFFSET ?
");
$stmt->execute($params);
$logs = $stmt->fetchAll();

// Stats
$stats = $pdo->query("
    SELECT
        COUNT(*) as total,
        SUM(level = 'error') as errors,
        SUM(level = 'warning') as warnings,
        SUM(created_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)) as last_hour,
        SUM(created_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)) as last_24h
    FROM error_log
")->fetch();

$level_icons = ['error' => '🔴', 'warning' => '🟡', 'info' => '🔵'];
$source_icons = ['php' => '🐘', 'js' => '⚡', 'api' => '🔗'];
?>

<!-- Stats -->
<div class="stat-cards-grid" style="margin-bottom:16px;">
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:#DC2626;"><?= $stats['errors'] ?? 0 ?></div>
        <div style="font-size:13px;color:var(--gray);">Chyby</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:#F59E0B;"><?= $stats['warnings'] ?? 0 ?></div>
        <div style="font-size:13px;color:var(--gray);">Varovania</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:var(--blue);"><?= $stats['last_hour'] ?? 0 ?></div>
        <div style="font-size:13px;color:var(--gray);">Za hodinu</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:var(--green);"><?= $stats['last_24h'] ?? 0 ?></div>
        <div style="font-size:13px;color:var(--gray);">Za 24h</div>
    </div>
</div>

<!-- Filters -->
<div class="admin-filter" style="margin-bottom:16px;">
    <select onchange="location.href='/admin/?tab=logs&log_level='+this.value+'&log_source=<?= $filter_source ?>'">
        <option value="all" <?= $filter_level === 'all' ? 'selected' : '' ?>>Všetky úrovne</option>
        <option value="error" <?= $filter_level === 'error' ? 'selected' : '' ?>>🔴 Chyby</option>
        <option value="warning" <?= $filter_level === 'warning' ? 'selected' : '' ?>>🟡 Varovania</option>
        <option value="info" <?= $filter_level === 'info' ? 'selected' : '' ?>>🔵 Info</option>
    </select>
    <select onchange="location.href='/admin/?tab=logs&log_level=<?= $filter_level ?>&log_source='+this.value">
        <option value="all" <?= $filter_source === 'all' ? 'selected' : '' ?>>Všetky zdroje</option>
        <option value="php" <?= $filter_source === 'php' ? 'selected' : '' ?>>🐘 PHP</option>
        <option value="js" <?= $filter_source === 'js' ? 'selected' : '' ?>>⚡ JavaScript</option>
        <option value="api" <?= $filter_source === 'api' ? 'selected' : '' ?>>🔗 API</option>
    </select>
    <span style="font-size:13px;color:var(--gray);"><?= $total ?> záznamov</span>
    <?php if ($total > 0): ?>
    <form method="POST" action="/admin/api/logs.php" style="display:inline;margin-left:auto;"
          onsubmit="return confirm('Vymazať všetky logy?')">
        <?= csrf_field() ?>
        <input type="hidden" name="action" value="clear">
        <button type="submit" style="background:#DC2626;color:white;border:none;padding:6px 14px;border-radius:6px;font-size:13px;cursor:pointer;font-weight:600;">
            Vymazať logy
        </button>
    </form>
    <?php endif; ?>
</div>

<!-- Log entries -->
<?php if (empty($logs)): ?>
<div style="text-align:center;color:var(--gray);margin:40px 0;">
    <div style="font-size:48px;margin-bottom:8px;">✅</div>
    <p>Žiadne chyby. Všetko funguje.</p>
</div>
<?php else: ?>
<div style="display:flex;flex-direction:column;gap:8px;">
<?php foreach ($logs as $log):
    $extra = $log['extra'] ? json_decode($log['extra'], true) : null;
?>
<div class="card" style="padding:12px 16px;margin-bottom:0;">
    <div style="display:flex;align-items:flex-start;gap:8px;">
        <span style="font-size:16px;"><?= $level_icons[$log['level']] ?? '⚪' ?><?= $source_icons[$log['source']] ?? '' ?></span>
        <div style="flex:1;min-width:0;">
            <div style="font-size:14px;font-weight:600;word-break:break-word;"><?= htmlspecialchars(mb_substr($log['message'], 0, 300)) ?></div>
            <div style="font-size:12px;color:var(--gray);margin-top:4px;">
                <?= date('d.m.Y H:i:s', strtotime($log['created_at'])) ?>
                <?php if ($log['url']): ?>
                 · <code style="font-size:11px;"><?= htmlspecialchars(mb_substr($log['url'], 0, 80)) ?></code>
                <?php endif; ?>
                <?php if ($log['user_email']): ?>
                 · <?= htmlspecialchars($log['user_email']) ?>
                <?php endif; ?>
            </div>
            <?php if ($extra): ?>
            <details style="margin-top:6px;">
                <summary style="font-size:12px;color:var(--blue);cursor:pointer;">Detaily</summary>
                <pre style="font-size:11px;color:var(--gray);margin-top:4px;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow:auto;"><?= htmlspecialchars(json_encode($extra, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE)) ?></pre>
            </details>
            <?php endif; ?>
        </div>
    </div>
</div>
<?php endforeach; ?>
</div>

<!-- Pagination -->
<?php if ($total_pages > 1): ?>
<div style="display:flex;justify-content:center;gap:8px;margin-top:16px;">
    <?php for ($p = 1; $p <= min($total_pages, 20); $p++): ?>
    <a href="/admin/?tab=logs&log_level=<?= $filter_level ?>&log_source=<?= $filter_source ?>&log_page=<?= $p ?>"
       style="padding:6px 12px;border-radius:6px;font-size:14px;text-decoration:none;<?= $p == $page ? 'background:var(--blue);color:white;' : 'background:var(--light-gray);color:var(--dark);' ?>">
        <?= $p ?>
    </a>
    <?php endfor; ?>
</div>
<?php endif; ?>
<?php endif; ?>
