<?php
/**
 * Admin — Metrics tab: page views, visitors, referrers, server health
 * (included by admin/index.php, admin-only)
 *
 * Today's data: queried live from page_views.
 * Past data: read from daily_metrics (pre-aggregated).
 */

require_once __DIR__ . '/../includes/analytics.php';

$pdo = get_db();

// Handle aggregate action — skip already-aggregated days
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'aggregate') {
    require_csrf();
    $aggregated = 0;
    for ($i = 1; $i <= 30; $i++) {
        $d = date('Y-m-d', strtotime("-{$i} days"));
        $exists = $pdo->prepare("SELECT 1 FROM daily_metrics WHERE date = ?");
        $exists->execute([$d]);
        if (!$exists->fetchColumn()) {
            aggregate_daily_metrics($d);
            $aggregated++;
        }
    }
    header('Location: /admin/?tab=metrics&msg=aggregated&count=' . $aggregated);
    exit;
}

// Handle prune action
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'prune') {
    require_csrf();
    $pruned = prune_page_views(ANALYTICS_RETENTION_DAYS);
    header('Location: /admin/?tab=metrics&msg=pruned&count=' . $pruned);
    exit;
}

// --- Period selection ---
$period = $_GET['period'] ?? '7d';
$valid_periods = ['24h' => 1, '7d' => 7, '30d' => 30];
$days = $valid_periods[$period] ?? 7;
$period_label = ['24h'=>'24 hodín','7d'=>'7 dní','30d'=>'30 dní'][$period] ?? '7 dní';

// --- Today's live stats ---
$today_start = date('Y-m-d');
$today_end = date('Y-m-d', strtotime('+1 day'));

$today = $pdo->prepare("
    SELECT
        COUNT(*) as views,
        COUNT(DISTINCT ip_hash) as visitors,
        COUNT(DISTINCT session_id) as sessions,
        ROUND(AVG(response_time_ms)) as avg_ms
    FROM page_views
    WHERE created_at >= ? AND created_at < ?
");
$today->execute([$today_start, $today_end]);
$today = $today->fetch();

// --- Hourly breakdown today ---
$hourly_stmt = $pdo->prepare("
    SELECT HOUR(created_at) as hr, COUNT(*) as cnt
    FROM page_views
    WHERE created_at >= ? AND created_at < ?
    GROUP BY HOUR(created_at) ORDER BY hr
");
$hourly_stmt->execute([$today_start, $today_end]);
$hourly = $hourly_stmt->fetchAll(PDO::FETCH_KEY_PAIR);

// --- Period stats from daily_metrics ---
$period_stmt = $pdo->prepare("
    SELECT
        COALESCE(SUM(page_views), 0) as views,
        COALESCE(SUM(unique_visitors), 0) as visitors,
        COALESCE(SUM(unique_sessions), 0) as sessions,
        ROUND(AVG(avg_response_ms)) as avg_ms,
        MAX(avg_response_ms) as max_ms
    FROM daily_metrics
    WHERE date >= DATE_SUB(CURDATE(), INTERVAL ? DAY) AND date < CURDATE()
");
$period_stmt->execute([$days]);
$ps_past = $period_stmt->fetch();

$ps = [
    'views' => ($ps_past['views'] ?? 0) + ($today['views'] ?? 0),
    'visitors' => ($ps_past['visitors'] ?? 0) + ($today['visitors'] ?? 0),
    'sessions' => ($ps_past['sessions'] ?? 0) + ($today['sessions'] ?? 0),
    'avg_ms' => $ps_past['avg_ms'] ?? $today['avg_ms'],
    'max_ms' => max($ps_past['max_ms'] ?? 0, $today['avg_ms'] ?? 0),
];

// --- Daily trend ---
$trend_stmt = $pdo->prepare("
    SELECT date as day, page_views as views, unique_visitors as visitors
    FROM daily_metrics
    WHERE date >= DATE_SUB(CURDATE(), INTERVAL ? DAY)
    ORDER BY date
");
$trend_stmt->execute([$days]);
$trend = $trend_stmt->fetchAll(PDO::FETCH_ASSOC);
$trend[] = ['day' => date('Y-m-d'), 'views' => $today['views'], 'visitors' => $today['visitors']];

// --- Top pages ---
$top_pages_today = $pdo->prepare("
    SELECT page, COUNT(*) as cnt,
           COUNT(DISTINCT ip_hash) as unique_cnt,
           ROUND(AVG(response_time_ms)) as avg_ms
    FROM page_views
    WHERE created_at >= ? AND created_at < ?
    GROUP BY page ORDER BY cnt DESC LIMIT 10
");
$top_pages_today->execute([$today_start, $today_end]);
$pages = $top_pages_today->fetchAll(PDO::FETCH_ASSOC);
$max_page_cnt = max(array_column($pages, 'cnt') ?: [1]);

// --- Top referrers ---
$top_refs_today = $pdo->prepare("
    SELECT referrer, COUNT(*) as cnt, COUNT(DISTINCT ip_hash) as unique_cnt
    FROM page_views
    WHERE created_at >= ? AND created_at < ? AND referrer IS NOT NULL
    GROUP BY referrer ORDER BY cnt DESC LIMIT 10
");
$top_refs_today->execute([$today_start, $today_end]);
$referrers = $top_refs_today->fetchAll(PDO::FETCH_ASSOC);

// --- Device breakdown ---
$dev_today = $pdo->prepare("
    SELECT device_type, COUNT(*) as cnt
    FROM page_views WHERE created_at >= ? AND created_at < ? GROUP BY device_type
");
$dev_today->execute([$today_start, $today_end]);
$device_data = [];
foreach ($dev_today->fetchAll(PDO::FETCH_ASSOC) as $d) {
    $device_data[$d['device_type']] = (int) $d['cnt'];
}
$past_devs = $pdo->prepare("
    SELECT device_breakdown FROM daily_metrics
    WHERE date >= DATE_SUB(CURDATE(), INTERVAL ? DAY) AND date < CURDATE()
");
$past_devs->execute([$days]);
foreach ($past_devs->fetchAll(PDO::FETCH_COLUMN) as $json) {
    foreach (json_decode($json, true) ?: [] as $type => $cnt) {
        $device_data[$type] = ($device_data[$type] ?? 0) + (int) $cnt;
    }
}
$device_total = array_sum($device_data) ?: 1;

// --- Server ---
$disk = get_disk_usage();
$db_size_bytes = (int) ($pdo->query("
    SELECT SUM(data_length + index_length) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()
")->fetchColumn() ?? 0);

$pv_approx = $pdo->query("
    SELECT TABLE_ROWS FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'page_views'
")->fetchColumn();
$pv_count = $pv_approx !== false ? (int) $pv_approx : 0;

// --- Slow pages ---
$slow_pages = $pdo->prepare("
    SELECT page, ROUND(AVG(response_time_ms)) as avg_ms, MAX(response_time_ms) as max_ms, COUNT(*) as cnt
    FROM page_views
    WHERE created_at >= ? AND created_at < ? AND response_time_ms IS NOT NULL
    GROUP BY page HAVING cnt >= 3 ORDER BY avg_ms DESC LIMIT 5
");
$slow_pages->execute([$today_start, $today_end]);
$slow = $slow_pages->fetchAll(PDO::FETCH_ASSOC);

$msg = $_GET['msg'] ?? null;

// Page name labels for display
$page_labels = [
    'index' => 'Domov', 'record' => 'Nahrávanie', 'themes' => 'Témy',
    'validate' => 'Overovanie', 'progress' => 'Profil', 'thanks' => 'Ďakujeme',
    'consent' => 'Súhlas', 'terms' => 'Podmienky', 'admin/index' => 'Admin',
    'admin/admin' => 'Admin',
];
?>

<?php if ($msg === 'aggregated'): ?>
<div class="m-toast m-toast-ok">Agregácia hotová — <?= (int)($_GET['count'] ?? 0) ?> nových dní.</div>
<?php elseif ($msg === 'pruned'): ?>
<div class="m-toast m-toast-ok">Vymazaných <?= (int)($_GET['count'] ?? 0) ?> starších záznamov.</div>
<?php endif; ?>

<!-- ═══ HEADER: Period toggle ═══ -->
<div class="m-header">
    <div class="m-period-tabs">
        <?php foreach (['24h' => '24h', '7d' => '7 dní', '30d' => '30 dní'] as $k => $label): ?>
        <a href="/admin/?tab=metrics&period=<?= $k ?>"
           class="m-period-tab <?= $period === $k ? 'active' : '' ?>">
            <?= $label ?>
        </a>
        <?php endforeach; ?>
    </div>
</div>

<!-- ═══ SECTION 1: Key numbers (period) ═══ -->
<div class="m-section">
    <div class="m-kpi-row">
        <div class="m-kpi">
            <span class="m-kpi-value" style="color:var(--blue);"><?= number_format($ps['views']) ?></span>
            <span class="m-kpi-label">Zobrazení</span>
        </div>
        <div class="m-kpi">
            <span class="m-kpi-value" style="color:var(--green);"><?= number_format($ps['visitors']) ?></span>
            <span class="m-kpi-label">Návštevníkov</span>
        </div>
        <div class="m-kpi">
            <span class="m-kpi-value"><?= number_format($ps['sessions']) ?></span>
            <span class="m-kpi-label">Sedení</span>
        </div>
        <div class="m-kpi">
            <span class="m-kpi-value" style="color:<?= ($ps['avg_ms'] ?? 0) > 500 ? 'var(--red)' : 'var(--green)' ?>;"><?= ($ps['avg_ms'] ?? 0) > 500 ? '⚠ ' : '✓ ' ?><?= $ps['avg_ms'] ?? '—' ?><small>ms</small></span>
            <span class="m-kpi-label">Odozva</span>
        </div>
    </div>

    <!-- Device bar (inline, compact) -->
    <?php
    $dt_desktop = $device_data['desktop'] ?? 0;
    $dt_mobile = $device_data['mobile'] ?? 0;
    $dt_tablet = $device_data['tablet'] ?? 0;
    $pct_d = round($dt_desktop / $device_total * 100);
    $pct_m = round($dt_mobile / $device_total * 100);
    $pct_t = round($dt_tablet / $device_total * 100);
    ?>
    <div class="m-device-bar-wrap">
        <div class="m-device-bar">
            <?php if ($pct_d > 0): ?><div style="width:<?= $pct_d ?>%;background:var(--blue);" title="Desktop <?= $pct_d ?>%"></div><?php endif; ?>
            <?php if ($pct_m > 0): ?><div style="width:<?= $pct_m ?>%;background:var(--green);" title="Mobil <?= $pct_m ?>%"></div><?php endif; ?>
            <?php if ($pct_t > 0): ?><div style="width:<?= $pct_t ?>%;background:#F59E0B;" title="Tablet <?= $pct_t ?>%"></div><?php endif; ?>
        </div>
        <div class="m-device-legend">
            <span><i style="background:var(--blue);"></i> Desktop <?= $pct_d ?>%</span>
            <span><i style="background:var(--green);"></i> Mobil <?= $pct_m ?>%</span>
            <?php if ($pct_t > 0): ?><span><i style="background:#F59E0B;"></i> Tablet <?= $pct_t ?>%</span><?php endif; ?>
        </div>
    </div>
</div>

<!-- ═══ SECTION 2: Charts ═══ -->
<div class="m-section">
    <!-- Daily trend -->
    <?php if (count($trend) > 1): ?>
    <div class="m-chart-label">Trend za <?= $period_label ?></div>
    <?php $max_views = max(array_column($trend, 'views') ?: [1]); ?>
    <div class="m-chart" style="height:120px;">
        <?php foreach ($trend as $i => $t):
            $h = $max_views > 0 ? max(4, $t['views'] / $max_views * 100) : 4;
            $is_today = ($t['day'] === date('Y-m-d'));
        ?>
        <div class="m-bar-col" title="<?= $t['day'] ?>&#10;<?= $t['views'] ?> zobrazení&#10;<?= $t['visitors'] ?> návštevníkov">
            <span class="m-bar-val"><?= $t['views'] ?: '' ?></span>
            <div class="m-bar <?= $is_today ? 'm-bar-today' : '' ?>" style="height:<?= $h ?>%;"></div>
            <span class="m-bar-date"><?= date('d.m', strtotime($t['day'])) ?></span>
        </div>
        <?php endforeach; ?>
    </div>
    <?php endif; ?>

    <!-- Hourly sparkline (today) -->
    <div class="m-chart-label" style="margin-top:16px;">Dnes po hodinách</div>
    <?php $max_hr = max($hourly ?: [1]); ?>
    <div class="m-sparkline">
        <?php for ($h = 0; $h < 24; $h++):
            $cnt = $hourly[$h] ?? 0;
            $pct = $max_hr > 0 ? max(3, $cnt / $max_hr * 100) : 3;
            $is_now = ($h === (int) date('G'));
        ?>
        <div class="m-spark-bar <?= $is_now ? 'm-spark-now' : '' ?> <?= $cnt === 0 ? 'm-spark-empty' : '' ?>"
             style="height:<?= $pct ?>%;"
             title="<?= sprintf('%02d:00 — %d', $h, $cnt) ?>">
        </div>
        <?php endfor; ?>
    </div>
    <div class="m-spark-labels">
        <span>0h</span><span>6h</span><span>12h</span><span>18h</span><span>23h</span>
    </div>
</div>

<!-- ═══ SECTION 3: Top pages + Referrers ═══ -->
<div class="m-section">
    <div class="m-chart-label">Najpopulárnejšie stránky dnes</div>
    <?php if (empty($pages)): ?>
    <div class="m-empty">Zatiaľ žiadne návštevy</div>
    <?php else: ?>
    <div class="m-page-list">
        <?php foreach ($pages as $p):
            $pname = $page_labels[$p['page']] ?? $p['page'];
            $bar_w = $max_page_cnt > 0 ? max(2, $p['cnt'] / $max_page_cnt * 100) : 2;
        ?>
        <div class="m-page-row">
            <span class="m-page-name"><?= htmlspecialchars($pname) ?></span>
            <div class="m-page-bar-wrap">
                <div class="m-page-bar" style="width:<?= $bar_w ?>%;"></div>
            </div>
            <span class="m-page-count"><?= number_format($p['cnt']) ?></span>
            <span class="m-page-uniq"><?= number_format($p['unique_cnt']) ?> uník.</span>
        </div>
        <?php endforeach; ?>
    </div>
    <?php endif; ?>

    <?php if (!empty($referrers)): ?>
    <div class="m-chart-label" style="margin-top:16px;">Odkiaľ prichádzajú</div>
    <div class="m-ref-list">
        <?php foreach ($referrers as $r): ?>
        <div class="m-ref-row">
            <span class="m-ref-name" title="<?= htmlspecialchars($r['referrer']) ?>"><?= htmlspecialchars($r['referrer']) ?></span>
            <span class="m-ref-count"><?= number_format($r['cnt']) ?></span>
        </div>
        <?php endforeach; ?>
    </div>
    <?php endif; ?>
</div>

<!-- ═══ SECTION 4: Server + Performance ═══ -->
<div class="m-section">
    <div class="m-chart-label">Server a výkon</div>
    <div class="m-server-grid">
        <!-- Storage gauge -->
        <div class="m-gauge">
            <svg viewBox="0 0 120 70" class="m-gauge-svg" aria-hidden="true">
                <path d="M15,60 A45,45 0 0,1 105,60" fill="none" stroke="var(--light-gray)" stroke-width="10" stroke-linecap="round"/>
                <path d="M15,60 A45,45 0 0,1 105,60" fill="none"
                      stroke="<?= $disk['used_pct'] > 90 ? 'var(--red)' : ($disk['used_pct'] > 80 ? '#F59E0B' : 'var(--green)') ?>"
                      stroke-width="10" stroke-linecap="round"
                      stroke-dasharray="<?= round(141.37 * $disk['used_pct'] / 100) ?> 141.37"/>
            </svg>
            <div class="m-gauge-text">
                <strong><?= $disk['used_pct'] ?>%</strong>
                <span><?= number_format($disk['used'] / (1024*1024), 0) ?> / <?= number_format(STORAGE_LIMIT_GB * 1024) ?> MB</span>
            </div>
            <div class="m-gauge-title">Úložisko</div>
        </div>

        <!-- Other server stats -->
        <div class="m-server-stats">
            <div class="m-server-row">
                <span>Databáza</span>
                <strong><?= number_format($db_size_bytes / (1024*1024), 1) ?> MB</strong>
            </div>
            <div class="m-server-row">
                <span>PHP</span>
                <strong><?= PHP_VERSION ?></strong>
            </div>
            <div class="m-server-row">
                <span>Analytika</span>
                <strong>~<?= number_format($pv_count) ?> záznamov</strong>
            </div>
            <?php if ($ps['avg_ms']): ?>
            <div class="m-server-row">
                <span>Priem. odozva</span>
                <strong style="color:<?= $ps['avg_ms'] > 500 ? 'var(--red)' : 'var(--green)' ?>;"><?= $ps['avg_ms'] > 500 ? '⚠ ' : '✓ ' ?><?= $ps['avg_ms'] ?> ms</strong>
            </div>
            <?php endif; ?>
        </div>
    </div>

    <!-- Slow pages (only if data exists) -->
    <?php if (!empty($slow)): ?>
    <div class="m-chart-label" style="margin-top:16px;">Pomalé stránky dnes</div>
    <?php foreach ($slow as $s):
        $pname = $page_labels[$s['page']] ?? $s['page'];
    ?>
    <div class="m-slow-row">
        <span class="m-slow-name"><?= htmlspecialchars($pname) ?></span>
        <span class="m-slow-bar">
            <span style="width:<?= min(100, ($s['avg_ms'] ?? 0) / 5) ?>%;background:<?= $s['avg_ms'] > 500 ? 'var(--red)' : ($s['avg_ms'] > 200 ? '#F59E0B' : 'var(--green)') ?>;"></span>
        </span>
        <span class="m-slow-val"><?= $s['avg_ms'] ?> ms</span>
    </div>
    <?php endforeach; ?>
    <?php endif; ?>
</div>

<!-- ═══ SECTION 5: Data management (collapsed) ═══ -->
<details class="m-section m-details">
    <summary class="m-chart-label" style="cursor:pointer;">Správa dát</summary>
    <div style="padding-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <form method="POST" style="display:inline;">
            <?= csrf_field() ?>
            <input type="hidden" name="action" value="aggregate">
            <button type="submit" class="m-btn m-btn-blue">Agregovať chýbajúce dni</button>
        </form>
        <form method="POST" style="display:inline;" onsubmit="return confirm('Vymazať záznamy staršie ako <?= ANALYTICS_RETENTION_DAYS ?> dni?')">
            <?= csrf_field() ?>
            <input type="hidden" name="action" value="prune">
            <button type="submit" class="m-btn m-btn-red">Vymazať staré (<?= ANALYTICS_RETENTION_DAYS ?>+ dni)</button>
        </form>
    </div>
</details>

<!-- ═══ SCOPED STYLES ═══ -->
<style>
/* ── Layout ── */
.m-section {
    background: var(--card-bg);
    border: 1px solid var(--light-gray);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.m-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 8px;
}
.m-toast { padding: 10px 16px; border-radius: 10px; font-size: 14px; font-weight: 600; margin-bottom: 12px; }
.m-toast-ok { background: #DCFCE7; color: #065F46; }
.m-empty { text-align: center; color: var(--gray); padding: 20px; font-size: 14px; }

/* ── Period tabs ── */
.m-period-tabs { display: flex; gap: 4px; background: var(--light-gray); border-radius: 10px; padding: 3px; }
.m-period-tab {
    padding: 8px 18px; border-radius: 8px; font-size: 14px; font-weight: 700;
    text-decoration: none; color: var(--gray); transition: all 0.15s;
}
.m-period-tab.active { background: var(--blue); color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
.m-period-tab:not(.active):hover { background: rgba(0,0,0,0.05); }

/* ── KPI row ── */
.m-kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.m-kpi { text-align: center; }
.m-kpi-value { display: block; font-size: 32px; font-weight: 900; line-height: 1.1; }
.m-kpi-value small { font-size: 16px; font-weight: 600; }
.m-kpi-label { display: block; font-size: 12px; color: var(--gray); margin-top: 2px; }

/* ── Device bar ── */
.m-device-bar-wrap { margin-top: 4px; }
.m-device-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; }
.m-device-bar > div { border-radius: 4px; transition: width 0.3s; }
.m-device-legend { display: flex; gap: 14px; margin-top: 6px; font-size: 12px; color: var(--gray); }
.m-device-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }

/* ── Charts ── */
.m-chart-label { font-size: 13px; font-weight: 700; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
.m-chart { display: flex; align-items: flex-end; gap: 3px; padding-bottom: 20px; position: relative; }
.m-bar-col { flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 0; }
.m-bar { background: var(--blue); border-radius: 4px 4px 0 0; width: 100%; min-height: 4px; opacity: 0.8; transition: opacity 0.15s; }
.m-bar:hover { opacity: 1; }
.m-bar-today { background: var(--green); opacity: 1; }
.m-bar-val { font-size: 9px; color: var(--gray); margin-bottom: 2px; min-height: 12px; }
.m-bar-date { font-size: 9px; color: var(--gray); margin-top: 4px; white-space: nowrap; }

/* ── Sparkline (hourly) ── */
.m-sparkline { display: flex; align-items: flex-end; gap: 2px; height: 50px; }
.m-spark-bar { flex: 1; background: var(--blue); border-radius: 2px 2px 0 0; min-height: 2px; opacity: 0.7; transition: opacity 0.15s; }
.m-spark-bar:hover { opacity: 1; }
.m-spark-now { background: var(--green); opacity: 1; }
.m-spark-empty { opacity: 0.1; }
.m-spark-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--gray); padding: 2px 0 0; }

/* ── Page list ── */
.m-page-list { display: flex; flex-direction: column; gap: 6px; }
.m-page-row { display: grid; grid-template-columns: 100px 1fr 50px 60px; align-items: center; gap: 8px; font-size: 13px; }
.m-page-name { font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.m-page-bar-wrap { height: 6px; background: var(--light-gray); border-radius: 3px; overflow: hidden; }
.m-page-bar { height: 100%; background: var(--blue); border-radius: 3px; min-width: 2px; }
.m-page-count { text-align: right; font-weight: 700; }
.m-page-uniq { text-align: right; color: var(--gray); font-size: 11px; }

/* ── Referrer list ── */
.m-ref-list { display: flex; flex-direction: column; gap: 4px; }
.m-ref-row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; border-bottom: 1px solid var(--light-gray); }
.m-ref-row:last-child { border-bottom: none; }
.m-ref-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 80%; }
.m-ref-count { font-weight: 700; flex-shrink: 0; }

/* ── Server + gauge ── */
.m-server-grid { display: grid; grid-template-columns: 160px 1fr; gap: 20px; align-items: center; }
.m-gauge { text-align: center; }
.m-gauge-svg { width: 120px; height: 70px; }
.m-gauge-text { margin-top: -8px; }
.m-gauge-text strong { display: block; font-size: 24px; font-weight: 900; }
.m-gauge-text span { font-size: 11px; color: var(--gray); }
.m-gauge-title { font-size: 12px; color: var(--gray); margin-top: 2px; }
.m-server-stats { display: flex; flex-direction: column; gap: 8px; }
.m-server-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--light-gray); font-size: 14px; }
.m-server-row:last-child { border-bottom: none; }
.m-server-row span { color: var(--gray); }

/* ── Slow pages ── */
.m-slow-row { display: grid; grid-template-columns: 100px 1fr 60px; align-items: center; gap: 8px; padding: 4px 0; font-size: 13px; }
.m-slow-name { font-weight: 600; }
.m-slow-bar { height: 6px; background: var(--light-gray); border-radius: 3px; overflow: hidden; }
.m-slow-bar > span { display: block; height: 100%; border-radius: 3px; }
.m-slow-val { text-align: right; font-weight: 700; font-size: 12px; }

/* ── Buttons ── */
.m-btn { font-size: 13px; padding: 8px 16px; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; }
.m-btn-blue { background: var(--blue); color: white; }
.m-btn-red { background: var(--red); color: white; }

/* ── Details toggle ── */
.m-details summary { list-style: none; }
.m-details summary::-webkit-details-marker { display: none; }
.m-details summary::after { content: ' +'; }
.m-details[open] summary::after { content: ' −'; }

/* ── Responsive ── */
@media (max-width: 600px) {
    .m-kpi-row { grid-template-columns: repeat(2, 1fr); }
    .m-kpi-value { font-size: 24px; }
    .m-server-grid { grid-template-columns: 1fr; }
    .m-gauge { margin-bottom: 8px; }
    .m-page-row { grid-template-columns: 80px 1fr 40px; }
    .m-page-uniq { display: none; }
    .m-bar-date { font-size: 8px; }
    .m-chart { height: 90px; }
}

/* ── Dark mode ── */
html.dark .m-section { background: #1E293B; border-color: #334155; }
html.dark .m-period-tabs { background: #334155; }
html.dark .m-period-tab { color: #94A3B8; }
html.dark .m-period-tab:not(.active):hover { background: rgba(255,255,255,0.05); }
html.dark .m-page-bar-wrap, html.dark .m-slow-bar, html.dark .m-device-bar { background: #334155; }
html.dark .m-ref-row, html.dark .m-server-row { border-color: #334155; }
html.dark .m-toast-ok { background: #064E3B; color: #6EE7B7; }
</style>
