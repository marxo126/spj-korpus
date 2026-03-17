<?php
/**
 * SPJ Collector — Lightweight page view tracking
 *
 * Call track_page_view() from header.php to log each visit.
 * No raw IPs stored (GDPR-safe: SHA-256 hash only).
 * Skips bots, static assets, and API calls.
 */

require_once __DIR__ . '/db.php';

/** Start timing at script start — call from top of header.php */
function analytics_start_timer(): void {
    if (!defined('ANALYTICS_START')) {
        define('ANALYTICS_START', hrtime(true));
    }
}

/** Detect device type from User-Agent */
function detect_device_type(string $ua): string {
    $ua = strtolower($ua);
    if (preg_match('/tablet|ipad|playbook|silk/', $ua)) return 'tablet';
    if (preg_match('/mobile|android|iphone|ipod|phone|blackberry|opera mini|iemobile/', $ua)) return 'mobile';
    return 'desktop';
}

/** Check if request is from a known bot */
function is_bot(string $ua): bool {
    $ua = strtolower($ua);
    return (bool) preg_match('/bot|crawl|spider|slurp|mediapartners|lighthouse|pagespeed|gtmetrix|pingdom|uptimerobot/', $ua);
}

/** Get disk usage based on hosting quota — shared by status.php and metrics.php */
function get_disk_usage(): array {
    $upload_dir = UPLOAD_DIR;
    $used = 0;
    if (is_dir($upload_dir)) {
        $output = shell_exec("du -sb " . escapeshellarg($upload_dir) . " 2>/dev/null");
        if ($output) $used = (int) explode("\t", $output)[0];
    }
    $limit = STORAGE_LIMIT_GB * 1024 * 1024 * 1024;
    $free = max(0, $limit - $used);
    return [
        'used' => $used,
        'total' => $limit,
        'free' => $free,
        'used_pct' => $limit > 0 ? (int) round($used / $limit * 100) : 0,
    ];
}

/**
 * Track a page view. Safe to call on every request —
 * silently skips bots, API endpoints, and static assets.
 */
function track_page_view(): void {
    try {
        $uri = $_SERVER['REQUEST_URI'] ?? '';
        $ua = $_SERVER['HTTP_USER_AGENT'] ?? '';

        // Skip: bots, API calls, static assets, admin API
        if (is_bot($ua)) return;
        if (str_starts_with($uri, '/api/')) return;
        if (str_starts_with($uri, '/admin/api/')) return;
        if (preg_match('/\.(css|js|png|jpg|jpeg|gif|svg|ico|woff2?|map|webp)(\?|$)/', $uri)) return;

        $pdo = get_db();

        // Extract page name from URL
        $path = parse_url($uri, PHP_URL_PATH) ?: '/';
        $page = basename($path, '.php');
        if ($page === '' || $page === '/') $page = 'index';
        if (str_starts_with($path, '/admin/')) $page = 'admin/' . $page;

        // GDPR-safe: hash IP with daily salt (prevents long-term tracking)
        $ip = $_SERVER['REMOTE_ADDR'] ?? '0.0.0.0';
        $daily_salt = date('Y-m-d') . '|' . ANALYTICS_SALT;
        $ip_hash = hash('sha256', $ip . '|' . $daily_salt);

        // Session-based unique visitor
        $session_id = null;
        if (session_status() === PHP_SESSION_ACTIVE && !empty(session_id())) {
            $session_id = hash('sha256', session_id() . '|' . $daily_salt);
        }

        // Response time
        $response_ms = null;
        if (defined('ANALYTICS_START')) {
            $response_ms = (int) ((hrtime(true) - ANALYTICS_START) / 1_000_000);
        }

        // Referrer (strip query params for privacy)
        $referrer = $_SERVER['HTTP_REFERER'] ?? null;
        if ($referrer) {
            $parsed = parse_url($referrer);
            $ref_host = $parsed['host'] ?? '';
            // Skip self-referrals
            if (str_contains($ref_host, 'zber.spj.sk') || str_contains($ref_host, 'localhost')) {
                $referrer = null;
            } else {
                $referrer = mb_substr($ref_host . ($parsed['path'] ?? ''), 0, 500);
            }
        }

        $stmt = $pdo->prepare('
            INSERT INTO page_views (url, page, user_id, session_id, ip_hash, user_agent, referrer, device_type, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ');
        $stmt->execute([
            mb_substr($uri, 0, 500),
            mb_substr($page, 0, 100),
            $_SESSION['user_id'] ?? null,
            $session_id,
            $ip_hash,
            mb_substr($ua, 0, 500),
            $referrer,
            detect_device_type($ua),
            $response_ms,
        ]);
    } catch (\Throwable $e) {
        // Never break the page for analytics failure
        error_log("SPJ analytics error: " . $e->getMessage());
    }
}

/**
 * Aggregate daily metrics from page_views into daily_metrics table.
 * Uses range predicates (not DATE()) to allow index usage.
 */
function aggregate_daily_metrics(?string $date = null): void {
    $date = $date ?? date('Y-m-d', strtotime('-1 day'));
    $next_date = date('Y-m-d', strtotime($date . ' +1 day'));
    $pdo = get_db();

    // Combined stats + device breakdown in one query
    $stats = $pdo->prepare("
        SELECT
            COUNT(*) as page_views,
            COUNT(DISTINCT ip_hash) as unique_visitors,
            COUNT(DISTINCT session_id) as unique_sessions,
            ROUND(AVG(response_time_ms)) as avg_response_ms,
            SUM(device_type = 'desktop') as dt_desktop,
            SUM(device_type = 'mobile') as dt_mobile,
            SUM(device_type = 'tablet') as dt_tablet
        FROM page_views
        WHERE created_at >= ? AND created_at < ?
    ");
    $stats->execute([$date, $next_date]);
    $row = $stats->fetch();

    $device_breakdown = [
        'desktop' => (int) ($row['dt_desktop'] ?? 0),
        'mobile' => (int) ($row['dt_mobile'] ?? 0),
        'tablet' => (int) ($row['dt_tablet'] ?? 0),
    ];

    // Recordings that day
    $rec = $pdo->prepare("SELECT COUNT(*) FROM recordings WHERE created_at >= ? AND created_at < ?");
    $rec->execute([$date, $next_date]);
    $recordings = (int) $rec->fetchColumn();

    // New users
    $users = $pdo->prepare("SELECT COUNT(*) FROM users WHERE created_at >= ? AND created_at < ?");
    $users->execute([$date, $next_date]);
    $new_users = (int) $users->fetchColumn();

    // Top pages
    $pages = $pdo->prepare("
        SELECT page, COUNT(*) as cnt FROM page_views
        WHERE created_at >= ? AND created_at < ? GROUP BY page ORDER BY cnt DESC LIMIT 10
    ");
    $pages->execute([$date, $next_date]);
    $top_pages = $pages->fetchAll(PDO::FETCH_ASSOC);

    // Top referrers
    $refs = $pdo->prepare("
        SELECT referrer, COUNT(*) as cnt FROM page_views
        WHERE created_at >= ? AND created_at < ? AND referrer IS NOT NULL
        GROUP BY referrer ORDER BY cnt DESC LIMIT 10
    ");
    $refs->execute([$date, $next_date]);
    $top_referrers = $refs->fetchAll(PDO::FETCH_ASSOC);

    $stmt = $pdo->prepare("
        INSERT INTO daily_metrics (date, page_views, unique_visitors, unique_sessions, recordings_count, new_users, avg_response_ms, top_pages, top_referrers, device_breakdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            page_views = VALUES(page_views),
            unique_visitors = VALUES(unique_visitors),
            unique_sessions = VALUES(unique_sessions),
            recordings_count = VALUES(recordings_count),
            new_users = VALUES(new_users),
            avg_response_ms = VALUES(avg_response_ms),
            top_pages = VALUES(top_pages),
            top_referrers = VALUES(top_referrers),
            device_breakdown = VALUES(device_breakdown)
    ");
    $stmt->execute([
        $date,
        $row['page_views'],
        $row['unique_visitors'],
        $row['unique_sessions'],
        $recordings,
        $new_users,
        $row['avg_response_ms'],
        json_encode($top_pages, JSON_UNESCAPED_UNICODE),
        json_encode($top_referrers, JSON_UNESCAPED_UNICODE),
        json_encode($device_breakdown, JSON_UNESCAPED_UNICODE),
    ]);
}

/**
 * Prune old page_views rows (keep raw data for N days, aggregated forever).
 */
function prune_page_views(int $keep_days = ANALYTICS_RETENTION_DAYS): int {
    $pdo = get_db();
    $stmt = $pdo->prepare("DELETE FROM page_views WHERE created_at < DATE_SUB(NOW(), INTERVAL ? DAY)");
    $stmt->execute([$keep_days]);
    return $stmt->rowCount();
}
