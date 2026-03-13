<?php
/**
 * SPJ Collector — Record Screen (main screen)
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
require_login();

$user = get_user();
$pdo = get_db();

// Count today's recordings
$stmt = $pdo->prepare('SELECT COUNT(*) FROM recordings WHERE user_id = ? AND DATE(created_at) = CURDATE()');
$stmt->execute([$user['id']]);
$today_count = (int) $stmt->fetchColumn();

// Theme filtering
$theme_id = (int) ($_GET['theme_id'] ?? 0);
$sign_id = (int) ($_GET['sign_id'] ?? 0);
$theme = null;
if ($theme_id > 0) {
    $stmt = $pdo->prepare('SELECT id, name, emoji FROM themes WHERE id = ?');
    $stmt->execute([$theme_id]);
    $theme = $stmt->fetch();
}

$page_title = 'Nahrať znak — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<?php if ($theme): ?>
<a href="/themes.php" style="display: inline-block; margin-bottom: 12px; color: var(--blue); text-decoration: none; font-weight: 600; font-size: 15px;">
    ← <?= htmlspecialchars($theme['emoji'] . ' ' . $theme['name']) ?>
</a>
<?php elseif (isset($_GET['theme_id']) && $theme_id === 0): ?>
<a href="/themes.php" style="display: inline-block; margin-bottom: 12px; color: var(--blue); text-decoration: none; font-weight: 600; font-size: 15px;">
    ← 📋 Bez témy
</a>
<?php endif; ?>

<input type="hidden" id="current-theme-id" value="<?= $theme_id ?>">
<input type="hidden" id="initial-sign-id" value="<?= $sign_id ?>">
<input type="hidden" id="theme-id-set" value="<?= isset($_GET['theme_id']) ? '1' : '0' ?>">

<div class="record-layout">
    <!-- Camera column -->
    <div class="col-camera">
        <!-- Progress bar -->
        <div class="progress-info">
            <span class="label">Dnes</span>
            <span class="count" id="progress-count"><?= $today_count ?> / <?= DAILY_GOAL ?></span>
        </div>
        <div class="progress-bar" style="margin-bottom: 16px;">
            <div class="fill" id="progress-fill"
                 style="width: <?= min(100, ($today_count / DAILY_GOAL) * 100) ?>%;"></div>
        </div>

        <!-- Camera preview (shown during recording) -->
        <div class="camera-container border-green" id="camera-container" style="display: none;">
            <video id="camera-preview" autoplay playsinline muted></video>
            <div class="camera-banner ok" id="camera-banner">✅ Pripravené na nahrávanie</div>
            <div class="countdown-overlay" id="countdown-overlay" style="display: none;">
                <div class="countdown-num" id="countdown-num">3</div>
            </div>
            <div class="recording-indicator" id="recording-indicator" style="display: none;">
                <div class="rec-dot"></div>
                <span id="recording-timer">0:00 / 0:03</span>
            </div>
        </div>

        <!-- Video preview (shown after recording) -->
        <video class="video-preview" id="video-preview" style="display: none;"
               controls autoplay playsinline loop></video>

        <!-- Camera loading -->
        <div class="camera-container" id="camera-loading">
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: white;">
                <p style="font-size: 16px;">📷 Spúšťam kameru...</p>
            </div>
        </div>

        <!-- Status badges -->
        <div class="status-row" id="status-badges">
            <span class="status-badge status-ok" id="badge-hands">✅ Ruky</span>
            <span class="status-badge status-ok" id="badge-face">✅ Tvár</span>
            <span class="status-badge status-ok" id="badge-light">✅ Svetlo</span>
        </div>
    </div>

    <!-- Controls column -->
    <div class="col-controls">
        <!-- Word display -->
        <div id="word-section">
            <div class="word-display" id="word-display">...</div>
            <div class="help-link" id="help-links">
                Nepoznáte znak?
            </div>
        </div>

        <!-- Record button (before recording) -->
        <div class="record-btn-wrap" id="record-controls">
            <div class="record-btn" id="record-btn" onclick="startRecording()">
                <div class="inner"></div>
            </div>
            <p class="record-btn-label" id="record-label">Stlačte pre nahrávanie</p>
        </div>

        <!-- Submit/re-record (after recording) -->
        <div id="preview-controls" style="display: none;">
            <p style="text-align: center; font-size: 15px; color: var(--gray); margin-bottom: 12px;">
                Skontrolujte vaše video
            </p>
            <button class="btn btn-green" onclick="submitRecording()" id="submit-btn"
                    style="margin-bottom: 12px; font-size: 18px;">✓ Odoslať</button>
            <button class="btn btn-gray" onclick="retryRecording()">↻ Nahrať znova</button>
        </div>

        <!-- Total counter (desktop) -->
        <div class="card desktop-only" style="text-align: center; margin-top: auto;">
            <p style="font-size: 13px; color: var(--gray);">Celkom nahrané</p>
            <p style="font-size: 28px; font-weight: 900;"><?= number_format($user['total_recordings']) ?></p>
        </div>
    </div>
</div>

<!-- Upload toast -->
<div class="toast success" id="toast" style="display: none;"></div>

<script src="/js/qualityCheck.js"></script>
<script src="/js/offlineStore.js"></script>
<script src="/js/recorder.js"></script>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
