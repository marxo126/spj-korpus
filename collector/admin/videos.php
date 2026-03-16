<?php
/**
 * Admin — Video moderation tab (included by admin/index.php)
 */

$pdo = get_db();

$filter_status = $_GET['status'] ?? 'pending';
$filter_theme = (int) ($_GET['vtheme'] ?? 0);
$page = max(1, (int) ($_GET['page'] ?? 1));
$per_page = 20;
$offset = ($page - 1) * $per_page;

$where = '1=1';
$params = [];

if ($filter_status && $filter_status !== 'all') {
    $where .= ' AND r.status = ?';
    $params[] = $filter_status;
}
if ($filter_theme > 0) {
    $where .= ' AND s.theme_id = ?';
    $params[] = $filter_theme;
}

$count_stmt = $pdo->prepare("SELECT COUNT(*) FROM recordings r JOIN signs s ON r.sign_id = s.id WHERE $where");
$count_stmt->execute($params);
$total = (int) $count_stmt->fetchColumn();

$params[] = $per_page;
$params[] = $offset;
$stmt = $pdo->prepare("
    SELECT r.*, s.word_sk, s.gloss_id, u.display_name, u.email
    FROM recordings r
    JOIN signs s ON r.sign_id = s.id
    JOIN users u ON r.user_id = u.id
    WHERE $where
    ORDER BY r.created_at DESC
    LIMIT ? OFFSET ?
");
$stmt->execute($params);
$recordings = $stmt->fetchAll();

$themes = $pdo->query('SELECT id, name, emoji FROM themes ORDER BY sort_order ASC')->fetchAll();
$total_pages = ceil($total / $per_page);
?>

<!-- Filters -->
<div class="admin-filter">
    <select onchange="location.href='/admin/?tab=videos&status='+this.value+'&vtheme=<?= $filter_theme ?>'">
        <option value="pending" <?= $filter_status === 'pending' ? 'selected' : '' ?>>Čakajúce</option>
        <option value="approved" <?= $filter_status === 'approved' ? 'selected' : '' ?>>Schválené</option>
        <option value="rejected" <?= $filter_status === 'rejected' ? 'selected' : '' ?>>Zamietnuté</option>
        <option value="all" <?= $filter_status === 'all' ? 'selected' : '' ?>>Všetky</option>
    </select>
    <select onchange="location.href='/admin/?tab=videos&status=<?= $filter_status ?>&vtheme='+this.value">
        <option value="0">Všetky témy</option>
        <?php foreach ($themes as $t): ?>
        <option value="<?= $t['id'] ?>" <?= $filter_theme == $t['id'] ? 'selected' : '' ?>>
            <?= htmlspecialchars($t['emoji'] . ' ' . $t['name']) ?>
        </option>
        <?php endforeach; ?>
    </select>
    <span style="font-size:13px;color:var(--gray);"><?= $total ?> videí</span>
</div>

<!-- Video grid -->
<div class="video-grid">
    <?php foreach ($recordings as $r):
        $dir = $r['status'] === 'approved' ? 'approved' : 'pending';
        $src = '/api/video.php?file=' . urlencode($r['video_filename']) . '&dir=' . $dir;
        $status_labels = ['pending' => '⏳', 'approved' => '✅', 'rejected' => '❌'];
        $status_icon = $status_labels[$r['status']] ?? '?';
    ?>
    <div class="video-card">
        <div class="video-preview-wrap" data-src="<?= htmlspecialchars($src) ?>">
            <video muted loop playsinline preload="metadata"
                   src="<?= htmlspecialchars($src) ?>"
                   onmouseenter="hoverPlay(this)" onmouseleave="hoverPause(this)"
            ></video>
            <span class="video-status-badge"><?= $status_icon ?></span>
            <button class="video-play-btn" onclick="openVideoModal('<?= htmlspecialchars($src, ENT_QUOTES) ?>')" aria-label="Prehrať video">▶</button>
        </div>
        <div class="video-card-info">
            <strong><?= htmlspecialchars($r['word_sk']) ?></strong>
            <span class="video-card-meta">
                <?= htmlspecialchars($r['display_name'] ?: $r['email']) ?>
                · <?= date('d.m', strtotime($r['created_at'])) ?>
                · 👍<?= $r['validations_up'] ?> 👎<?= $r['validations_down'] ?>
            </span>
        </div>
        <div class="video-card-actions">
            <?php if ($r['status'] === 'pending'): ?>
            <form method="POST" action="/admin/api/videos.php" style="display:inline;">
                <input type="hidden" name="action" value="approve">
                <input type="hidden" name="recording_id" value="<?= $r['id'] ?>">
                <?= csrf_field() ?>
                <button type="submit" class="btn-approve" title="Schváliť">✓</button>
            </form>
            <form method="POST" action="/admin/api/videos.php" style="display:inline;">
                <input type="hidden" name="action" value="reject">
                <input type="hidden" name="recording_id" value="<?= $r['id'] ?>">
                <?= csrf_field() ?>
                <button type="submit" class="btn-delete" title="Zamietnuť" style="background:#F59E0B;">✗</button>
            </form>
            <?php endif; ?>
            <form method="POST" action="/admin/api/videos.php" style="display:inline;"
                  onsubmit="return confirm('Natrvalo zmazať video?')">
                <input type="hidden" name="action" value="delete">
                <input type="hidden" name="recording_id" value="<?= $r['id'] ?>">
                <?= csrf_field() ?>
                <button type="submit" class="btn-delete" title="Vymazať">🗑</button>
            </form>
        </div>
    </div>
    <?php endforeach; ?>
</div>

<?php if (empty($recordings)): ?>
<p style="text-align:center;color:var(--gray);margin:40px 0;">Žiadne videá.</p>
<?php endif; ?>

<!-- Pagination -->
<?php if ($total_pages > 1): ?>
<div style="display:flex;justify-content:center;gap:8px;margin-top:16px;">
    <?php for ($p = 1; $p <= $total_pages; $p++): ?>
    <a href="/admin/?tab=videos&status=<?= $filter_status ?>&vtheme=<?= $filter_theme ?>&page=<?= $p ?>"
       style="padding:6px 12px;border-radius:6px;font-size:14px;text-decoration:none;<?= $p == $page ? 'background:var(--blue);color:white;' : 'background:var(--light-gray);color:var(--dark);' ?>">
        <?= $p ?>
    </a>
    <?php endfor; ?>
</div>
<?php endif; ?>

<!-- Video modal -->
<div id="video-modal" class="video-modal" style="display:none;" onclick="this.style.display='none'">
    <button class="close-btn" onclick="document.getElementById('video-modal').style.display='none'">×</button>
    <video id="modal-video" controls autoplay style="max-width:90%;max-height:80vh;border-radius:12px;"></video>
</div>

<style>
.video-play-btn {
    position: absolute; inset: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.3); border: none; cursor: pointer;
    font-size: 32px; color: white; display: flex; align-items: center;
    justify-content: center; opacity: 1; transition: opacity 0.2s;
}
.video-preview-wrap:hover .video-play-btn { opacity: 0; }
@media (max-width: 767px) {
    .video-preview-wrap:hover .video-play-btn { opacity: 1; }
}
</style>
<script>
// Desktop: hover to preview (silently catch if browser blocks)
function hoverPlay(el) {
    if (window.innerWidth < 768) return; // mobile: tap play button instead
    var p = el.play();
    if (p) p.catch(function(){});
}
function hoverPause(el) {
    try { el.pause(); el.currentTime = 0; } catch(e) {}
}

// Modal: works on all browsers — user gesture (click/tap) triggers play
function openVideoModal(src) {
    event.stopPropagation();
    var modal = document.getElementById('video-modal');
    var video = document.getElementById('modal-video');
    video.src = src;
    modal.style.display = 'flex';
    video.play().catch(function(){});
}
document.getElementById('video-modal').addEventListener('click', function(e) {
    if (e.target === this || e.target.classList.contains('close-btn')) {
        var video = document.getElementById('modal-video');
        video.pause(); video.removeAttribute('src'); video.load();
        this.style.display = 'none';
    }
});
</script>
</script>
