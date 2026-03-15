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
        <div class="video-preview-wrap"
             onclick="openVideoModal('<?= htmlspecialchars($src, ENT_QUOTES) ?>')" data-src="<?= htmlspecialchars($src) ?>">
            <video muted loop playsinline preload="none"
                   onmouseenter="if(!this.src)this.src='<?= htmlspecialchars($src) ?>';this.play()"
                   onmouseleave="this.pause();this.currentTime=0"
                   ontouchstart="if(!this.src)this.src='<?= htmlspecialchars($src) ?>';this.play()"
                   ontouchend="this.pause();this.currentTime=0"
                   data-src="<?= htmlspecialchars($src) ?>"
            ></video>
            <span class="video-status-badge"><?= $status_icon ?></span>
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
                <button type="submit" class="btn-approve">✓</button>
            </form>
            <?php endif; ?>
            <form method="POST" action="/admin/api/videos.php" style="display:inline;"
                  onsubmit="return confirm('Zmazať video?')">
                <input type="hidden" name="action" value="delete">
                <input type="hidden" name="recording_id" value="<?= $r['id'] ?>">
                <?= csrf_field() ?>
                <button type="submit" class="btn-delete">✗</button>
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

<script>
function openVideoModal(src) {
    var video = document.getElementById('modal-video');
    video.src = src;
    document.getElementById('video-modal').style.display = 'flex';
}
</script>
