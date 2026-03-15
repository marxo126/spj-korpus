<?php
/**
 * Admin — Words management tab (included by admin/index.php)
 */

$pdo = get_db();

// Get filter params
$filter_theme = (int) ($_GET['theme_filter'] ?? 0);
$filter_search = trim($_GET['search'] ?? '');

// Build query
$where = '1=1';
$params = [];
if ($filter_theme > 0) {
    $where .= ' AND s.theme_id = ?';
    $params[] = $filter_theme;
}
if ($filter_search) {
    $where .= ' AND (s.word_sk LIKE ? OR s.gloss_id LIKE ?)';
    $params[] = "%$filter_search%";
    $params[] = "%$filter_search%";
}

$stmt = $pdo->prepare("
    SELECT s.*, t.name as theme_name, t.emoji as theme_emoji
    FROM signs s
    LEFT JOIN themes t ON s.theme_id = t.id
    WHERE $where
    ORDER BY t.sort_order ASC, s.sort_order_in_theme ASC, s.word_sk ASC
");
$stmt->execute($params);
$signs = $stmt->fetchAll();

$themes = $pdo->query('SELECT id, name, emoji FROM themes ORDER BY sort_order ASC')->fetchAll();
?>

<?php if (isset($_GET['success']) && $_GET['success'] === 'imported'): ?>
<div style="background:#DCFCE7;color:#15803D;padding:12px 16px;border-radius:8px;margin-bottom:16px;font-weight:600;">
    Importovaných: <?= (int)($_GET['count'] ?? 0) ?> slov<?php if (($_GET['skipped'] ?? 0) > 0): ?>, preskočených: <?= (int)$_GET['skipped'] ?><?php endif; ?>
</div>
<?php elseif (isset($_GET['error'])): ?>
<div style="background:#FEE2E2;color:#DC2626;padding:12px 16px;border-radius:8px;margin-bottom:16px;font-weight:600;">
    <?php
    $errors = [
        'no_file' => 'Žiadny súbor.',
        'read_fail' => 'Nepodarilo sa prečítať súbor.',
        'bad_type' => 'Neplatný formát — použite CSV.',
        'too_large' => 'Súbor je príliš veľký (max 1 MB).',
        'bad_header' => 'CSV musí obsahovať stĺpec "word_sk".',
        'not_found' => 'Slovo nebolo nájdené.',
    ];
    echo htmlspecialchars($errors[$_GET['error']] ?? 'Neznáma chyba.');
    ?>
</div>
<?php endif; ?>

<!-- Add word form -->
<details class="admin-form">
    <summary style="cursor:pointer;font-weight:700;font-size:15px;">+ Pridať slovo</summary>
    <form method="POST" action="/admin/api/words.php" style="margin-top:12px;">
        <input type="hidden" name="action" value="add">
        <?= csrf_field() ?>
        <div class="form-row">
            <div class="form-group">
                <label>Gloss ID</label>
                <input type="text" name="gloss_id" placeholder="VODA-1" required pattern="[A-Z0-9_-]+">
            </div>
            <div class="form-group">
                <label>Slovensky</label>
                <input type="text" name="word_sk" placeholder="voda" required>
            </div>
            <div class="form-group">
                <label>Téma</label>
                <select name="theme_id">
                    <option value="">— Bez témy —</option>
                    <?php foreach ($themes as $t): ?>
                    <option value="<?= $t['id'] ?>"><?= htmlspecialchars($t['emoji'] . ' ' . $t['name']) ?></option>
                    <?php endforeach; ?>
                </select>
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Link Posunky.sk</label>
                <input type="url" name="link_posunky" placeholder="https://posunky.sk/...">
            </div>
            <div class="form-group">
                <label>Link Dictio.info</label>
                <input type="url" name="link_dictio" placeholder="https://dictio.info/...">
            </div>
        </div>
        <button type="submit" class="btn btn-blue" style="width:auto;padding:10px 24px;">Pridať</button>
    </form>
</details>

<!-- CSV Import -->
<details class="admin-form">
    <summary style="cursor:pointer;font-weight:700;font-size:15px;">Importovať CSV</summary>
    <form method="POST" action="/admin/api/import.php" enctype="multipart/form-data" style="margin-top:12px;">
        <?= csrf_field() ?>
        <p style="font-size:13px;color:var(--gray);margin-bottom:8px;">
            Formát: gloss_id, word_sk, theme_name, link_posunky, link_dictio (max 500 riadkov)
        </p>
        <input type="file" name="csv_file" accept=".csv" required style="margin-bottom:10px;">
        <button type="submit" class="btn btn-gray" style="width:auto;padding:10px 24px;">Importovať</button>
    </form>
</details>

<!-- Filters -->
<div class="admin-filter">
    <div class="form-group">
        <select onchange="location.href='/admin/?tab=words&theme_filter='+this.value+'&search=<?= urlencode($filter_search) ?>'">
            <option value="0">Všetky témy</option>
            <?php foreach ($themes as $t): ?>
            <option value="<?= $t['id'] ?>" <?= $filter_theme == $t['id'] ? 'selected' : '' ?>>
                <?= htmlspecialchars($t['emoji'] . ' ' . $t['name']) ?>
            </option>
            <?php endforeach; ?>
        </select>
    </div>
    <div class="form-group">
        <input type="text" placeholder="Hľadať..." value="<?= htmlspecialchars($filter_search) ?>"
               onchange="location.href='/admin/?tab=words&theme_filter=<?= $filter_theme ?>&search='+encodeURIComponent(this.value)">
    </div>
    <span style="font-size:13px;color:var(--gray);"><?= count($signs) ?> slov</span>
</div>

<!-- Word table -->
<div style="overflow-x:auto;">
<table class="admin-table">
    <thead>
        <tr>
            <th>Slovo</th>
            <th>Gloss</th>
            <th>Téma</th>
            <th>Nahrávky</th>
            <th>Linky</th>
            <th>Akcie</th>
        </tr>
    </thead>
    <tbody>
        <?php foreach ($signs as $s): ?>
        <tr>
            <td style="font-weight:600;"><?= htmlspecialchars($s['word_sk']) ?></td>
            <td><code style="font-size:12px;"><?= htmlspecialchars($s['gloss_id']) ?></code></td>
            <td><?= $s['theme_name'] ? htmlspecialchars($s['theme_emoji'] . ' ' . $s['theme_name']) : '—' ?></td>
            <td><?= $s['total_recordings'] ?>/<?= $s['target_recordings'] ?></td>
            <td>
                <?php if ($s['link_posunky']): ?><a href="<?= htmlspecialchars($s['link_posunky']) ?>" target="_blank" style="font-size:12px;">P</a><?php endif; ?>
                <?php if ($s['link_dictio']): ?><a href="<?= htmlspecialchars($s['link_dictio']) ?>" target="_blank" style="font-size:12px;">D</a><?php endif; ?>
            </td>
            <td class="admin-actions">
                <button class="edit-btn" data-sign='<?= htmlspecialchars(json_encode($s, JSON_UNESCAPED_UNICODE), ENT_QUOTES, "UTF-8") ?>'>Upraviť</button>
                <form method="POST" action="/admin/api/words.php" style="display:inline;"
                      onsubmit="return confirm('Zmazať slovo ' + <?= json_encode($s['word_sk'], JSON_UNESCAPED_UNICODE) ?> + ' a všetky jeho nahrávky?')">
                    <input type="hidden" name="action" value="delete">
                    <input type="hidden" name="sign_id" value="<?= $s['id'] ?>">
                    <?= csrf_field() ?>
                    <button type="submit" class="delete">Zmazať</button>
                </form>
            </td>
        </tr>
        <?php endforeach; ?>
    </tbody>
</table>
</div>

<!-- Edit modal -->
<div id="edit-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:200;padding:20px;overflow:auto;">
    <div style="max-width:500px;margin:40px auto;background:white;border-radius:16px;padding:24px;">
        <h3 style="margin-bottom:16px;">Upraviť slovo</h3>
        <form method="POST" action="/admin/api/words.php">
            <input type="hidden" name="action" value="edit">
            <input type="hidden" name="sign_id" id="edit-sign-id">
            <?= csrf_field() ?>
            <div class="form-group">
                <label>Gloss ID</label>
                <input type="text" name="gloss_id" id="edit-gloss-id" required>
            </div>
            <div class="form-group">
                <label>Slovensky</label>
                <input type="text" name="word_sk" id="edit-word-sk" required>
            </div>
            <div class="form-group">
                <label>Téma</label>
                <select name="theme_id" id="edit-theme-id">
                    <option value="">— Bez témy —</option>
                    <?php foreach ($themes as $t): ?>
                    <option value="<?= $t['id'] ?>"><?= htmlspecialchars($t['emoji'] . ' ' . $t['name']) ?></option>
                    <?php endforeach; ?>
                </select>
            </div>
            <div class="form-group">
                <label>Link Posunky.sk</label>
                <input type="url" name="link_posunky" id="edit-link-posunky">
            </div>
            <div class="form-group">
                <label>Link Dictio.info</label>
                <input type="url" name="link_dictio" id="edit-link-dictio">
            </div>
            <div style="display:flex;gap:10px;">
                <button type="submit" class="btn btn-blue" style="flex:1;">Uložiť</button>
                <button type="button" class="btn btn-gray" style="flex:1;" onclick="document.getElementById('edit-modal').style.display='none'">Zrušiť</button>
            </div>
        </form>
    </div>
</div>

<script>
document.addEventListener('click', function(e) {
    var btn = e.target.closest('.edit-btn');
    if (!btn) return;
    var data = JSON.parse(btn.dataset.sign);
    document.getElementById('edit-sign-id').value = data.id;
    document.getElementById('edit-gloss-id').value = data.gloss_id;
    document.getElementById('edit-word-sk').value = data.word_sk;
    document.getElementById('edit-theme-id').value = data.theme_id || '';
    document.getElementById('edit-link-posunky').value = data.link_posunky || '';
    document.getElementById('edit-link-dictio').value = data.link_dictio || '';
    document.getElementById('edit-modal').style.display = 'block';
});
</script>
