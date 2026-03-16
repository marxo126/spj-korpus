<?php
/**
 * Admin — Users management tab (included by admin/index.php)
 * Only visible to superadmin (is_admin).
 */

$pdo = get_db();

// Handle success/error messages
$msg = $_GET['msg'] ?? '';
$msg_map = [
    'role_updated' => ['✅ Rola bola aktualizovaná.', '#DCFCE7', '#15803D'],
    'role_error' => ['❌ Chyba pri aktualizácii role.', '#FEE2E2', '#DC2626'],
    'self_error' => ['❌ Nemôžete zmeniť vlastnú rolu.', '#FEF3C7', '#A16207'],
];

// Filters
$filter_role = $_GET['role'] ?? 'all';
$filter_search = trim($_GET['q'] ?? '');

$where = '1=1';
$params = [];
if ($filter_role === 'admin') {
    $where .= ' AND is_admin = 1';
} elseif ($filter_role === 'researcher') {
    $where .= ' AND is_researcher = 1 AND is_admin = 0';
} elseif ($filter_role === 'user') {
    $where .= ' AND is_admin = 0 AND is_researcher = 0';
}
if ($filter_search) {
    $where .= ' AND (email LIKE ? OR display_name LIKE ?)';
    $params[] = "%$filter_search%";
    $params[] = "%$filter_search%";
}

$stmt = $pdo->prepare("
    SELECT id, email, display_name, is_admin, is_researcher,
           total_recordings, created_at, last_active,
           consent_service, consent_biometric, consent_retention,
           COALESCE(email_verified, 1) as email_verified
    FROM users
    WHERE $where
    ORDER BY is_admin DESC, is_researcher DESC, total_recordings DESC
");
$stmt->execute($params);
$users = $stmt->fetchAll();

$counts = $pdo->query("
    SELECT
        COUNT(*) as total,
        SUM(is_admin) as admins,
        SUM(is_researcher AND NOT is_admin) as researchers,
        SUM(NOT is_admin AND NOT is_researcher) as regular
    FROM users
")->fetch();
?>

<?php if (isset($msg_map[$msg])): ?>
<div style="background:<?= $msg_map[$msg][1] ?>;color:<?= $msg_map[$msg][2] ?>;padding:12px 16px;border-radius:8px;margin-bottom:16px;font-weight:600;">
    <?= $msg_map[$msg][0] ?>
</div>
<?php endif; ?>

<!-- Stats -->
<div class="stat-cards-grid" style="margin-bottom:16px;">
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:var(--blue);"><?= $counts['total'] ?></div>
        <div style="font-size:13px;color:var(--gray);">Celkom</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:#DC2626;"><?= $counts['admins'] ?></div>
        <div style="font-size:13px;color:var(--gray);">Admin</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:#F59E0B;"><?= $counts['researchers'] ?></div>
        <div style="font-size:13px;color:var(--gray);">Výskumník</div>
    </div>
    <div class="stat-card">
        <div style="font-size:24px;font-weight:800;color:var(--green);"><?= $counts['regular'] ?></div>
        <div style="font-size:13px;color:var(--gray);">Používateľ</div>
    </div>
</div>

<!-- Filters -->
<div class="admin-filter" style="margin-bottom:16px;">
    <select onchange="location.href='/admin/?tab=users&role='+this.value+'&q=<?= urlencode($filter_search) ?>'">
        <option value="all" <?= $filter_role === 'all' ? 'selected' : '' ?>>Všetci</option>
        <option value="admin" <?= $filter_role === 'admin' ? 'selected' : '' ?>>Admini</option>
        <option value="researcher" <?= $filter_role === 'researcher' ? 'selected' : '' ?>>Výskumníci</option>
        <option value="user" <?= $filter_role === 'user' ? 'selected' : '' ?>>Používatelia</option>
    </select>
    <input type="text" placeholder="Hľadať email alebo meno..." value="<?= htmlspecialchars($filter_search) ?>"
           onchange="location.href='/admin/?tab=users&role=<?= $filter_role ?>&q='+encodeURIComponent(this.value)"
           style="padding:8px 12px;border:2px solid var(--light-gray);border-radius:8px;font-size:14px;flex:1;min-width:200px;">
    <span style="font-size:13px;color:var(--gray);"><?= count($users) ?> používateľov</span>
</div>

<!-- Users table -->
<div style="overflow-x:auto;">
<table class="admin-table" style="width:100%;border-collapse:collapse;font-size:14px;">
<thead>
<tr style="border-bottom:2px solid var(--light-gray);text-align:left;">
    <th style="padding:10px 8px;">Meno</th>
    <th style="padding:10px 8px;">Email</th>
    <th style="padding:10px 8px;">Rola</th>
    <th style="padding:10px 8px;">Nahrávky</th>
    <th style="padding:10px 8px;">Súhlasy</th>
    <th style="padding:10px 8px;">Registrácia</th>
    <th style="padding:10px 8px;">Posledná aktivita</th>
    <th style="padding:10px 8px;">Akcie</th>
</tr>
</thead>
<tbody>
<?php foreach ($users as $u):
    $role = $u['is_admin'] ? 'admin' : ($u['is_researcher'] ? 'researcher' : 'user');
    $role_label = ['admin' => '🔴 Admin', 'researcher' => '🟡 Výskumník', 'user' => '🟢 Používateľ'];
    $is_self = ($u['id'] == get_user_id());
?>
<tr style="border-bottom:1px solid var(--light-gray);<?= $is_self ? 'background:rgba(37,99,235,0.05);' : '' ?>">
    <td style="padding:10px 8px;font-weight:600;">
        <?= htmlspecialchars($u['display_name'] ?: '—') ?>
        <?= $is_self ? '<span style="font-size:11px;color:var(--blue);">(vy)</span>' : '' ?>
    </td>
    <td style="padding:10px 8px;font-size:13px;color:var(--gray);">
        <?= $u['email_verified'] ? '✅' : '⚠️' ?>
        <?= htmlspecialchars($u['email']) ?>
    </td>
    <td style="padding:10px 8px;"><?= $role_label[$role] ?></td>
    <td style="padding:10px 8px;text-align:center;"><?= $u['total_recordings'] ?></td>
    <td style="padding:10px 8px;font-size:13px;">
        <?= $u['consent_service'] ? '✅' : '❌' ?>
        <?= $u['consent_biometric'] ? '✅' : '❌' ?>
        <?= $u['consent_retention'] ? '✅' : '❌' ?>
    </td>
    <td style="padding:10px 8px;font-size:13px;color:var(--gray);"><?= $u['created_at'] ? date('d.m.Y', strtotime($u['created_at'])) : '—' ?></td>
    <td style="padding:10px 8px;font-size:13px;color:var(--gray);"><?= $u['last_active'] ?: '—' ?></td>
    <td style="padding:10px 8px;">
        <?php if (!$is_self): ?>
        <form method="POST" action="/admin/api/users.php" style="display:inline;">
            <?= csrf_field() ?>
            <input type="hidden" name="user_id" value="<?= $u['id'] ?>">
            <select name="role" onchange="this.form.submit()" style="padding:4px 8px;border-radius:6px;border:1px solid var(--light-gray);font-size:13px;cursor:pointer;">
                <option value="user" <?= $role === 'user' ? 'selected' : '' ?>>Používateľ</option>
                <option value="researcher" <?= $role === 'researcher' ? 'selected' : '' ?>>Výskumník</option>
                <option value="admin" <?= $role === 'admin' ? 'selected' : '' ?>>Admin</option>
            </select>
        </form>
        <?php else: ?>
        <span style="font-size:12px;color:var(--gray);">—</span>
        <?php endif; ?>
    </td>
</tr>
<?php endforeach; ?>
</tbody>
</table>
</div>

<?php if (empty($users)): ?>
<p style="text-align:center;color:var(--gray);margin:40px 0;">Žiadni používatelia.</p>
<?php endif; ?>
