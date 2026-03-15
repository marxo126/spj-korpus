<?php
/**
 * SPJ Collector — User profile + progress
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
require_login();

$user = get_user();
$pdo = get_db();
$stats = get_community_stats();

// Today's count
$stmt = $pdo->prepare('SELECT COUNT(*) FROM recordings WHERE user_id = ? AND DATE(created_at) = CURDATE()');
$stmt->execute([$user['id']]);
$today = (int) $stmt->fetchColumn();

// Handle profile update
$saved = false;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'update_profile') {
    require_csrf();
    $stmt = $pdo->prepare('
        UPDATE users SET display_name = ?, public_name = ?, show_public_name = ?,
                        school = ?, location = ?, age_range = ?, gender = ?
        WHERE id = ?
    ');
    $stmt->execute([
        trim($_POST['display_name'] ?? ''),
        trim($_POST['public_name'] ?? ''),
        isset($_POST['show_public_name']) ? 1 : 0,
        trim($_POST['school'] ?? ''),
        trim($_POST['location'] ?? ''),
        $_POST['age_range'] ?? null,
        $_POST['gender'] ?? null,
        $user['id']
    ]);
    $user = get_user(); // refresh
    $saved = true;
}

$page_title = 'Profil — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1 style="margin-bottom: 20px;">👤 Profil</h1>

<?php if ($saved): ?>
<div style="background: #DCFCE7; color: #15803D; padding: 12px 16px; border-radius: 10px; font-weight: 600; margin-bottom: 16px;">
    ✅ Uložené
</div>
<?php endif; ?>

<!-- Stats -->
<div class="card">
    <div class="stats-grid">
        <div>
            <span class="stat-num" style="color: var(--blue);"><?= $today ?></span>
            <span class="stat-label">dnes</span>
        </div>
        <div>
            <span class="stat-num" style="color: var(--green);"><?= number_format($user['total_recordings']) ?></span>
            <span class="stat-label">celkom</span>
        </div>
    </div>

    <div class="progress-info" style="margin-top: 16px;">
        <span class="label">Dnešný cieľ</span>
        <span class="count"><?= $today ?> / <?= DAILY_GOAL ?></span>
    </div>
    <div class="progress-bar">
        <div class="fill" style="width: <?= min(100, ($today / DAILY_GOAL) * 100) ?>%;"></div>
    </div>
</div>

<!-- Profile form -->
<div class="card">
    <h2 style="margin-bottom: 16px;">Upraviť profil</h2>
    <form method="POST">
        <input type="hidden" name="action" value="update_profile">
        <?= csrf_field() ?>

        <div class="form-group">
            <label>Meno</label>
            <input type="text" name="display_name" value="<?= htmlspecialchars($user['display_name'] ?? '') ?>">
        </div>

        <div class="form-group">
            <label>Škola / organizácia</label>
            <input type="text" name="school" value="<?= htmlspecialchars($user['school'] ?? '') ?>">
        </div>

        <div class="form-group">
            <label>Mesto / región</label>
            <input type="text" name="location" value="<?= htmlspecialchars($user['location'] ?? '') ?>">
        </div>

        <div class="form-group">
            <label>Vek</label>
            <div class="radio-group">
                <?php foreach (['under_18' => 'do 18', '18-25' => '18–25', '26-35' => '26–35', '36-50' => '36–50', '50+' => '50+'] as $val => $label): ?>
                <label>
                    <input type="radio" name="age_range" value="<?= $val ?>"
                        <?= ($user['age_range'] ?? '') === $val ? 'checked' : '' ?>>
                    <?= $label ?>
                </label>
                <?php endforeach; ?>
            </div>
        </div>

        <div class="form-group">
            <label>Pohlavie</label>
            <div class="radio-group">
                <?php foreach (['woman' => 'Žena', 'man' => 'Muž', 'neutral' => 'Neuvádzam'] as $val => $label): ?>
                <label>
                    <input type="radio" name="gender" value="<?= $val ?>"
                        <?= ($user['gender'] ?? '') === $val ? 'checked' : '' ?>>
                    <?= $label ?>
                </label>
                <?php endforeach; ?>
            </div>
        </div>

        <h3 style="margin: 20px 0 12px; padding-top: 12px; border-top: 1px solid var(--light-gray);">
            Verejné poďakovanie
        </h3>

        <div class="checkbox-group">
            <input type="checkbox" id="show-public" name="show_public_name"
                <?= $user['show_public_name'] ? 'checked' : '' ?>
                onchange="document.getElementById('public-name-field').style.display = this.checked ? 'block' : 'none'">
            <label for="show-public">Zobraziť moje meno na stránke Ďakujeme</label>
        </div>
        <div class="form-group" id="public-name-field"
             style="display: <?= $user['show_public_name'] ? 'block' : 'none' ?>;">
            <label>Meno na stránke</label>
            <input type="text" name="public_name" value="<?= htmlspecialchars($user['public_name'] ?? '') ?>"
                   placeholder="napr. Janka M.">
        </div>

        <button type="submit" class="btn btn-blue" style="margin-top: 16px;">Uložiť zmeny</button>
    </form>
</div>

<!-- Logout -->
<a href="/api/auth.php?action=logout" class="btn btn-outline" style="margin-top: 8px;">
    Odhlásiť sa
</a>

<!-- GDPR: Export data -->
<div class="card" style="margin-top: 16px;">
    <h3 style="margin-bottom: 8px;">Export údajov</h3>
    <p style="font-size: 13px; color: var(--gray); margin-bottom: 12px;">
        Stiahnite si všetky svoje osobné údaje vo formáte JSON (GDPR Art. 20).
    </p>
    <form method="POST" action="/api/export.php">
        <?= csrf_field() ?>
        <button type="submit" class="btn btn-outline">
            Exportovať moje údaje
        </button>
    </form>
</div>

<!-- GDPR: Delete account -->
<div class="card" style="margin-top: 32px; border: 1px solid #FCA5A5;">
    <h3 style="color: var(--red); margin-bottom: 8px;">Vymazať účet</h3>
    <p style="font-size: 13px; color: var(--gray); margin-bottom: 12px;">
        Váš účet, meno a email budú vymazané. Videá (vrátane tváre) zostanú súčasťou výskumného korpusu SPJ.
        Chcete vymazať aj videá? Napíšte na <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a>
    </p>
    <form method="POST" action="/api/auth.php?action=delete_account"
          onsubmit="return confirm('Naozaj chcete vymazať váš účet? Osobné údaje budú vymazané, anonymizované videá zostanú vo výskumnom korpuse.');">
        <?= csrf_field() ?>
        <button type="submit" class="btn btn-outline" style="color: var(--red); border-color: var(--red);">
            Vymazať môj účet a údaje
        </button>
    </form>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
