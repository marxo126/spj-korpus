<?php
/**
 * SPJ Collector — Reset Password
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

if (is_logged_in()) {
    header('Location: /themes.php');
    exit;
}

$error = '';
$success = '';
$token = $_GET['token'] ?? $_POST['token'] ?? '';
$valid = false;

if ($token) {
    $valid = validate_reset_token($token) !== null;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && $token) {
    require_csrf();

    $password = $_POST['password'] ?? '';
    $password2 = $_POST['password2'] ?? '';

    if (strlen($password) < 6) {
        $error = 'Heslo musí mať aspoň 6 znakov.';
    } elseif ($password !== $password2) {
        $error = 'Heslá sa nezhodujú.';
    } else {
        if (reset_password($token, $password)) {
            $success = 'Heslo bolo zmenené. Teraz sa môžete prihlásiť.';
            $valid = false; // Hide form after success
        } else {
            $error = 'Odkaz vypršal alebo je neplatný. Požiadajte o nový.';
            $valid = false;
        }
    }
}

$page_title = 'Nové heslo — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<div style="text-align: center; margin-bottom: 24px; padding-top: 20px;">
    <div style="font-size: 48px; margin-bottom: 8px;">🔐</div>
    <h1>Nové heslo</h1>
</div>

<?php if ($success): ?>
    <div style="background: #DCFCE7; color: #15803D; padding: 12px 16px; border-radius: 10px; font-weight: 600; margin-bottom: 16px;">
        <?= htmlspecialchars($success) ?>
    </div>
    <div class="card" style="text-align: center;">
        <a href="/index.php" class="btn btn-blue">Prihlásiť sa →</a>
    </div>
<?php elseif ($valid): ?>
    <?php if ($error): ?>
        <div class="error-msg"><?= htmlspecialchars($error) ?></div>
    <?php endif; ?>
    <div class="card">
        <form method="POST">
            <?= csrf_field() ?>
            <input type="hidden" name="token" value="<?= htmlspecialchars($token) ?>">
            <div class="form-group">
                <label for="password">Nové heslo</label>
                <input type="password" id="password" name="password" placeholder="Aspoň 6 znakov" required minlength="6">
            </div>
            <div class="form-group">
                <label for="password2">Zopakujte heslo</label>
                <input type="password" id="password2" name="password2" placeholder="Aspoň 6 znakov" required minlength="6">
            </div>
            <button type="submit" class="btn btn-blue">Zmeniť heslo</button>
        </form>
    </div>
<?php else: ?>
    <div class="card" style="text-align: center;">
        <p style="color: var(--gray); margin-bottom: 16px;">
            Odkaz na obnovenie hesla je neplatný alebo vypršal.
        </p>
        <a href="/forgot-password.php" class="btn btn-blue">Požiadať o nový odkaz</a>
    </div>
<?php endif; ?>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
