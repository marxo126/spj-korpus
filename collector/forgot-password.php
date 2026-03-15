<?php
/**
 * SPJ Collector — Forgot Password
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

if (is_logged_in()) {
    header('Location: /themes.php');
    exit;
}

$error = '';
$success = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_csrf();

    if (!check_rate_limit('password_reset', 3)) {
        $error = 'Príliš veľa pokusov. Skúste to o 15 minút.';
    } else {
        $email = trim($_POST['email'] ?? '');
        if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $error = 'Zadajte platný email.';
        } else {
            $token = request_password_reset($email);
            if ($token) {
                send_reset_email($email, $token);
            }
            // Always show success to prevent email enumeration
            $success = 'Ak je email registrovaný, poslali sme vám odkaz na obnovenie hesla. Skontrolujte aj spam.';
        }
    }
}

$page_title = 'Obnovenie hesla — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<div style="text-align: center; margin-bottom: 24px; padding-top: 20px;">
    <div style="font-size: 48px; margin-bottom: 8px;">🔑</div>
    <h1>Zabudnuté heslo</h1>
    <p style="color: var(--gray); font-size: 15px; margin-top: 4px;">
        Zadajte email a pošleme vám odkaz na obnovenie
    </p>
</div>

<?php if ($success): ?>
    <div style="background: #DCFCE7; color: #15803D; padding: 12px 16px; border-radius: 10px; font-weight: 600; margin-bottom: 16px;">
        <?= htmlspecialchars($success) ?>
    </div>
<?php endif; ?>

<?php if ($error): ?>
    <div class="error-msg"><?= htmlspecialchars($error) ?></div>
<?php endif; ?>

<div class="card">
    <form method="POST">
        <?= csrf_field() ?>
        <div class="form-group">
            <label for="email">Email</label>
            <input type="email" id="email" name="email" placeholder="vas@email.sk" required
                   value="<?= htmlspecialchars($_POST['email'] ?? '') ?>">
        </div>
        <button type="submit" class="btn btn-blue">Odoslať odkaz</button>
    </form>
    <div style="text-align: center; margin-top: 16px;">
        <a href="/index.php" style="color: var(--blue); font-weight: 600; text-decoration: none;">
            ← Späť na prihlásenie
        </a>
    </div>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
