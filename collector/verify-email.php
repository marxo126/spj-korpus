<?php
/**
 * SPJ Collector — Email Verification
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

$token = $_GET['token'] ?? '';
$success = false;
$error = '';

if ($token) {
    $user_id = verify_email_token($token);
    if ($user_id) {
        $success = true;
        // Auto-login if not logged in
        if (!is_logged_in()) {
            login_user($user_id);
        }
    } else {
        $error = 'Neplatný alebo expirovaný odkaz. Skúste sa prihlásiť a požiadať o nový.';
    }
} else {
    $error = 'Chýba overovací token.';
}

// Handle resend request
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['resend']) && is_logged_in()) {
    if (!check_rate_limit('verify_email', 3, 900)) {
        $error = 'Príliš veľa pokusov. Skúste to o 15 minút.';
    } else {
        $user = get_user();
        if ($user && !$user['email_verified']) {
            $new_token = create_email_verification($user['id'], $user['email']);
            send_verification_email($user['email'], $new_token);
            $success = false;
            $resent = true;
        }
    }
}

$page_title = 'Overenie emailu — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<div style="text-align: center; padding-top: 40px;">
    <?php if ($success): ?>
    <div style="font-size: 64px; margin-bottom: 16px;">✅</div>
    <h1>Email overený!</h1>
    <p style="color: var(--gray); margin: 12px 0 24px;">Váš účet je teraz plne aktívny. Môžete začať nahrávať posunky.</p>
    <a href="/themes.php" class="btn btn-blue" style="width: auto; padding: 14px 32px;">Začať nahrávať →</a>

    <?php elseif (isset($resent) && $resent): ?>
    <div style="font-size: 64px; margin-bottom: 16px;">📧</div>
    <h1>Email odoslaný!</h1>
    <p style="color: var(--gray); margin: 12px 0 24px;">Skontrolujte svoju schránku (aj spam). Odkaz je platný 24 hodín.</p>

    <?php else: ?>
    <div style="font-size: 64px; margin-bottom: 16px;">⚠️</div>
    <h1>Overenie zlyhalo</h1>
    <p style="color: var(--gray); margin: 12px 0 24px;"><?= htmlspecialchars($error) ?></p>
    <?php if (is_logged_in() && !is_email_verified()): ?>
    <form method="POST" style="margin-top: 16px;">
        <?= csrf_field() ?>
        <input type="hidden" name="resend" value="1">
        <button type="submit" class="btn btn-blue" style="width: auto; padding: 14px 32px;">Poslať nový overovací email</button>
    </form>
    <?php else: ?>
    <a href="/" class="btn btn-blue" style="width: auto; padding: 14px 32px;">← Späť na prihlásenie</a>
    <?php endif; ?>
    <?php endif; ?>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
