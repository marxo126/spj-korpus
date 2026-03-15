<?php
/**
 * SPJ Collector — Consent collection (Google OAuth users)
 */
require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
require_login();

$user = get_user();

// Already has all consents -> redirect to themes
if ($user['consent_service'] && $user['consent_biometric'] && $user['consent_retention']) {
    header('Location: /themes.php');
    exit;
}

$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_csrf();
    $consent_service = isset($_POST['consent_service']) ? 1 : 0;
    $consent_biometric = isset($_POST['consent_biometric']) ? 1 : 0;
    $consent_retention = isset($_POST['consent_retention']) ? 1 : 0;

    if (!$consent_service || !$consent_biometric || !$consent_retention) {
        $error = 'Pre pokračovanie musíte súhlasiť so všetkými bodmi.';
    } else {
        $pdo = get_db();
        $stmt = $pdo->prepare('
            UPDATE users SET consent_service = 1, consent_biometric = 1,
                consent_retention = 1, consent_date = NOW(),
                consent_service_date = NOW(), consent_biometric_date = NOW(),
                consent_retention_date = NOW()
            WHERE id = ?
        ');
        $stmt->execute([$user['id']]);
        header('Location: /themes.php');
        exit;
    }
}

$page_title = 'Súhlasy — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1 style="text-align: center; margin-bottom: 16px;">Súhlasy</h1>

<div class="card">
    <p style="font-size: 14px; color: var(--gray); margin-bottom: 16px;">
        Pre používanie aplikácie potrebujeme váš súhlas s nasledujúcimi bodmi.
    </p>

    <?php if ($error): ?>
        <div class="error-msg"><?= htmlspecialchars($error) ?></div>
    <?php endif; ?>

    <form method="POST">
        <?= csrf_field() ?>

        <div class="consent-card">
            <div class="checkbox-group">
                <span class="consent-icon">👤</span>
                <input type="checkbox" id="gdpr-service" name="consent_service" required>
                <label for="gdpr-service" class="consent-text">Môj účet a osobné údaje</label>
            </div>
            <p class="consent-detail">Súhlasím, že moje údaje (email, meno, škola, mesto, vek) sa použijú na vytvorenie účtu a výskum.</p>
        </div>

        <div class="consent-card">
            <div class="checkbox-group">
                <span class="consent-icon">📹</span>
                <input type="checkbox" id="gdpr-biometric" name="consent_biometric" required>
                <label for="gdpr-biometric" class="consent-text">Video s mojou tvárou</label>
            </div>
            <p class="consent-detail">Súhlasím s nahrávaním videa, kde je vidieť moju tvár a ruky. Viem, že tvár = biometrické údaje.</p>
        </div>

        <div class="consent-card">
            <div class="checkbox-group">
                <span class="consent-icon">🔬</span>
                <input type="checkbox" id="gdpr-retention" name="consent_retention" required>
                <label for="gdpr-retention" class="consent-text">Videá zostanú vo výskume</label>
            </div>
            <p class="consent-detail">Aj keď zmažem účet, anonymizované videá a pohybové dáta zostanú v korpuse SPJ na výskum.</p>
        </div>

        <div style="background: #EFF6FF; border-radius: 8px; padding: 12px; margin-top: 12px; text-align: center;">
            <a href="/terms.php" target="_blank" style="color: var(--blue); font-weight: 700; font-size: 15px; text-decoration: none;">
                Prečítajte si podmienky používania
            </a>
            <p style="font-size: 12px; color: var(--gray); margin-top: 4px;">Súhlas môžete kedykoľvek odvolať.</p>
        </div>

        <button type="submit" class="btn btn-blue" style="margin-top: 16px;">Súhlasím a pokračujem</button>
    </form>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
