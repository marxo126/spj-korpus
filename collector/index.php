<?php
/**
 * SPJ Collector — Home / Login / Register
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

// Already logged in → go to themes
if (is_logged_in()) {
    header('Location: /themes.php');
    exit;
}

$error = '';
$success = '';
$mode = $_GET['mode'] ?? 'login'; // 'login' or 'register'

// GDPR account deletion confirmation
if (isset($_GET['deleted'])) {
    $success = 'Váš účet a všetky údaje boli vymazané.';
}

// Handle form submission
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';

    require_csrf();

    if ($action === 'login') {
        if (!check_rate_limit('login', 10)) {
            $error = 'Príliš veľa pokusov. Skúste to neskôr.';
        } else {
            $email = trim($_POST['email'] ?? '');
            $password = $_POST['password'] ?? '';
            $user_id = verify_login($email, $password);
            if ($user_id) {
                login_user($user_id);
                header('Location: /themes.php');
                exit;
            }
            $error = 'Nesprávny email alebo heslo.';
        }
    }

    if ($action === 'register') {
        $email = trim($_POST['email'] ?? '');
        $password = $_POST['password'] ?? '';
        $password2 = $_POST['password2'] ?? '';
        if ($password !== $password2) {
            $error = 'Heslá sa nezhodujú.';
        } elseif (strlen($password) < 8) {
            $error = 'Heslo musí mať aspoň 8 znakov.';
        } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $error = 'Neplatný email.';
        } else {
            $demographics = [
                'display_name' => trim($_POST['display_name'] ?? ''),
                'public_name' => trim($_POST['public_name'] ?? ''),
                'show_public_name' => isset($_POST['show_public_name']) ? 1 : 0,
                'school' => trim($_POST['school'] ?? ''),
                'location' => trim($_POST['location'] ?? ''),
                'age_range' => $_POST['age_range'] ?? null,
                'gender' => $_POST['gender'] ?? null,
                'dominant_hand' => $_POST['dominant_hand'] ?? 'right',
                'consent_service' => isset($_POST['consent_service']) ? 1 : 0,
                'consent_biometric' => isset($_POST['consent_biometric']) ? 1 : 0,
                'consent_retention' => isset($_POST['consent_retention']) ? 1 : 0,
            ];
            $user_id = register_user($email, $password, $demographics);
            if ($user_id) {
                login_user($user_id);
                header('Location: /themes.php');
                exit;
            }
            $error = 'Email je už registrovaný.';
        }
        $mode = 'register';
    }
}

$stats = get_community_stats();
$page_title = SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<div style="text-align: center; margin-bottom: 20px; padding-top: 20px;">
    <img src="/img/spj-logo.png" alt="SPJ" style="width: 72px; height: 72px; margin-bottom: 8px;">
    <h1 style="font-family: 'Manrope', sans-serif; font-weight: 800;">Zber posunkov SPJ</h1>
    <p style="color: var(--gray); font-size: 15px; margin-top: 4px;">
        Pomôžte nám vytvoriť prvý veľký korpus slovenského posunkového jazyka
    </p>
</div>

<?php if ($mode === 'login'): ?>
<!-- Quick action buttons — visible immediately -->
<div style="display: flex; gap: 10px; margin-bottom: 20px;">
    <a href="#login-form" class="btn btn-blue" style="flex: 1; text-align: center; font-size: 16px; padding: 14px;">
        Prihlásiť sa
    </a>
    <a href="?mode=register" class="btn btn-green" style="flex: 1; text-align: center; font-size: 16px; padding: 14px; background: var(--green); color: white;">
        Registrovať sa
    </a>
</div>
<?php endif; ?>

<!-- ══ PURPOSE EXPLANATION ══ -->
<?php if ($mode === 'login'): ?>
<div style="margin-bottom: 20px;">

    <!-- What is this -->
    <div class="card" style="margin-bottom: 12px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 32px; line-height: 1;">📹</span>
            <div>
                <strong style="font-size: 16px;">Čo je to?</strong>
                <p style="color: var(--gray); font-size: 14px; margin-top: 4px;">
                    Nahráte krátke video jedného posunku (5 sekúnd). Z videí naučíme AI rozpoznávať posunky v dlhších videách — rozhovoroch, správach, prednáškach.
                </p>
            </div>
        </div>
    </div>

    <!-- Why we need it -->
    <div class="card" style="margin-bottom: 12px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 32px; line-height: 1;">🎯</span>
            <div>
                <strong style="font-size: 16px;">Prečo to potrebujeme?</strong>
                <p style="color: var(--gray); font-size: 14px; margin-top: 4px;">
                    Máme videá rozhovorov v SPJ, ale AI ich nevie automaticky označiť. Potrebujeme príklady jednotlivých posunkov od rôznych ľudí — potom AI naučíme:
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
                    <div style="background: #EFF6FF; border-radius: 8px; padding: 10px; text-align: center;">
                        <div style="font-size: 24px;">🤖</div>
                        <strong style="font-size: 13px;">AI rozpoznávanie</strong>
                        <p style="font-size: 12px; color: var(--gray);">Automaticky nájde posunky v dlhých videách</p>
                    </div>
                    <div style="background: #EFF6FF; border-radius: 8px; padding: 10px; text-align: center;">
                        <div style="font-size: 24px;">📖</div>
                        <strong style="font-size: 13px;">Korpus SPJ</strong>
                        <p style="font-size: 12px; color: var(--gray);">Prvá veľká databáza SPJ s anotáciami</p>
                    </div>
                    <div style="background: #EFF6FF; border-radius: 8px; padding: 10px; text-align: center;">
                        <div style="font-size: 24px;">🔬</div>
                        <strong style="font-size: 13px;">Výskum SPJ</strong>
                        <p style="font-size: 12px; color: var(--gray);">Dialekty, regionálne rozdiely, gramatika</p>
                    </div>
                    <div style="background: #EFF6FF; border-radius: 8px; padding: 10px; text-align: center;">
                        <div style="font-size: 24px;">🎓</div>
                        <strong style="font-size: 13px;">Slovník a vzdelávanie</strong>
                        <p style="font-size: 12px; color: var(--gray);">Video slovník, učebné materiály</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- How it works -->
    <div class="card" style="margin-bottom: 12px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 32px; line-height: 1;">✋</span>
            <div>
                <strong style="font-size: 16px;">Ako to funguje?</strong>
                <div style="margin-top: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
                        <span style="background: var(--blue); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0;">1</span>
                        <span style="font-size: 14px;">Vyberte si tému (pozdravy, rodina, jedlo...)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
                        <span style="background: var(--blue); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0;">2</span>
                        <span style="font-size: 14px;">Ukáže sa slovo — nahráte posunok (5 sek)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
                        <span style="background: var(--blue); color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0;">3</span>
                        <span style="font-size: 14px;">Skontrolujete a odošlete — hotovo!</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Vision -->
    <div class="card" style="margin-bottom: 12px; background: #111827; color: white;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 32px; line-height: 1;">💡</span>
            <div>
                <strong style="font-size: 16px;">Naša vízia</strong>
                <p style="color: #D1D5DB; font-size: 14px; margin-top: 4px;">
                    Ak AI porozumie nášmu jazyku, môžeme odbúrať bariéry, ktorým čelíme dlhodobo — v komunikácii, vzdelávaní, na úradoch, v zdravotníctve, všade.
                </p>
                <p style="color: #D1D5DB; font-size: 14px; margin-top: 8px;">
                    AI nám pomôže rýchlejšie postaviť most medzi vizuálnym a zvukovým svetom — posunky na text, posunky na hlas, ale aj text a hlas na posunky. Nie ako náhrada tlmočníkov, ale ako ďalší nástroj tam, kde tlmočník nie je k dispozícii.
                </p>
                <p style="color: white; font-size: 14px; margin-top: 8px; font-weight: 600;">
                    Váš príspevok pomáha vybudovať technológiu, ktorá bude slúžiť celej komunite.
                </p>
            </div>
        </div>
    </div>

    <!-- Who needs it -->
    <div class="card" style="margin-bottom: 12px; border: 2px solid var(--green);">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 32px; line-height: 1;">🤝</span>
            <div>
                <strong style="font-size: 16px; color: var(--green);">Kto môže prispieť?</strong>
                <p style="color: var(--gray); font-size: 14px; margin-top: 4px;">
                    <strong>Každý, kto ovláda SPJ</strong> — nepočujúci, nedoslýchaví, CODA, tlmočníci, učitelia.
                    Čím viac ľudí, tým lepší výskum. Každý variant posunku je cenný!
                </p>
            </div>
        </div>
    </div>

</div>
<?php endif; ?>

<?php if ($success): ?>
    <div style="background: #DCFCE7; color: #15803D; padding: 12px 16px; border-radius: 10px; font-weight: 600; margin-bottom: 16px;">
        ✅ <?= htmlspecialchars($success) ?>
    </div>
<?php endif; ?>

<?php if ($error): ?>
    <div class="error-msg"><?= htmlspecialchars($error) ?></div>
<?php endif; ?>

<?php if ($mode === 'login'): ?>
<!-- ── LOGIN FORM ── -->
<div class="card" id="login-form">
    <form method="POST">
        <input type="hidden" name="action" value="login">
        <?= csrf_field() ?>
        <div class="form-group">
            <label for="email">Email</label>
            <input type="email" id="email" name="email" placeholder="vas@email.sk" required
                   value="<?= htmlspecialchars($_POST['email'] ?? '') ?>">
        </div>
        <div class="form-group">
            <label for="password">Heslo</label>
            <input type="password" id="password" name="password" placeholder="••••••••" required>
        </div>
        <button type="submit" class="btn btn-blue" style="margin-bottom: 8px;">Prihlásiť sa</button>
    </form>
    <div style="text-align: center; margin-bottom: 12px;">
        <a href="/forgot-password.php" style="color: var(--gray); font-size: 13px; text-decoration: none;">Zabudnuté heslo?</a>
    </div>
    <div class="divider">alebo</div>
    <?php if (!empty(GOOGLE_CLIENT_ID)): ?>
    <a href="/api/auth.php?action=google" class="btn btn-gray" style="margin-bottom: 12px;">
        <svg width="18" height="18" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 8px;">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
        </svg>
        Prihlásiť sa cez Google
    </a>
    <?php endif; ?>
    <a href="?mode=register" class="btn btn-gray">Vytvoriť účet →</a>
</div>

<?php else: ?>
<!-- ── REGISTER FORM ── -->
<div class="card">
    <h2 style="text-align: center; margin-bottom: 16px;">Registrácia</h2>
    <form method="POST">
        <input type="hidden" name="action" value="register">
        <?= csrf_field() ?>

        <p style="font-size: 13px; color: var(--gray); margin-bottom: 16px;">
            <span style="color: var(--red); font-weight: 700;">*</span> = povinné pole
        </p>

        <div class="form-group">
            <label for="reg-email">Email <span class="req">*</span></label>
            <input type="email" id="reg-email" name="email" placeholder="vas@email.sk" required
                   value="<?= htmlspecialchars($_POST['email'] ?? '') ?>">
        </div>
        <div class="form-group">
            <label for="reg-password">Heslo <span class="req">*</span></label>
            <input type="password" id="reg-password" name="password" placeholder="Aspoň 8 znakov" required minlength="8">
        </div>
        <div class="form-group">
            <label for="reg-password2">Heslo znova <span class="req">*</span></label>
            <input type="password" id="reg-password2" name="password2" placeholder="Zopakujte heslo" required minlength="8">
        </div>
        <script>
        document.getElementById('reg-password2')?.addEventListener('input', function() {
            const pw = document.getElementById('reg-password').value;
            this.setCustomValidity(this.value !== pw ? 'Heslá sa nezhodujú' : '');
        });
        document.getElementById('reg-password')?.addEventListener('input', function() {
            const pw2 = document.getElementById('reg-password2');
            if (pw2.value) pw2.setCustomValidity(pw2.value !== this.value ? 'Heslá sa nezhodujú' : '');
        });
        </script>
        <div class="form-group">
            <label for="display-name">Meno <span class="req">*</span></label>
            <input type="text" id="display-name" name="display_name" placeholder="napr. Janka" required
                   value="<?= htmlspecialchars($_POST['display_name'] ?? '') ?>">
        </div>

        <h3 style="margin: 20px 0 12px; padding-top: 12px; border-top: 1px solid var(--light-gray);">
            O vás (dôležité pre výskum) <span style="color: var(--red); font-weight: 700;">*</span>
        </h3>

        <div class="form-group">
            <label for="school">Škola pre nepočujúcich <span class="req">*</span></label>
            <input type="text" id="school" name="school" placeholder="napr. Kremnica, Bratislava, Lučenec..." required
                   value="<?= htmlspecialchars($_POST['school'] ?? '') ?>">
            <span style="font-size: 12px; color: var(--gray);">Ak ste navštevovali — ovplyvňuje štýl posunkovania a dialekt.</span>
        </div>
        <div class="form-group">
            <label for="location">Mesto / región <span class="req">*</span></label>
            <input type="text" id="location" name="location" placeholder="napr. Bratislava" required
                   value="<?= htmlspecialchars($_POST['location'] ?? '') ?>">
        </div>
        <div class="form-group">
            <label>Vek <span class="req">*</span></label>
            <div class="radio-group">
                <?php foreach (['under_18' => 'do 18', '18-25' => '18–25', '26-35' => '26–35', '36-50' => '36–50', '50+' => '50+'] as $val => $label): ?>
                <label>
                    <input type="radio" name="age_range" value="<?= $val ?>" required
                        <?= ($_POST['age_range'] ?? '') === $val ? 'checked' : '' ?>>
                    <?= $label ?>
                </label>
                <?php endforeach; ?>
            </div>
            <p id="age-warning" style="display:none; background: #FEF3C7; color: #92400E; padding: 8px 12px; border-radius: 8px; font-size: 13px; margin-top: 8px;">
                ⚠️ Ak máte menej ako 16 rokov, potrebujete súhlas rodiča alebo zákonného zástupcu.
            </p>
            <script>
            document.querySelectorAll('input[name="age_range"]').forEach(r => {
                r.addEventListener('change', e => {
                    document.getElementById('age-warning').style.display = e.target.value === 'under_18' ? 'block' : 'none';
                });
            });
            </script>
        </div>
        <div class="form-group">
            <label>Pohlavie <span class="req">*</span></label>
            <div class="radio-group">
                <?php foreach (['woman' => 'Žena', 'man' => 'Muž', 'neutral' => 'Neuvádzam'] as $val => $label): ?>
                <label>
                    <input type="radio" name="gender" value="<?= $val ?>" required
                        <?= ($_POST['gender'] ?? '') === $val ? 'checked' : '' ?>>
                    <?= $label ?>
                </label>
                <?php endforeach; ?>
            </div>
        </div>

        <div class="form-group">
            <label>Posunková ruka <span class="req">*</span></label>
            <div class="radio-group">
                <label>
                    <input type="radio" name="dominant_hand" value="right"
                        <?= ($_POST['dominant_hand'] ?? 'right') === 'right' ? 'checked' : '' ?>>
                    Pravá
                </label>
                <label>
                    <input type="radio" name="dominant_hand" value="left"
                        <?= ($_POST['dominant_hand'] ?? '') === 'left' ? 'checked' : '' ?>>
                    Ľavá
                </label>
            </div>
        </div>

        <h3 style="margin: 20px 0 12px; padding-top: 12px; border-top: 1px solid var(--light-gray);">
            Verejné poďakovanie
        </h3>

        <div class="checkbox-group">
            <input type="checkbox" id="show-public" name="show_public_name"
                <?= isset($_POST['show_public_name']) ? 'checked' : '' ?>
                onchange="document.getElementById('public-name-field').style.display = this.checked ? 'block' : 'none'">
            <label for="show-public">Zobraziť moje meno na stránke Ďakujeme</label>
        </div>
        <div class="form-group" id="public-name-field"
             style="display: <?= isset($_POST['show_public_name']) ? 'block' : 'none' ?>;">
            <label for="public-name">Meno na stránke</label>
            <input type="text" id="public-name" name="public_name" placeholder="napr. Janka M."
                   value="<?= htmlspecialchars($_POST['public_name'] ?? '') ?>">
        </div>

        <div style="margin-top: 16px; padding-top: 12px; border-top: 2px solid var(--light-gray);">
            <h3 style="margin-bottom: 4px; color: var(--dark); font-size: 17px;">Súhlasy <span class="req">*</span></h3>
            <p style="font-size: 13px; color: var(--gray); margin-bottom: 12px;">Zaškrtnite všetky tri políčka.</p>

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
                    📋 Prečítajte si podmienky používania →
                </a>
                <p style="font-size: 12px; color: var(--gray); margin-top: 4px;">Súhlas môžete kedykoľvek odvolať.</p>
            </div>
        </div>

        <button type="submit" class="btn btn-blue" style="margin-top: 8px;">Registrovať sa</button>
    </form>
    <div style="text-align: center; margin-top: 16px;">
        <a href="?mode=login" style="color: var(--blue); font-weight: 600; text-decoration: none;">
            ← Už mám účet
        </a>
    </div>
</div>
<?php endif; ?>

<!-- ── Community Stats ── -->
<div class="card" style="text-align: center;">
    <h3>📊 Komunita</h3>
    <div class="stats-grid" style="margin: 16px 0;">
        <div>
            <span class="stat-num" style="color: var(--blue);"><?= number_format($stats['total_recordings'] ?? 0) ?></span>
            <span class="stat-label">nahrávok</span>
        </div>
        <div>
            <span class="stat-num" style="color: var(--green);"><?= number_format($stats['total_contributors'] ?? 0) ?></span>
            <span class="stat-label">prispievateľov</span>
        </div>
    </div>
    <a href="/thanks.php" class="btn btn-outline" style="font-size: 14px;">🤟 Ďakujeme prispievateľom →</a>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
