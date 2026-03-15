<?php
/**
 * SPJ Collector — Authentication helpers
 */

require_once __DIR__ . '/db.php';

function start_session(): void {
    if (session_status() === PHP_SESSION_NONE) {
        session_start();
    }
}

function csrf_token(): string {
    start_session();
    if (empty($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

function csrf_field(): string {
    return '<input type="hidden" name="csrf_token" value="' . htmlspecialchars(csrf_token()) . '">';
}

function verify_csrf(): bool {
    start_session();
    $token = $_POST['csrf_token'] ?? $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '';
    return !empty($token) && hash_equals($_SESSION['csrf_token'] ?? '', $token);
}

function require_csrf(): void {
    if (!verify_csrf()) {
        http_response_code(403);
        if (str_contains($_SERVER['HTTP_ACCEPT'] ?? '', 'application/json')) {
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Neplatný CSRF token']);
        } else {
            header('Location: /index.php?error=csrf');
        }
        exit;
    }
}

function is_logged_in(): bool {
    start_session();
    return isset($_SESSION['user_id']);
}

function require_login(): void {
    if (!is_logged_in()) {
        header('Location: /index.php');
        exit;
    }
    // Allow consent.php itself without redirect loop
    $current = basename($_SERVER['SCRIPT_NAME']);
    if ($current !== 'consent.php') {
        $user = get_user();
        if ($user && (!$user['consent_service'] || !$user['consent_biometric'] || !$user['consent_retention'])) {
            header('Location: /consent.php');
            exit;
        }
    }
}

function get_user_id(): ?int {
    start_session();
    return $_SESSION['user_id'] ?? null;
}

function get_user(): ?array {
    $id = get_user_id();
    if (!$id) return null;
    $pdo = get_db();
    $stmt = $pdo->prepare('SELECT * FROM users WHERE id = ?');
    $stmt->execute([$id]);
    return $stmt->fetch() ?: null;
}

function login_user(int $user_id): void {
    start_session();
    session_regenerate_id(true);
    $_SESSION['user_id'] = $user_id;

    // Update last_active
    $pdo = get_db();
    $stmt = $pdo->prepare('UPDATE users SET last_active = CURDATE() WHERE id = ?');
    $stmt->execute([$user_id]);
}

function logout_user(): void {
    start_session();
    $_SESSION = [];
    session_destroy();
}

function register_user(string $email, string $password, array $demographics): ?int {
    $pdo = get_db();

    // Check if email exists
    $stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
    $stmt->execute([$email]);
    if ($stmt->fetch()) return null; // email taken

    $hash = password_hash($password, PASSWORD_BCRYPT);

    $stmt = $pdo->prepare('
        INSERT INTO users (email, password_hash, display_name, public_name, show_public_name,
                          school, location, age_range, gender, dominant_hand,
                          consent_service, consent_service_date,
                          consent_biometric, consent_biometric_date,
                          consent_retention, consent_retention_date,
                          consent_date, last_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, IF(?, NOW(), NULL), ?, IF(?, NOW(), NULL), ?, IF(?, NOW(), NULL), NOW(), CURDATE())
    ');
    $stmt->execute([
        $email,
        $hash,
        $demographics['display_name'] ?? '',
        $demographics['public_name'] ?? '',
        $demographics['show_public_name'] ?? 0,
        $demographics['school'] ?? '',
        $demographics['location'] ?? '',
        $demographics['age_range'] ?? null,
        $demographics['gender'] ?? null,
        $demographics['dominant_hand'] ?? 'right',
        $demographics['consent_service'] ?? 0,
        $demographics['consent_service'] ?? 0,    // for IF()
        $demographics['consent_biometric'] ?? 0,
        $demographics['consent_biometric'] ?? 0,   // for IF()
        $demographics['consent_retention'] ?? 0,
        $demographics['consent_retention'] ?? 0,    // for IF()
    ]);

    return (int) $pdo->lastInsertId();
}

function verify_login(string $email, string $password): ?int {
    $pdo = get_db();
    $stmt = $pdo->prepare('SELECT id, password_hash FROM users WHERE email = ?');
    $stmt->execute([$email]);
    $user = $stmt->fetch();

    if ($user && password_verify($password, $user['password_hash'])) {
        return (int) $user['id'];
    }
    return null;
}

function google_login_or_register(string $google_id, string $email, string $name): ?int {
    $pdo = get_db();

    // Check if Google user exists
    $stmt = $pdo->prepare('SELECT id FROM users WHERE google_id = ?');
    $stmt->execute([$google_id]);
    $user = $stmt->fetch();
    if ($user) return (int) $user['id'];

    // Check if email exists (link Google to existing account)
    $stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
    $stmt->execute([$email]);
    $user = $stmt->fetch();
    if ($user) {
        $pdo->prepare('UPDATE users SET google_id = ? WHERE id = ?')->execute([$google_id, $user['id']]);
        return (int) $user['id'];
    }

    // Create new user
    $stmt = $pdo->prepare('
        INSERT INTO users (email, password_hash, google_id, display_name, last_active)
        VALUES (?, ?, ?, ?, CURDATE())
    ');
    $stmt->execute([$email, '!google-oauth', $google_id, $name]);
    return (int) $pdo->lastInsertId();
}

function delete_user_data(int $user_id): void {
    $pdo = get_db();

    // Anonymize recordings — keep videos for research, remove user link
    // Recordings.user_id → NULL (ON DELETE SET NULL handles this)
    // Validations.validator_id → NULL (ON DELETE SET NULL handles this)

    // Delete user_theme_progress (personal tracking data)
    $pdo->prepare('DELETE FROM user_theme_progress WHERE user_id = ?')->execute([$user_id]);

    // Delete user — CASCADE sets recordings.user_id and validations.validator_id to NULL
    // Videos and recording counters are preserved for the research corpus
    $pdo->prepare('DELETE FROM users WHERE id = ?')->execute([$user_id]);
}

/**
 * Session-based rate limiting.
 * LIMITATION: This can be bypassed by clearing cookies / using incognito.
 * For stronger protection, implement DB-backed rate limiting keyed on IP
 * (requires a rate_limits table with ip, action, attempt_count, window_start).
 * Current approach is acceptable for low-risk actions (password reset, login).
 */
function check_rate_limit(string $action, int $max_attempts, int $window_seconds = 900): bool {
    start_session();
    $key = "rate_limit_{$action}";
    $now = time();
    $attempts = $_SESSION[$key] ?? [];
    // Prune expired attempts
    $attempts = array_filter($attempts, fn($t) => $t > $now - $window_seconds);
    if (count($attempts) >= $max_attempts) {
        return false; // rate limited
    }
    $attempts[] = $now;
    $_SESSION[$key] = $attempts;
    return true;
}

function request_password_reset(string $email): ?string {
    $pdo = get_db();
    $stmt = $pdo->prepare('SELECT id FROM users WHERE email = ? AND password_hash != "!google-oauth"');
    $stmt->execute([$email]);
    $user = $stmt->fetch();
    if (!$user) return null; // email not found or Google-only account

    $token = bin2hex(random_bytes(32));
    $hash = hash('sha256', $token);
    $expires = date('Y-m-d H:i:s', time() + 3600); // 1 hour

    $stmt = $pdo->prepare('UPDATE users SET password_reset_token = ?, password_reset_expires = ? WHERE id = ?');
    $stmt->execute([$hash, $expires, $user['id']]);

    return $token;
}

function validate_reset_token(string $token): ?int {
    $hash = hash('sha256', $token);
    $pdo = get_db();
    $stmt = $pdo->prepare('SELECT id FROM users WHERE password_reset_token = ? AND password_reset_expires > NOW()');
    $stmt->execute([$hash]);
    $user = $stmt->fetch();
    return $user ? (int) $user['id'] : null;
}

function reset_password(string $token, string $new_password): bool {
    $user_id = validate_reset_token($token);
    if (!$user_id) return false;

    $hash = password_hash($new_password, PASSWORD_BCRYPT);
    $pdo = get_db();
    $stmt = $pdo->prepare('UPDATE users SET password_hash = ?, password_reset_token = NULL, password_reset_expires = NULL WHERE id = ?');
    $stmt->execute([$hash, $user_id]);
    return true;
}

function send_reset_email(string $email, string $token): bool {
    $reset_url = SITE_URL . '/reset-password.php?token=' . urlencode($token);

    // Always log for dev/debug
    error_log("Password reset link for {$email}: {$reset_url}");

    $subject = SITE_NAME . ' — Obnovenie hesla';
    $body = "Dobrý deň,\n\n";
    $body .= "Požiadali ste o obnovenie hesla na " . SITE_NAME . ".\n\n";
    $body .= "Kliknite na tento odkaz pre nastavenie nového hesla:\n";
    $body .= $reset_url . "\n\n";
    $body .= "Odkaz je platný 1 hodinu.\n\n";
    $body .= "Ak ste o obnovenie nepožiadali, tento email ignorujte.\n";

    $headers = "From: noreply@" . parse_url(SITE_URL, PHP_URL_HOST) . "\r\n";
    $headers .= "Content-Type: text/plain; charset=UTF-8\r\n";

    return @mail($email, $subject, $body, $headers);
}

function get_community_stats(): array {
    $pdo = get_db();
    $stats = $pdo->query('
        SELECT
            (SELECT COUNT(*) FROM recordings) as total_recordings,
            (SELECT COUNT(DISTINCT user_id) FROM recordings) as total_contributors,
            (SELECT COUNT(DISTINCT sign_id) FROM recordings WHERE status = "approved") as total_signs
    ')->fetch();
    return $stats;
}
