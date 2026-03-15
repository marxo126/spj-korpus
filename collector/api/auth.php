<?php
/**
 * SPJ Collector — Auth API
 * Handles logout + Google OAuth callback
 */

require_once __DIR__ . '/../includes/config.php';
require_once __DIR__ . '/../includes/auth.php';

$action = $_GET['action'] ?? $_POST['action'] ?? '';

switch ($action) {
    case 'logout':
        logout_user();
        header('Location: /index.php');
        exit;

    case 'google':
        // Google OAuth callback
        if (empty(GOOGLE_CLIENT_ID)) {
            header('Location: /index.php?error=google_not_configured');
            exit;
        }
        $code = $_GET['code'] ?? '';
        if (!$code) {
            // Redirect to Google consent screen
            $params = http_build_query([
                'client_id' => GOOGLE_CLIENT_ID,
                'redirect_uri' => GOOGLE_REDIRECT_URI,
                'response_type' => 'code',
                'scope' => 'email profile',
                'access_type' => 'online',
            ]);
            header('Location: https://accounts.google.com/o/oauth2/v2/auth?' . $params);
            exit;
        }

        // Exchange code for token
        $token_data = json_decode(file_get_contents('https://oauth2.googleapis.com/token', false,
            stream_context_create(['http' => [
                'method' => 'POST',
                'header' => 'Content-Type: application/x-www-form-urlencoded',
                'content' => http_build_query([
                    'code' => $code,
                    'client_id' => GOOGLE_CLIENT_ID,
                    'client_secret' => GOOGLE_CLIENT_SECRET,
                    'redirect_uri' => GOOGLE_REDIRECT_URI,
                    'grant_type' => 'authorization_code',
                ]),
            ]])
        ), true);

        if (!$token_data || !isset($token_data['access_token'])) {
            header('Location: /index.php?error=google_failed');
            exit;
        }

        // Get user info
        $user_info = json_decode(file_get_contents(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            false,
            stream_context_create(['http' => [
                'header' => 'Authorization: Bearer ' . $token_data['access_token'],
            ]])
        ), true);

        if (!$user_info || !isset($user_info['email'])) {
            header('Location: /index.php?error=google_failed');
            exit;
        }

        // Find or create user
        $user_id = google_login_or_register($user_info['id'], $user_info['email'], $user_info['name'] ?? '');
        if ($user_id) {
            login_user($user_id);
            $u = get_user();
            if ($u && $u['consent_service'] && $u['consent_biometric'] && $u['consent_retention']) {
                header('Location: /themes.php');
            } else {
                header('Location: /consent.php');
            }
        } else {
            header('Location: /index.php?error=google_failed');
        }
        exit;

    case 'delete_account':
        // GDPR: delete all user data (POST only + CSRF)
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
            header('Location: /progress.php');
            exit;
        }
        if (!is_logged_in()) {
            header('Location: /index.php');
            exit;
        }
        require_csrf();
        delete_user_data(get_user_id());
        logout_user();
        header('Location: /index.php?deleted=1');
        exit;

    default:
        header('Location: /index.php');
        exit;
}
