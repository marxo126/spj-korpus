<?php
/**
 * SPJ Collector — Admin authentication helper
 */

require_once __DIR__ . '/auth.php';

function is_admin(): bool {
    $user = get_user();
    return $user && !empty($user['is_admin']);
}

function require_admin(): void {
    if (!is_logged_in()) {
        header('Location: /index.php');
        exit;
    }
    if (!is_admin()) {
        http_response_code(403);
        echo '<!DOCTYPE html><html><body><h1>403 — Prístup zamietnutý</h1><p><a href="/">Späť</a></p></body></html>';
        exit;
    }
}

function is_researcher(): bool {
    $user = get_user();
    return $user && (!empty($user['is_admin']) || !empty($user['is_researcher']));
}

function require_researcher(): void {
    if (!is_logged_in()) {
        header('Location: /index.php');
        exit;
    }
    if (!is_researcher()) {
        http_response_code(403);
        echo '<!DOCTYPE html><html><body><h1>403 — Prístup zamietnutý</h1><p><a href="/">Späť</a></p></body></html>';
        exit;
    }
}
