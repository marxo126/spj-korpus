<?php
/**
 * SPJ Collector — Lightweight error logging to DB
 * Captures PHP errors/exceptions + provides API for JS errors.
 */

require_once __DIR__ . '/db.php';

function log_error(string $message, string $level = 'error', string $source = 'php', ?array $extra = null): void {
    try {
        $pdo = get_db();
        $stmt = $pdo->prepare('
            INSERT INTO error_log (level, source, message, url, user_id, user_agent, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ');
        $stmt->execute([
            $level,
            $source,
            mb_substr($message, 0, 5000),
            mb_substr($_SERVER['REQUEST_URI'] ?? '', 0, 500),
            $_SESSION['user_id'] ?? null,
            mb_substr($_SERVER['HTTP_USER_AGENT'] ?? '', 0, 500),
            $extra ? json_encode($extra, JSON_UNESCAPED_UNICODE) : null,
        ]);
    } catch (\Throwable $e) {
        // Fallback to PHP error log if DB fails
        error_log("SPJ error_logger failed: " . $e->getMessage() . " | Original: $message");
    }
}

// Register PHP error handler
set_error_handler(function (int $errno, string $errstr, string $errfile, string $errline) {
    // Skip suppressed errors (@)
    if (!(error_reporting() & $errno)) return false;

    $level_map = [
        E_ERROR => 'error', E_WARNING => 'warning', E_NOTICE => 'info',
        E_USER_ERROR => 'error', E_USER_WARNING => 'warning', E_USER_NOTICE => 'info',
    ];
    $level = $level_map[$errno] ?? 'error';
    log_error("$errstr in $errfile:$errline", $level, 'php', ['errno' => $errno]);
    return false; // Continue to default handler
});

// Register uncaught exception handler
set_exception_handler(function (\Throwable $e) {
    log_error(
        $e->getMessage() . ' in ' . $e->getFile() . ':' . $e->getLine(),
        'error',
        'php',
        ['trace' => mb_substr($e->getTraceAsString(), 0, 2000)]
    );
    // Re-throw for default handling
    throw $e;
});

// Register shutdown handler for fatal errors
register_shutdown_function(function () {
    $error = error_get_last();
    if ($error && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR])) {
        log_error(
            $error['message'] . ' in ' . $error['file'] . ':' . $error['line'],
            'error',
            'php',
            ['type' => $error['type']]
        );
    }
});
