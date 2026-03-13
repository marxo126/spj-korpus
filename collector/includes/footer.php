    </main>

    <?php if ($logged_in): ?>
    <!-- Bottom nav (mobile) -->
    <nav class="bottom-nav mobile-only">
        <a href="/themes.php" class="nav-item <?= $current_page === 'themes' ? 'active' : '' ?>">
            <span class="icon">📚</span>
            <span>Témy</span>
        </a>
        <a href="/record.php" class="nav-item <?= $current_page === 'record' ? 'active' : '' ?>">
            <span class="icon">⏺</span>
            <span>Nahrať</span>
        </a>
        <a href="/validate.php" class="nav-item <?= $current_page === 'validate' ? 'active' : '' ?>">
            <span class="icon">✓</span>
            <span>Overiť</span>
        </a>
        <a href="/progress.php" class="nav-item <?= $current_page === 'progress' ? 'active' : '' ?>">
            <span class="icon">👤</span>
            <span>Profil</span>
        </a>
    </nav>
    <?php endif; ?>
</body>
</html>
