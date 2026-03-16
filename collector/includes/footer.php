    </main>

    <?php if ($logged_in): ?>
    <!-- Bottom nav (mobile) -->
    <nav class="bottom-nav mobile-only" aria-label="Mobilná navigácia">
        <a href="/themes.php" class="nav-item <?= $current_page === 'themes' ? 'active' : '' ?>">
            <span class="icon">📚</span>
            <span>Témy</span>
        </a>
        <a href="/record.php" class="nav-item <?= $current_page === 'record' ? 'active' : '' ?>">
            <span class="icon">⏺</span>
            <span>Nahrať</span>
        </a>
        <?php if (is_researcher()): ?>
        <a href="/validate.php" class="nav-item <?= $current_page === 'validate' ? 'active' : '' ?>">
            <span class="icon">✓</span>
            <span>Overiť</span>
        </a>
        <?php endif; ?>
        <a href="/progress.php" class="nav-item <?= $current_page === 'progress' ? 'active' : '' ?>">
            <span class="icon">👤</span>
            <span>Profil</span>
        </a>
    </nav>
    <?php endif; ?>

    <!-- Cookie consent banner -->
    <div id="cookie-banner" style="display:none; position:fixed; bottom:0; left:0; right:0; background:#111827; color:white; padding:16px 20px; z-index:300; box-shadow:0 -2px 10px rgba(0,0,0,0.3);">
        <div style="max-width:600px; margin:0 auto; display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <p style="flex:1; min-width:200px; font-size:14px; line-height:1.5;">
                🍪 Používame cookies na prihlásenie a zapamätanie nastavení. Žiadne reklamné ani sledovacie cookies.
                <a href="/terms.php" style="color:#60A5FA; text-decoration:none;"> Podmienky</a>
            </p>
            <button onclick="acceptCookies()" style="background:var(--green); color:white; border:none; padding:10px 24px; border-radius:8px; font-size:15px; font-weight:700; cursor:pointer; white-space:nowrap;">
                Súhlasím
            </button>
        </div>
    </div>
    <script>
    function acceptCookies() {
        localStorage.setItem('cookie_consent', '1');
        document.getElementById('cookie-banner').style.display = 'none';
    }
    if (!localStorage.getItem('cookie_consent')) {
        document.getElementById('cookie-banner').style.display = 'block';
    }
    </script>

    <!-- JS error reporter -->
    <script>
    (function(){
        function reportError(msg, filename, lineno, colno, stack) {
            try {
                fetch('/api/log-error.php', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({message:msg, level:'error', url:location.href, filename:filename, lineno:lineno, colno:colno, stack:stack})
                }).catch(function(){});
            } catch(e) {}
        }
        // Suppress known harmless errors
        var _ignore = ['play() request was interrupted', 'operation is not supported'];
        window.onerror = function(msg, src, line, col, err) {
            var m = String(msg).toLowerCase();
            for (var i = 0; i < _ignore.length; i++) { if (m.indexOf(_ignore[i]) !== -1) return; }
            reportError(msg, src, line, col, err && err.stack ? err.stack : '');
        };
        window.addEventListener('unhandledrejection', function(e) {
            var msg = e.reason ? (e.reason.message || String(e.reason)) : 'Unhandled promise rejection';
            var stack = e.reason && e.reason.stack ? e.reason.stack : '';
            reportError(msg, '', 0, 0, stack);
        });
    })();
    </script>

    <!-- Legal links -->
    <div style="text-align: center; padding: 12px 16px 0; font-size: 12px;">
        <a href="/terms.php" style="color: #9CA3AF; text-decoration: none; margin: 0 8px;">Podmienky</a>
        <a href="/accessibility-statement.php" style="color: #9CA3AF; text-decoration: none; margin: 0 8px;">Prístupnosť</a>
        <a href="mailto:data@spj.sk" style="color: #9CA3AF; text-decoration: none; margin: 0 8px;">Kontakt</a>
    </div>

    <!-- Partner footer -->
    <footer style="text-align: center; padding: 24px 16px 32px; margin-top: 24px; border-top: 1px solid #E5E7EB;">
        <p style="font-size: 12px; color: #9CA3AF; margin-bottom: 12px;">Spolupráca</p>
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <a href="https://innosign.eu" target="_blank" rel="noopener"><img src="/img/innosign-logo.png" alt="Innosign" style="height: 26px;"></a>
            <a href="https://deafstudio.net" target="_blank" rel="noopener"><img src="/img/deafstudio-logo.png" alt="DeafStudio" style="height: 22px;"></a>
        </div>
        <p style="font-size: 11px; color: #9CA3AF; margin-top: 10px;">
            Technológia a web: Innosign · Výskum: Innosign &amp; DeafStudio · Server: DeafStudio
        </p>
    </footer>
</body>
</html>
