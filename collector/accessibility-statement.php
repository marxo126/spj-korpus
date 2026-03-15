<?php
/**
 * SPJ Collector — Accessibility Statement (EU Web Accessibility Directive)
 */
require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
$page_title = 'Vyhlásenie o prístupnosti — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1>Vyhlásenie o prístupnosti</h1>

<div class="card">
    <h2 style="font-size: 18px; margin-bottom: 12px;">Záväzok</h2>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6;">
        Prevádzkovateľ aplikácie <?= htmlspecialchars(SITE_NAME) ?> sa zaväzuje sprístupniť túto webovú aplikáciu
        v súlade so smernicou Európskeho parlamentu a Rady (EÚ) 2016/2102 o prístupnosti webových sídiel
        a mobilných aplikácií subjektov verejného sektora a zákonom č. 95/2019 Z. z.
    </p>
</div>

<div class="card">
    <h2 style="font-size: 18px; margin-bottom: 12px;">Stav súladu</h2>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6;">
        Táto aplikácia je <strong>čiastočne v súlade</strong> so štandardom WCAG 2.1 úrovne AA.
    </p>
    <h3 style="font-size: 15px; margin-top: 16px; margin-bottom: 8px;">Známe nedostatky</h3>
    <ul style="font-size: 14px; color: var(--gray); line-height: 1.8; padding-left: 20px;">
        <li>Ovládanie prehrávača videa klávesnicou je obmedzené</li>
        <li>Niektoré farebné kontrasty nemusia spĺňať minimálny pomer 4.5:1</li>
        <li>Nahrávacie rozhranie vyžaduje prístup ku kamere, čo obmedzuje niektoré asistenčné technológie</li>
    </ul>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6; margin-top: 12px;">
        Na odstraňovaní týchto nedostatkov aktívne pracujeme.
    </p>
</div>

<div class="card">
    <h2 style="font-size: 18px; margin-bottom: 12px;">Spätná väzba a kontakt</h2>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6;">
        Ak máte problémy s prístupnosťou tejto aplikácie alebo chcete nahlásiť nedostatok,
        kontaktujte nás na:
    </p>
    <p style="margin-top: 8px;">
        <a href="mailto:data@spj.sk" style="color: var(--blue); font-weight: 600;">data@spj.sk</a>
    </p>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6; margin-top: 8px;">
        Na váš podnet odpovieme do 30 dní.
    </p>
</div>

<div class="card">
    <h2 style="font-size: 18px; margin-bottom: 12px;">Vynucovacie konanie</h2>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6;">
        Ak nie ste spokojný/á s našou odpoveďou, môžete sa obrátiť na:
    </p>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6; margin-top: 8px;">
        <strong>Ministerstvo investícií, regionálneho rozvoja a informatizácie SR</strong><br>
        Sekcia digitálnej agendy<br>
        <a href="https://mirri.gov.sk" target="_blank" rel="noopener" style="color: var(--blue);">mirri.gov.sk</a>
    </p>
</div>

<div class="card">
    <h2 style="font-size: 18px; margin-bottom: 12px;">Dátum</h2>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6;">
        Toto vyhlásenie bolo pripravené dňa <strong>13. marca 2026</strong>.
    </p>
    <p style="font-size: 14px; color: var(--gray); line-height: 1.6; margin-top: 4px;">
        Vyhlásenie bude aktualizované pri významných zmenách aplikácie, minimálne raz ročne.
    </p>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
