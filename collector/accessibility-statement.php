<?php
/**
 * SPJ Collector — Accessibility Statement (EU Web Accessibility Directive)
 * Dual version: simplified (visual, deaf-friendly) + legal (formal)
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

$page_title = 'Vyhlásenie o prístupnosti — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1 style="text-align: center; margin-bottom: 8px;">Vyhlásenie o prístupnosti</h1>
<p style="text-align: center; color: var(--gray); margin-bottom: 16px; font-size: 15px;">
    Ako sa staráme o to, aby bola aplikácia prístupná pre všetkých.
</p>

<!-- Version toggle -->
<div style="display: flex; justify-content: center; gap: 8px; margin-bottom: 24px;">
    <button id="btn-simple" onclick="showVersion('simple')" class="btn btn-blue" style="padding: 10px 20px; font-size: 15px;">
        Jednoduchá verzia
    </button>
    <button id="btn-legal" onclick="showVersion('legal')" class="btn btn-gray" style="padding: 10px 20px; font-size: 15px;">
        Právna verzia
    </button>
</div>

<script>
function showVersion(v) {
    document.getElementById('acc-simple').style.display = v === 'simple' ? 'block' : 'none';
    document.getElementById('acc-legal').style.display = v === 'legal' ? 'block' : 'none';
    document.getElementById('btn-simple').className = v === 'simple' ? 'btn btn-blue' : 'btn btn-gray';
    document.getElementById('btn-legal').className = v === 'legal' ? 'btn btn-blue' : 'btn btn-gray';
}
</script>

<!-- ══════════════════════════════════════════════ -->
<!-- SIMPLIFIED VERSION — visual, deaf-friendly   -->
<!-- ══════════════════════════════════════════════ -->
<div id="acc-simple">

<!-- ══ KEY SUMMARY ══ -->
<div style="background: #EFF6FF; border: 2px solid #3B82F6; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
    <h2 style="text-align: center; font-size: 18px; margin-bottom: 16px;">Zhrnutie</h2>
    <div style="display: grid; grid-template-columns: 1fr; gap: 12px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">♿</span>
            <div>
                <strong>Snažíme sa byť prístupní pre všetkých</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Aplikáciu navrhujeme tak, aby ju mohli používať aj nepočujúci, slabozrakí a ľudia s obmedzenou pohyblivosťou.</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">🔧</span>
            <div>
                <strong>Stále pracujeme na vylepšeniach</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Niektoré časti ešte nie sú úplne prístupné — poznáme ich a opravujeme.</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">✉️</span>
            <div>
                <strong>Máte problém? Napíšte nám</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Na <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a> — odpovieme do 30 dní.</span>
            </div>
        </div>
    </div>
</div>

<!-- ══ WHAT WE DO ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">✅</span>
        <h2 style="font-size: 17px; margin: 0;">1. Čo robíme pre prístupnosť</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Veľké a čitateľné písmo</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Priblíženie stránky na mobile (pinch-to-zoom)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Tmavý režim pre citlivé oči</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Stránky majú orientačné body pre čítačky obrazovky</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Tlačidlo „Preskoč na obsah" pre ovládanie klávesnicou</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Znížené animácie pre ľudí citlivých na pohyb</span>
        </div>
    </div>
</div>

<!-- ══ KNOWN ISSUES ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">⚠️</span>
        <h2 style="font-size: 17px; margin: 0;">2. Na čom ešte pracujeme</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #F59E0B; font-size: 18px;">⏳</span>
            <span>Ovládanie prehrávača videa klávesnicou</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #F59E0B; font-size: 18px;">⏳</span>
            <span>Video návod v posunkovom jazyku</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
            <span style="color: #F59E0B; font-size: 18px;">⏳</span>
            <span>Nahrávacie rozhranie vyžaduje kameru — niektoré asistenčné technológie môžu byť obmedzené</span>
        </div>
    </div>
</div>

<!-- ══ STANDARD ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">📏</span>
        <h2 style="font-size: 17px; margin: 0;">3. Aký štandard dodržiavame</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            Snažíme sa dodržiavať medzinárodný štandard <strong>WCAG 2.1 úrovne AA</strong>.
            To je ako „pravidlá cestnej premávky" pre webové stránky — zabezpečujú, že weby sú použiteľné pre čo najviac ľudí.
        </p>
        <p style="margin-top: 8px; color: var(--gray); font-size: 14px;">
            Aktuálny stav: <strong>čiastočne v súlade</strong> — väčšina požiadaviek splnená, na zvyšku pracujeme.
        </p>
    </div>
</div>

<!-- ══ CONTACT ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">📩</span>
        <h2 style="font-size: 17px; margin: 0;">4. Kontakt a sťažnosti</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            Ak niečo nefunguje alebo máte nápad na zlepšenie prístupnosti, napíšte nám:
        </p>
        <p style="margin-top: 8px;">
            <a href="mailto:data@spj.sk" style="color: var(--blue); font-weight: 600; font-size: 18px;">data@spj.sk</a>
        </p>
        <p style="margin-top: 12px; color: var(--gray); font-size: 14px;">
            Odpovieme do 30 dní. Ak nebudete spokojní s našou odpoveďou, môžete sa obrátiť na:
        </p>
        <div style="background: var(--card-bg); border-radius: 8px; padding: 12px; margin-top: 8px; border: 1px solid var(--light-gray);">
            <strong>Ministerstvo investícií, regionálneho rozvoja a informatizácie SR</strong><br>
            <span style="color: var(--gray); font-size: 14px;">Sekcia digitálnej agendy</span><br>
            <a href="https://mirri.gov.sk" target="_blank" rel="noopener" style="color: var(--blue);">mirri.gov.sk</a>
        </div>
    </div>
</div>

<!-- ══ DATE ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">📅</span>
        <h2 style="font-size: 17px; margin: 0;">5. Dátum</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>Toto vyhlásenie bolo pripravené <strong>15. marca 2026</strong>.</p>
        <p style="color: var(--gray); font-size: 14px; margin-top: 4px;">Aktualizujeme ho pri významných zmenách, minimálne raz ročne.</p>
    </div>
</div>

</div><!-- /acc-simple -->

<!-- ══════════════════════════════════════════════ -->
<!-- LEGAL VERSION — formal language               -->
<!-- ══════════════════════════════════════════════ -->
<div id="acc-legal" style="display: none;">

<div class="card" style="margin-bottom: 16px; font-size: 14px; line-height: 1.8;">

<h2 style="font-size: 17px; margin-bottom: 16px;">Vyhlásenie o prístupnosti podľa smernice (EÚ) 2016/2102 a zákona č. 95/2019 Z. z.</h2>

<h3 style="margin: 20px 0 8px; color: var(--dark);">1. Prevádzkovateľ</h3>
<p>SPJ Collector / Výskumný tím SPJ<br>
Kontakt: <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a></p>
<p>Webová aplikácia: <a href="https://zber.spj.sk" style="color:var(--blue)">https://zber.spj.sk</a></p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">2. Stav súladu</h3>
<p>Táto webová aplikácia je <strong>čiastočne v súlade</strong> s požiadavkami zákona č. 95/2019 Z. z. o informačných technológiách vo verejnej správe a technickým štandardom WCAG 2.1 úrovne AA (<a href="https://www.w3.org/TR/WCAG21/" target="_blank" rel="noopener" style="color:var(--blue)">Web Content Accessibility Guidelines 2.1</a>).</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">3. Neprístupný obsah</h3>
<p>Nasledujúci obsah nie je v plnom súlade so štandardom WCAG 2.1 AA:</p>

<table style="width:100%; border-collapse:collapse; margin:8px 0; font-size:13px;">
<tr style="border-bottom:2px solid var(--light-gray); text-align:left;">
    <th style="padding:8px;">Nedostatok</th>
    <th style="padding:8px;">Kritérium WCAG</th>
    <th style="padding:8px;">Stav</th>
</tr>
<tr style="border-bottom:1px solid var(--light-gray);">
    <td style="padding:8px;">Ovládanie natívneho prehrávača videa klávesnicou je obmedzené</td>
    <td style="padding:8px;">2.1.1 Klávesnica (A)</td>
    <td style="padding:8px;">V riešení</td>
</tr>
<tr style="border-bottom:1px solid var(--light-gray);">
    <td style="padding:8px;">Nahrávacie rozhranie vyžaduje prístup ku kamere, čo obmedzuje niektoré asistenčné technológie</td>
    <td style="padding:8px;">4.1.2 Názov, úloha, hodnota (A)</td>
    <td style="padding:8px;">Neodstrániteľné obmedzenie</td>
</tr>
<tr>
    <td style="padding:8px;">Chýba video návod v slovenskom posunkovom jazyku</td>
    <td style="padding:8px;">3.1.5 Úroveň čítania (AAA)</td>
    <td style="padding:8px;">Plánované</td>
</tr>
</table>

<h3 style="margin: 20px 0 8px; color: var(--dark);">4. Splnené požiadavky</h3>
<ul style="padding-left:20px; margin:8px 0;">
    <li>Možnosť priblíženia obsahu na 200 % a viac (WCAG 1.4.4)</li>
    <li>Dostatočný farebný kontrast textu (WCAG 1.4.3)</li>
    <li>Navigačné orientačné body a ARIA atribúty pre čítačky obrazovky (WCAG 1.3.1, 4.1.2)</li>
    <li>Odkaz „Preskoč na obsah" (WCAG 2.4.1)</li>
    <li>Viditeľné zvýraznenie zamerania (focus) pre ovládanie klávesnicou (WCAG 2.4.7)</li>
    <li>Podpora zníženého pohybu cez <code>prefers-reduced-motion</code> (WCAG 2.3.3)</li>
    <li>Podpora tmavého režimu cez <code>prefers-color-scheme</code></li>
    <li>Jazyk stránky je označený (<code>lang="sk"</code>) (WCAG 3.1.1)</li>
</ul>

<h3 style="margin: 20px 0 8px; color: var(--dark);">5. Spätná väzba a kontakt</h3>
<p>Ak narazíte na problém s prístupnosťou tejto aplikácie, kontaktujte nás:</p>
<p style="margin-top:8px;">
    <strong>Email:</strong> <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a><br>
    <strong>Lehota na odpoveď:</strong> 30 dní od doručenia podnetu
</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">6. Vynucovacie konanie</h3>
<p>Ak nie ste spokojný/á s vybavením vášho podnetu, môžete sa obrátiť na orgán zodpovedný za presadzovanie požiadaviek na prístupnosť webových sídiel:</p>
<p style="margin-top:8px;">
    <strong>Ministerstvo investícií, regionálneho rozvoja a informatizácie SR</strong><br>
    Sekcia digitálnej agendy<br>
    <a href="https://mirri.gov.sk" target="_blank" rel="noopener" style="color:var(--blue)">https://mirri.gov.sk</a>
</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">7. Dátum vyhlásenia</h3>
<p>Toto vyhlásenie bolo vypracované dňa <strong>15. marca 2026</strong> na základe vlastného hodnotenia prevádzkovateľa.</p>
<p style="margin-top:4px;">Posledná aktualizácia: <strong>15. marca 2026</strong>.</p>
<p style="margin-top:4px; color: var(--gray);">Vyhlásenie bude preskúmané a aktualizované pri významných zmenách aplikácie, minimálne raz ročne.</p>

</div><!-- /card -->

</div><!-- /acc-legal -->

<?php require_once __DIR__ . '/includes/footer.php'; ?>
