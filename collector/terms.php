<?php
/**
 * SPJ Collector — Terms & Conditions / Podmienky používania
 * GDPR Art. 13 compliant — designed for clarity, visual, deaf-friendly.
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';

$page_title = 'Podmienky používania — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h1 style="text-align: center; margin-bottom: 8px;">Podmienky používania</h1>
<p style="text-align: center; color: var(--gray); margin-bottom: 16px; font-size: 15px;">
    Prečítajte si, čo sa deje s vašimi videami a údajmi.
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
    document.getElementById('terms-simple').style.display = v === 'simple' ? 'block' : 'none';
    document.getElementById('terms-legal').style.display = v === 'legal' ? 'block' : 'none';
    document.getElementById('btn-simple').className = v === 'simple' ? 'btn btn-blue' : 'btn btn-gray';
    document.getElementById('btn-legal').className = v === 'legal' ? 'btn btn-blue' : 'btn btn-gray';
}
</script>

<!-- ══════════════════════════════════════════════ -->
<!-- SIMPLIFIED VERSION — visual, deaf-friendly   -->
<!-- ══════════════════════════════════════════════ -->
<div id="terms-simple">

<!-- ══ KEY SUMMARY ══ -->
<div style="background: #EFF6FF; border: 2px solid #3B82F6; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
    <h2 style="text-align: center; font-size: 18px; margin-bottom: 16px;">Zhrnutie</h2>
    <div style="display: grid; grid-template-columns: 1fr; gap: 12px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">📹</span>
            <div>
                <strong>Vaše videá zostanú vo výskume</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Aj po zmazaní účtu. Videá (s tvárou) sú súčasťou korpusu SPJ.</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">🗑️</span>
            <div>
                <strong>Osobné údaje vymažeme</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Meno, email, profil — všetko zmizne. Videá nebudú prepojené s vami.</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">🔬</span>
            <div>
                <strong>Len na výskum, nie na predaj</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Videá sa nepoužijú na komerčné účely bez vášho súhlasu.</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 28px; line-height: 1;">✉️</span>
            <div>
                <strong>Chcete vymazať aj videá?</strong><br>
                <span style="color: var(--gray); font-size: 14px;">Napíšte na <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a> a vymažeme všetko.</span>
            </div>
        </div>
    </div>
</div>

<!-- ══ DATA CONTROLLER (GDPR Art. 13(1)(a)) ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🏛️</span>
        <h2 style="font-size: 17px; margin: 0;">1. Kto spracúva vaše údaje?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            <strong>Prevádzkovateľ:</strong> SPJ Collector / Výskumný tím SPJ
        </p>
        <p style="margin-top: 8px;">
            <strong>Kontakt na ochranu údajov:</strong> <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a><br>
            <strong>Kontakt na výskum:</strong> <a href="mailto:vyskum@spj.sk" style="color: var(--blue);">vyskum@spj.sk</a>
        </p>
    </div>
</div>

<!-- ══ PURPOSE (GDPR Art. 13(1)(c)) ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🎯</span>
        <h2 style="font-size: 17px; margin: 0;">2. Prečo zbierame videá?</h2>
    </div>
    <p style="font-size: 15px; line-height: 1.7;">
        Vytvárame <strong>korpus slovenského posunkového jazyka</strong> — veľkú zbierku videí posunkov.
        Korpus pomáha výskumníkom študovať SPJ a vyvíjať technológie na rozpoznávanie posunkov.
    </p>
</div>

<!-- ══ LEGAL BASIS (GDPR Art. 13(1)(c)) ══ -->
<div class="card" style="margin-bottom: 16px; border: 2px solid #3B82F6;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">⚖️</span>
        <h2 style="font-size: 17px; margin: 0;">3. Na akom právnom základe spracúvame údaje?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <strong>Účet a služba:</strong> Výslovný súhlas (GDPR čl. 6 ods. 1 písm. a)
            <br><span style="color: var(--gray); font-size: 13px;">Vaše osobné údaje (email, meno, demografické údaje) spracúvame na základe vášho súhlasu.</span>
        </div>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <strong>Video s tvárou (biometrické údaje):</strong> Výslovný súhlas (GDPR čl. 9 ods. 2 písm. a)
            <br><span style="color: var(--gray); font-size: 13px;">Video obsahuje vašu tvár — to sú biometrické údaje, ktoré vyžadujú váš výslovný súhlas.</span>
        </div>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px;">
            <strong>Uchovanie dát na výskum:</strong> Oprávnený záujem pre vedecký výskum (GDPR čl. 6 ods. 1 písm. f + čl. 89)
            <br><span style="color: var(--gray); font-size: 13px;">Anonymizované videá a pohybové dáta zostávajú v korpuse na vedecký výskum aj po zmazaní účtu.</span>
        </div>
    </div>
</div>

<!-- ══ THREE SEPARATE CONSENTS ══ -->
<div class="card" style="margin-bottom: 16px; border: 2px solid #F59E0B;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">📋</span>
        <h2 style="font-size: 17px; margin: 0;">4. Tri oddelené súhlasy</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p style="margin-bottom: 12px;">Pri registrácii vás požiadame o <strong>tri oddelené súhlasy</strong>. Každý sa týka iného spracovania vašich údajov:</p>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
            <strong>1. Účet a služba</strong>
            <br><span style="color: var(--gray); font-size: 13px;">Spracovanie emailu, mena a demografických údajov na vytvorenie účtu a používanie služby.</span>
        </div>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
            <strong>2. Nahrávanie videa s tvárou (biometrické údaje)</strong>
            <br><span style="color: var(--gray); font-size: 13px;">Videá obsahujú vašu tvár — ide o biometrické údaje podľa GDPR. Tento súhlas môžete kedykoľvek odvolať.</span>
        </div>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px;">
            <strong>3. Uchovanie dát na výskum</strong>
            <br><span style="color: var(--gray); font-size: 13px;">Anonymizované videá a pohybové dáta zostanú v korpuse SPJ aj po zmazaní účtu, kým budú potrebné na výskum.</span>
        </div>
    </div>
</div>

<!-- ══ WHAT DATA WE COLLECT ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">👤</span>
        <h2 style="font-size: 17px; margin: 0;">5. Aké údaje zbierame?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px;">
            <strong>Povinné:</strong> email, heslo (šifrované), meno, škola pre nepočujúcich, mesto, vek, pohlavie, posunková ruka
            <br><span style="color: var(--gray); font-size: 13px;">Tieto údaje sú dôležité pre výskum — napr. škola pre nepočujúcich silne ovplyvňuje štýl posunkovania a dialekt.</span>
        </div>
    </div>
</div>

<!-- ══ WHAT HAPPENS WITH VIDEOS ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">📹</span>
        <h2 style="font-size: 17px; margin: 0;">6. Čo sa deje s mojimi videami?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Video obsahuje vašu <strong>tvár a ruky</strong> — to je potrebné na výskum.</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Video bude uložené na serveroch projektu.</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Video môžu vidieť <strong>výskumníci</strong> pracujúci na korpuse SPJ.</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
            <span style="color: #EF4444; font-size: 18px;">✗</span>
            <span>Video sa <strong>nebude predávať</strong> ani používať na reklamu.</span>
        </div>
    </div>
</div>

<!-- ══ DATA RETENTION (GDPR Art. 13(2)(a)) ══ -->
<div class="card" style="margin-bottom: 16px; border: 2px solid #F59E0B;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">⏳</span>
        <h2 style="font-size: 17px; margin: 0;">7. Ako dlho uchovávame údaje?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
            <strong>Osobné údaje (email, meno, profil):</strong> Do zmazania účtu.
            <br><span style="color: var(--gray); font-size: 13px;">Po zmazaní účtu sa osobné údaje okamžite vymažú.</span>
        </div>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px;">
            <strong>Videá a pohybové dáta:</strong> Po dobu trvania výskumného projektu.
            <br><span style="color: var(--gray); font-size: 13px;">Videá sa uchovávajú anonymizovane (bez prepojenia s vami) kým sú potrebné na vedecký výskum SPJ.</span>
        </div>
    </div>
</div>

<!-- ══ ACCOUNT DELETION ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🗑️</span>
        <h2 style="font-size: 17px; margin: 0;">8. Čo sa stane, keď zmažem účet?</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; background: #DCFCE7; border-radius: 8px; padding: 12px;">
                <strong style="color: #15803D;">Vymazané:</strong>
                <ul style="margin: 8px 0 0; padding-left: 16px; color: #15803D;">
                    <li>Email</li>
                    <li>Meno a profil</li>
                    <li>Štatistiky</li>
                    <li>Prepojenie videí s vami</li>
                </ul>
            </div>
            <div style="flex: 1; min-width: 200px; background: #FEF3C7; border-radius: 8px; padding: 12px;">
                <strong style="color: #92400E;">Zostáva v korpuse:</strong>
                <ul style="margin: 8px 0 0; padding-left: 16px; color: #92400E;">
                    <li>Videá (s tvárou)</li>
                    <li>Kým sú potrebné na výskum</li>
                    <li>Anonymne — bez vášho mena</li>
                </ul>
            </div>
        </div>
        <p style="margin-top: 12px; font-size: 14px; color: var(--gray);">
            Chcete vymazať <strong>úplne všetko vrátane videí</strong>?
            Napíšte na <a href="mailto:data@spj.sk" style="color: var(--blue); font-weight: 600;">data@spj.sk</a>
        </p>

        <!-- Visual: what happens when videos are deleted on request -->
        <div style="margin-top: 20px; padding-top: 16px; border-top: 2px solid var(--light-gray);">
            <p style="text-align: center; font-weight: 700; font-size: 15px; margin-bottom: 16px;">
                Čo sa stane, keď požiadate o vymazanie videí?
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;">
                <!-- VIDEO = deleted -->
                <div style="flex: 1; min-width: 160px; max-width: 220px; text-align: center; background: #FEE2E2; border-radius: 12px; padding: 16px;">
                    <div style="font-size: 14px; font-weight: 700; color: #DC2626; margin-bottom: 12px;">VYMAZANÉ</div>
                    <svg width="100" height="120" viewBox="0 0 100 120" style="margin: 0 auto;">
                        <!-- Face circle -->
                        <circle cx="50" cy="25" r="18" fill="#FCA5A5" stroke="#DC2626" stroke-width="2"/>
                        <!-- Eyes -->
                        <circle cx="43" cy="22" r="2.5" fill="#DC2626"/>
                        <circle cx="57" cy="22" r="2.5" fill="#DC2626"/>
                        <!-- Mouth -->
                        <path d="M42 30 Q50 36 58 30" stroke="#DC2626" stroke-width="2" fill="none"/>
                        <!-- Body -->
                        <line x1="50" y1="43" x2="50" y2="78" stroke="#FCA5A5" stroke-width="3"/>
                        <!-- Arms -->
                        <line x1="50" y1="55" x2="25" y2="48" stroke="#FCA5A5" stroke-width="3"/>
                        <line x1="50" y1="55" x2="75" y2="48" stroke="#FCA5A5" stroke-width="3"/>
                        <!-- Legs -->
                        <line x1="50" y1="78" x2="35" y2="110" stroke="#FCA5A5" stroke-width="3"/>
                        <line x1="50" y1="78" x2="65" y2="110" stroke="#FCA5A5" stroke-width="3"/>
                        <!-- X cross overlay -->
                        <line x1="15" y1="10" x2="85" y2="110" stroke="#DC2626" stroke-width="4" stroke-linecap="round"/>
                        <line x1="85" y1="10" x2="15" y2="110" stroke="#DC2626" stroke-width="4" stroke-linecap="round"/>
                    </svg>
                    <div style="font-size: 13px; color: #DC2626; margin-top: 8px; font-weight: 600;">
                        Video s tvárou
                    </div>
                </div>

                <!-- Arrow -->
                <div style="display: flex; align-items: center; font-size: 28px; padding: 0 4px;">→</div>

                <!-- POSE = kept -->
                <div style="flex: 1; min-width: 160px; max-width: 220px; text-align: center; background: #DCFCE7; border-radius: 12px; padding: 16px;">
                    <div style="font-size: 14px; font-weight: 700; color: #15803D; margin-bottom: 12px;">ZOSTÁVA</div>
                    <svg width="100" height="120" viewBox="0 0 100 120" style="margin: 0 auto;">
                        <!-- Head outline (dashed = no photo) -->
                        <ellipse cx="50" cy="24" rx="16" ry="19" fill="none" stroke="#22C55E" stroke-width="1.5" stroke-dasharray="3,2"/>
                        <!-- Face landmarks as dots -->
                        <!-- Eyebrows -->
                        <circle cx="40" cy="15" r="1.2" fill="#22C55E"/><circle cx="43" cy="14" r="1.2" fill="#22C55E"/><circle cx="46" cy="14.5" r="1.2" fill="#22C55E"/>
                        <circle cx="54" cy="14.5" r="1.2" fill="#22C55E"/><circle cx="57" cy="14" r="1.2" fill="#22C55E"/><circle cx="60" cy="15" r="1.2" fill="#22C55E"/>
                        <!-- Eyes -->
                        <circle cx="43" cy="19" r="1.5" fill="#22C55E"/><circle cx="45" cy="19" r="1" fill="#22C55E"/><circle cx="41" cy="19" r="1" fill="#22C55E"/>
                        <circle cx="57" cy="19" r="1.5" fill="#22C55E"/><circle cx="55" cy="19" r="1" fill="#22C55E"/><circle cx="59" cy="19" r="1" fill="#22C55E"/>
                        <!-- Nose -->
                        <circle cx="50" cy="23" r="1.2" fill="#22C55E"/><circle cx="48" cy="25" r="1" fill="#22C55E"/><circle cx="52" cy="25" r="1" fill="#22C55E"/>
                        <!-- Mouth -->
                        <circle cx="45" cy="30" r="1" fill="#22C55E"/><circle cx="48" cy="31" r="1" fill="#22C55E"/><circle cx="50" cy="31.5" r="1.2" fill="#22C55E"/><circle cx="52" cy="31" r="1" fill="#22C55E"/><circle cx="55" cy="30" r="1" fill="#22C55E"/>
                        <circle cx="47" cy="33" r="1" fill="#22C55E"/><circle cx="50" cy="33.5" r="1" fill="#22C55E"/><circle cx="53" cy="33" r="1" fill="#22C55E"/>
                        <!-- Body -->
                        <line x1="50" y1="43" x2="50" y2="78" stroke="#22C55E" stroke-width="2.5"/>
                        <!-- Shoulders -->
                        <circle cx="38" cy="48" r="2.5" fill="#22C55E"/>
                        <circle cx="62" cy="48" r="2.5" fill="#22C55E"/>
                        <line x1="38" y1="48" x2="62" y2="48" stroke="#22C55E" stroke-width="2"/>
                        <!-- Arms raised (signing pose) -->
                        <line x1="38" y1="48" x2="22" y2="40" stroke="#22C55E" stroke-width="2.5"/>
                        <line x1="22" y1="40" x2="15" y2="28" stroke="#22C55E" stroke-width="2.5"/>
                        <line x1="62" y1="48" x2="78" y2="40" stroke="#22C55E" stroke-width="2.5"/>
                        <line x1="78" y1="40" x2="85" y2="28" stroke="#22C55E" stroke-width="2.5"/>
                        <!-- Elbow dots -->
                        <circle cx="22" cy="40" r="2.5" fill="#22C55E"/>
                        <circle cx="78" cy="40" r="2.5" fill="#22C55E"/>
                        <!-- Hands (multiple finger dots) -->
                        <circle cx="15" cy="28" r="2" fill="#22C55E"/>
                        <circle cx="12" cy="26" r="1.2" fill="#22C55E"/><circle cx="14" cy="24" r="1.2" fill="#22C55E"/><circle cx="16" cy="24" r="1.2" fill="#22C55E"/><circle cx="18" cy="25" r="1.2" fill="#22C55E"/><circle cx="13" cy="29" r="1.2" fill="#22C55E"/>
                        <circle cx="85" cy="28" r="2" fill="#22C55E"/>
                        <circle cx="82" cy="26" r="1.2" fill="#22C55E"/><circle cx="84" cy="24" r="1.2" fill="#22C55E"/><circle cx="86" cy="24" r="1.2" fill="#22C55E"/><circle cx="88" cy="25" r="1.2" fill="#22C55E"/><circle cx="87" cy="29" r="1.2" fill="#22C55E"/>
                        <!-- Hips -->
                        <circle cx="50" cy="78" r="2.5" fill="#22C55E"/>
                        <!-- Legs -->
                        <line x1="50" y1="78" x2="38" y2="110" stroke="#22C55E" stroke-width="2.5"/>
                        <line x1="50" y1="78" x2="62" y2="110" stroke="#22C55E" stroke-width="2.5"/>
                        <!-- Knee dots -->
                        <circle cx="44" cy="94" r="2.5" fill="#22C55E"/>
                        <circle cx="56" cy="94" r="2.5" fill="#22C55E"/>
                    </svg>
                    <div style="font-size: 13px; color: #15803D; margin-top: 8px; font-weight: 600;">
                        176 bodov — ruky, tvár, ústa, telo
                    </div>
                </div>
            </div>
            <p style="text-align: center; font-size: 13px; color: var(--gray); margin-top: 12px;">
                Z videa sa extrahuje <strong>176 bodov</strong> — poloha rúk, prstov, tváre, úst, očí a tela.
                Sú to iba súradnice bodov (čísla), nie obrázok. Z bodov sa nedá obnoviť vaša podoba ani identita.
            </p>
        </div>
    </div>
</div>

<!-- ══ RIGHT TO WITHDRAW (GDPR Art. 7(3)) ══ -->
<div class="card" style="margin-bottom: 16px; border: 2px solid #22C55E;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">↩️</span>
        <h2 style="font-size: 17px; margin: 0;">9. Odvolanie súhlasu</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            <strong>Súhlas môžete kedykoľvek odvolať</strong> — bez udania dôvodu a bez negatívnych následkov.
        </p>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-top: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
                <span style="font-size: 18px;">🔑</span>
                <span><strong>Zmazať účet:</strong> Profil → Vymazať účet</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
                <span style="font-size: 18px;">📧</span>
                <span><strong>Vymazať aj videá:</strong> <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a></span>
            </div>
        </div>
        <p style="margin-top: 8px; font-size: 13px; color: var(--gray);">
            Odvolanie súhlasu nemá vplyv na zákonnosť spracovania pred jeho odvolaním.
        </p>
    </div>
</div>

<!-- ══ YOUR RIGHTS (GDPR Art. 13(2)(b)) ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🛡️</span>
        <h2 style="font-size: 17px; margin: 0;">10. Vaše práva podľa GDPR</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="font-size: 18px;">📋</span>
            <span><strong>Právo na prístup</strong> — môžete požiadať o kópiu vašich údajov</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="font-size: 18px;">✏️</span>
            <span><strong>Právo na opravu</strong> — môžete opraviť nesprávne údaje</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="font-size: 18px;">🗑️</span>
            <span><strong>Právo na vymazanie</strong> — môžete požiadať o vymazanie údajov</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="font-size: 18px;">↩️</span>
            <span><strong>Právo na odvolanie súhlasu</strong> — kedykoľvek, bez udania dôvodu</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="font-size: 18px;">📦</span>
            <span><strong>Právo na prenosnosť</strong> — môžete získať údaje v strojovo čitateľnom formáte</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
            <span style="font-size: 18px;">🚫</span>
            <span><strong>Právo namietať</strong> — môžete namietať proti spracovaniu</span>
        </div>
        <p style="margin-top: 12px; font-size: 14px; color: var(--gray);">
            Na uplatnenie práv napíšte na <a href="mailto:data@spj.sk" style="color: var(--blue);">data@spj.sk</a>. Odpovieme do 30 dní.
        </p>
    </div>
</div>

<!-- ══ UNDER 16 (Slovak GDPR age) ══ -->
<div class="card" style="margin-bottom: 16px; border: 2px solid #F59E0B;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">👶</span>
        <h2 style="font-size: 17px; margin: 0;">11. Osoby mladšie ako 16 rokov</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            Podľa slovenského zákona o ochrane osobných údajov je na spracovanie údajov osôb <strong>mladších ako 16 rokov</strong>
            potrebný <strong>súhlas rodiča alebo zákonného zástupcu</strong>.
        </p>
        <p style="margin-top: 8px; color: var(--gray); font-size: 14px;">
            Ak máte menej ako 16 rokov a chcete prispieť, požiadajte rodiča alebo zákonného zástupcu, aby vám pomohol s registráciou.
        </p>
    </div>
</div>

<!-- ══ SECURITY ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🔒</span>
        <h2 style="font-size: 17px; margin: 0;">12. Bezpečnosť</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Heslo je šifrované — nikto ho nevidí</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--light-gray);">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Prenos dát je zabezpečený (HTTPS)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
            <span style="color: #22C55E; font-size: 18px;">✓</span>
            <span>Videá sú prístupné len prihláseným výskumníkom</span>
        </div>
    </div>
</div>

<!-- ══ RIGHT TO COMPLAIN (GDPR Art. 13(2)(d)) ══ -->
<div class="card" style="margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 32px;">🏢</span>
        <h2 style="font-size: 17px; margin: 0;">13. Právo podať sťažnosť</h2>
    </div>
    <div style="font-size: 15px; line-height: 1.7;">
        <p>
            Ak nie ste spokojní s tým, ako spracúvame vaše údaje, máte právo podať sťažnosť na dozorný orgán:
        </p>
        <div style="background: var(--dark-card); border-radius: 8px; padding: 12px; margin-top: 8px;">
            <strong>Úrad na ochranu osobných údajov Slovenskej republiky</strong><br>
            <a href="https://dataprotection.gov.sk" target="_blank" style="color: var(--blue);">https://dataprotection.gov.sk</a>
        </div>
    </div>
</div>

<!-- ══ CONTACT ══ -->
<div class="card" style="text-align: center; margin-bottom: 16px;">
    <h2 style="font-size: 17px; margin-bottom: 12px;">Máte otázky?</h2>
    <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;">
        <div>
            <div style="font-size: 28px; margin-bottom: 4px;">📊</div>
            <strong>Údaje a GDPR</strong><br>
            <a href="mailto:data@spj.sk" style="color: var(--blue); font-size: 15px;">data@spj.sk</a>
        </div>
        <div>
            <div style="font-size: 28px; margin-bottom: 4px;">🔬</div>
            <strong>Výskum</strong><br>
            <a href="mailto:vyskum@spj.sk" style="color: var(--blue); font-size: 15px;">vyskum@spj.sk</a>
        </div>
    </div>
</div>

</div><!-- /terms-simple -->

<!-- ══════════════════════════════════════════════ -->
<!-- LEGAL VERSION — formal GDPR language          -->
<!-- ══════════════════════════════════════════════ -->
<div id="terms-legal" style="display: none;">

<div class="card" style="margin-bottom: 16px; font-size: 14px; line-height: 1.8;">

<h2 style="font-size: 17px; margin-bottom: 16px;">Informácie o spracúvaní osobných údajov podľa čl. 13 a 14 nariadenia GDPR</h2>

<h3 style="margin: 20px 0 8px; color: var(--dark);">1. Prevádzkovateľ</h3>
<p>SPJ Collector / Výskumný tím SPJ<br>
Kontakt na zodpovednú osobu: <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a></p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">2. Účel a právny základ spracúvania</h3>
<table style="width:100%; border-collapse:collapse; margin:8px 0; font-size:13px;">
<tr style="border-bottom:2px solid var(--light-gray); text-align:left;">
    <th style="padding:8px;">Účel</th>
    <th style="padding:8px;">Právny základ</th>
    <th style="padding:8px;">Kategórie údajov</th>
</tr>
<tr style="border-bottom:1px solid var(--light-gray);">
    <td style="padding:8px;">Vytvorenie a správa používateľského účtu</td>
    <td style="padding:8px;">Čl. 6 ods. 1 písm. a) GDPR — súhlas</td>
    <td style="padding:8px;">Email, heslo (hash), meno, demografické údaje</td>
</tr>
<tr style="border-bottom:1px solid var(--light-gray);">
    <td style="padding:8px;">Nahrávanie videí obsahujúcich biometrické údaje (tvár)</td>
    <td style="padding:8px;">Čl. 9 ods. 2 písm. a) GDPR — výslovný súhlas so spracúvaním osobitnej kategórie údajov</td>
    <td style="padding:8px;">Videozáznamy obsahujúce tvár, ruky a telo dotknutej osoby</td>
</tr>
<tr style="border-bottom:1px solid var(--light-gray);">
    <td style="padding:8px;">Uchovanie anonymizovaných dát na vedecký výskum</td>
    <td style="padding:8px;">Čl. 6 ods. 1 písm. f) GDPR — oprávnený záujem v spojení s čl. 89 ods. 1 GDPR (záruky pre vedecký výskum)</td>
    <td style="padding:8px;">Anonymizované videá, pohybové dáta (176 bodov pose estimation)</td>
</tr>
<tr>
    <td style="padding:8px;">Lingvistický výskum variability SPJ</td>
    <td style="padding:8px;">Čl. 6 ods. 1 písm. a) GDPR — súhlas</td>
    <td style="padding:8px;">Škola pre nepočujúcich, región, vek, pohlavie, dominantná ruka</td>
</tr>
</table>

<h3 style="margin: 20px 0 8px; color: var(--dark);">3. Kategórie spracúvaných osobných údajov</h3>
<p><strong>Bežné osobné údaje:</strong> email, meno, škola pre nepočujúcich, mesto/región, veková kategória, pohlavie, dominantná ruka, heslový hash (bcrypt).</p>
<p><strong>Osobitná kategória údajov (čl. 9 GDPR):</strong> videozáznamy obsahujúce biometrické údaje (tvár dotknutej osoby). Spracúvanie prebieha na základe výslovného súhlasu dotknutej osoby podľa čl. 9 ods. 2 písm. a) GDPR.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">4. Príjemcovia osobných údajov</h3>
<p>Osobné údaje sú sprístupnené výhradne členom výskumného tímu SPJ. Videozáznamy môžu byť sprístupnené ďalším výskumníkom v rámci výskumného korpusu SPJ na základe dohody o spracúvaní údajov. Údaje sa neprenášajú do tretích krajín.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">5. Doba uchovávania</h3>
<p><strong>Údaje viazané na účet</strong> (email, meno, profil, demografické údaje): uchovávané po dobu existencie účtu. Po zmazaní účtu sa okamžite a nenávratne vymažú.</p>
<p><strong>Videozáznamy a pohybové dáta:</strong> po zmazaní účtu sa anonymizujú (odstránenie prepojenia s identitou dotknutej osoby) a uchovávajú sa po dobu trvania výskumného projektu korpusu SPJ. Na žiadosť dotknutej osoby sa vymažú aj videozáznamy (kontakt: <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a>). Pohybové dáta (pose estimation, 176 súradnicových bodov) sa uchovávajú aj po vymazaní videí, nakoľko z nich nie je možné rekonštruovať podobu ani identitu dotknutej osoby.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">6. Súhlas a jeho odvolanie (čl. 7 GDPR)</h3>
<p>Dotknutá osoba udeľuje pri registrácii tri oddelené súhlasy:</p>
<ol style="padding-left:20px; margin:8px 0;">
    <li>Súhlas so spracúvaním osobných údajov na vytvorenie účtu a poskytovanie služby</li>
    <li>Výslovný súhlas s nahrávaním a spracúvaním videozáznamov obsahujúcich biometrické údaje (tvár)</li>
    <li>Súhlas s uchovávaním anonymizovaných videí a pohybových dát v korpuse SPJ po zmazaní účtu</li>
</ol>
<p>Každý súhlas je možné kedykoľvek odvolať bez udania dôvodu a bez negatívnych následkov. Odvolanie súhlasu nemá vplyv na zákonnosť spracúvania vykonaného pred jeho odvolaním. Súhlas je možné odvolať zmazaním účtu v sekcii Profil alebo emailom na <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a>.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">7. Práva dotknutej osoby (čl. 15–22 GDPR)</h3>
<p>Dotknutá osoba má právo:</p>
<ul style="padding-left:20px; margin:8px 0;">
    <li><strong>na prístup</strong> k osobným údajom (čl. 15)</li>
    <li><strong>na opravu</strong> nesprávnych údajov (čl. 16)</li>
    <li><strong>na vymazanie</strong> („právo na zabudnutie", čl. 17)</li>
    <li><strong>na obmedzenie spracúvania</strong> (čl. 18)</li>
    <li><strong>na prenosnosť údajov</strong> (čl. 20)</li>
    <li><strong>namietať</strong> proti spracúvaniu (čl. 21)</li>
    <li><strong>odvolať súhlas</strong> kedykoľvek (čl. 7 ods. 3)</li>
</ul>
<p>Na uplatnenie práv kontaktujte: <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a>. Odpovieme do 30 dní od doručenia žiadosti.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">8. Osoby mladšie ako 16 rokov</h3>
<p>Podľa § 15 zákona č. 18/2018 Z. z. o ochrane osobných údajov je na spracúvanie osobných údajov osôb mladších ako 16 rokov potrebný súhlas zákonného zástupcu.</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">9. Bezpečnostné opatrenia</h3>
<p>Prevádzkovateľ prijal primerané technické a organizačné opatrenia na ochranu osobných údajov:</p>
<ul style="padding-left:20px; margin:8px 0;">
    <li>Heslá sú uchovávané výhradne vo forme kryptografického hashu (bcrypt)</li>
    <li>Prenos dát je zabezpečený šifrovaním (TLS/HTTPS)</li>
    <li>Prístup k videozáznamom je obmedzený na autentifikovaných používateľov a výskumníkov</li>
    <li>Ochrana proti CSRF útokom, obmedzenie počtu pokusov o prihlásenie</li>
</ul>

<h3 style="margin: 20px 0 8px; color: var(--dark);">10. Právo podať sťažnosť</h3>
<p>Dotknutá osoba má právo podať sťažnosť dozornému orgánu:</p>
<p style="margin-top:8px;">
    <strong>Úrad na ochranu osobných údajov Slovenskej republiky</strong><br>
    Hraničná 12, 820 07 Bratislava 27<br>
    <a href="https://dataprotection.gov.sk" target="_blank" style="color:var(--blue)">https://dataprotection.gov.sk</a>
</p>

<h3 style="margin: 20px 0 8px; color: var(--dark);">11. Kontaktné údaje</h3>
<p>
    <strong>Ochrana osobných údajov:</strong> <a href="mailto:data@spj.sk" style="color:var(--blue)">data@spj.sk</a><br>
    <strong>Výskum:</strong> <a href="mailto:vyskum@spj.sk" style="color:var(--blue)">vyskum@spj.sk</a>
</p>

</div><!-- /card -->

</div><!-- /terms-legal -->

<div style="text-align: center; margin-top: 16px;">
    <a href="/index.php?mode=register" class="btn btn-blue">← Späť na registráciu</a>
</div>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
