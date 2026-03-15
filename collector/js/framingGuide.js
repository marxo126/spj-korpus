/**
 * SPJ Collector — Visual Framing Guide
 * Part A: Static instruction card (first visit only)
 * Part B: SVG overlay on camera (every visit, fades out)
 */

const FramingGuide = {
    _overlayEl: null,
    _cardEl: null,

    /** Check if user has seen the instruction card */
    hasSeenCard() {
        return localStorage.getItem('framing_guide_seen') === '1';
    },

    /** Show the instruction card (full-screen, blocks camera) */
    showInstructionCard(onDismiss) {
        if (this._cardEl) return;

        const card = document.createElement('div');
        card.id = 'framing-instruction-card';
        card.innerHTML = `
            <div class="framing-card-inner">
                <h2 style="text-align:center;margin-bottom:16px;font-size:20px;">Ako správne nahrávať</h2>

                <div style="display:flex;gap:12px;margin-bottom:16px;">
                    <div style="flex:1;text-align:center;">
                        <div style="background:#064e3b;border:2px solid #22c55e;border-radius:10px;padding:14px 8px;margin-bottom:6px;">
                            <svg viewBox="0 0 80 110" width="60" height="82">
                                <ellipse cx="40" cy="22" rx="14" ry="17" fill="none" stroke="#60a5fa" stroke-width="2"/>
                                <line x1="40" y1="39" x2="40" y2="70" stroke="#9ca3af" stroke-width="2"/>
                                <line x1="40" y1="50" x2="20" y2="60" stroke="#9ca3af" stroke-width="2"/>
                                <line x1="40" y1="50" x2="60" y2="60" stroke="#9ca3af" stroke-width="2"/>
                                <text x="20" y="58" fill="#22c55e" font-size="16">&#x1f91a;</text>
                                <text x="52" y="58" fill="#22c55e" font-size="16">&#x270b;</text>
                            </svg>
                        </div>
                        <div style="color:#22c55e;font-size:13px;font-weight:700;">Správne</div>
                        <div style="color:#9ca3af;font-size:11px;">Tvár + ruky viditeľné</div>
                    </div>

                    <div style="flex:1;text-align:center;">
                        <div style="background:#450a0a;border:2px solid #dc2626;border-radius:10px;padding:14px 8px;margin-bottom:6px;">
                            <svg viewBox="0 0 80 110" width="60" height="82">
                                <ellipse cx="40" cy="22" rx="14" ry="17" fill="none" stroke="#60a5fa" stroke-width="2"/>
                                <line x1="40" y1="39" x2="40" y2="70" stroke="#9ca3af" stroke-width="2"/>
                                <text x="40" y="100" text-anchor="middle" fill="#dc2626" font-size="10">ruky mimo</text>
                            </svg>
                        </div>
                        <div style="color:#dc2626;font-size:13px;font-weight:700;">Zle</div>
                        <div style="color:#9ca3af;font-size:11px;">Ruky nie sú viditeľné</div>
                    </div>
                </div>

                <div style="display:flex;gap:12px;margin-bottom:20px;">
                    <div style="flex:1;text-align:center;">
                        <div style="background:#450a0a;border:2px solid #dc2626;border-radius:10px;padding:14px 8px;margin-bottom:6px;height:78px;display:flex;align-items:flex-end;justify-content:center;">
                            <svg viewBox="0 0 80 50" width="60" height="38">
                                <ellipse cx="40" cy="30" rx="14" ry="17" fill="none" stroke="#60a5fa" stroke-width="2"/>
                            </svg>
                        </div>
                        <div style="color:#dc2626;font-size:11px;">Príliš nízko</div>
                    </div>
                    <div style="flex:1;text-align:center;">
                        <div style="background:#450a0a;border:2px solid #dc2626;border-radius:10px;padding:14px 8px;margin-bottom:6px;height:78px;display:flex;align-items:center;justify-content:center;">
                            <svg viewBox="0 0 80 80" width="60" height="60">
                                <ellipse cx="40" cy="40" rx="30" ry="36" fill="none" stroke="#60a5fa" stroke-width="2"/>
                            </svg>
                        </div>
                        <div style="color:#dc2626;font-size:11px;">Príliš blízko</div>
                    </div>
                    <div style="flex:1;text-align:center;">
                        <div style="background:#450a0a;border:2px solid #dc2626;border-radius:10px;padding:14px 8px;margin-bottom:6px;height:78px;display:flex;align-items:center;justify-content:center;">
                            <svg viewBox="0 0 80 80" width="30" height="30">
                                <ellipse cx="40" cy="25" rx="8" ry="10" fill="none" stroke="#60a5fa" stroke-width="2"/>
                                <line x1="40" y1="35" x2="40" y2="55" stroke="#9ca3af" stroke-width="1.5"/>
                            </svg>
                        </div>
                        <div style="color:#dc2626;font-size:11px;">Príliš ďaleko</div>
                    </div>
                </div>

                <div style="background:#1e293b;border-radius:10px;padding:14px;margin-bottom:16px;">
                    <h3 style="color:white;font-size:15px;margin-bottom:8px;">⚡ Auto nahrávanie</h3>
                    <p style="color:#9ca3af;font-size:13px;margin-bottom:8px;">
                        Zapnite prepínač „Auto nahrávanie" nad tlačidlom nahrávania:
                    </p>
                    <div style="display:flex;flex-direction:column;gap:6px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="color:#22c55e;font-size:16px;">✋</span>
                            <span style="color:#d1d5db;font-size:13px;">Ukážte ruky + tvár → rýchle 3-2-1 → nahrávanie začne</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="color:#f59e0b;font-size:16px;">⏱️</span>
                            <span style="color:#d1d5db;font-size:13px;">Nahrávanie trvá max 5 sekúnd alebo stlačte tlačidlo pre zastavenie</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="color:#60a5fa;font-size:16px;">✓</span>
                            <span style="color:#d1d5db;font-size:13px;">Skontrolujte a odošlite → ďalší posunok</span>
                        </div>
                    </div>
                </div>

                <button class="btn btn-blue" id="framing-card-dismiss">Rozumiem, pokračovať</button>
            </div>
        `;

        document.body.appendChild(card);
        this._cardEl = card;

        document.getElementById('framing-card-dismiss').addEventListener('click', () => {
            localStorage.setItem('framing_guide_seen', '1');
            card.remove();
            this._cardEl = null;
            if (onDismiss) onDismiss();
        });
    },

    /** Show SVG overlay on camera container */
    showOverlay(containerEl) {
        if (this._overlayEl) return;

        const duration = this.hasSeenCard() ? 3000 : 5000;

        const overlay = document.createElement('div');
        overlay.id = 'framing-overlay';
        overlay.innerHTML = `
            <svg viewBox="0 0 300 400" preserveAspectRatio="xMidYMid meet">
                <!-- Face zone -->
                <ellipse cx="150" cy="90" rx="45" ry="55" fill="none"
                    stroke="#60a5fa" stroke-width="2.5" stroke-dasharray="6 3"/>
                <text x="150" y="95" text-anchor="middle" fill="#60a5fa"
                    font-size="11" font-family="system-ui, sans-serif">TVÁR</text>

                <!-- Shoulder line -->
                <path d="M70 155 Q150 140 230 155" fill="none"
                    stroke="#60a5fa" stroke-width="1.5" stroke-dasharray="4 3"/>

                <!-- Signing space -->
                <rect x="50" y="160" width="200" height="200" fill="rgba(34,197,94,0.08)"
                    stroke="#22c55e" stroke-width="2" stroke-dasharray="6 3" rx="8"/>
                <text x="150" y="265" text-anchor="middle" fill="#22c55e"
                    font-size="13" font-family="system-ui, sans-serif" font-weight="600">POSUNKOVÝ</text>
                <text x="150" y="283" text-anchor="middle" fill="#22c55e"
                    font-size="13" font-family="system-ui, sans-serif" font-weight="600">PRIESTOR</text>

                <!-- Outer guide border -->
                <rect x="30" y="10" width="240" height="380" fill="none"
                    stroke="#22c55e" stroke-width="2" stroke-dasharray="8 4" rx="8"/>
            </svg>
        `;

        containerEl.appendChild(overlay);
        this._overlayEl = overlay;

        // Fade out after duration
        setTimeout(() => {
            overlay.style.transition = 'opacity 1s ease-out';
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.remove();
                this._overlayEl = null;
            }, 1000);
        }, duration);
    },

    /** Remove overlay immediately (e.g. when recording starts) */
    hideOverlay() {
        if (this._overlayEl) {
            this._overlayEl.remove();
            this._overlayEl = null;
        }
    },

    /** Show instruction card again (from "?" button) */
    showHelp() {
        this.showInstructionCard(null);
    }
};
