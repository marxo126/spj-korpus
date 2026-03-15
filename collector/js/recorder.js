/**
 * SPJ Collector — Camera + MediaRecorder + Upload
 * Forces 720p, front camera, auto codec detection.
 */

let cameraStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let currentSign = null;
let todayCount = parseInt(document.getElementById('progress-count')?.textContent) || 0;
let totalCount = parseInt(document.getElementById('total-count')?.textContent) || 0;
let qualityChecker = null;
let isRecording = false;
let autoRecordEnabled = false;
let autoRecordActive = false;   // true while auto-countdown is running
let autoReadyFrames = 0;        // consecutive frames with hands+face detected
let recordTimerInterval = null;  // elapsed timer
let recordAutoStop = null;       // 5s auto-stop timeout
let reviewingRecording = false;  // true while preview/submit/variant screen is showing
let recordingStartTime = 0;      // timestamp when recording started
let recordedDurationMs = 0;      // duration of last recording in ms
const AUTO_READY_THRESHOLD = 5;  // need 5 consecutive good frames (~1 sec) before auto-start

function toggleAutoRecord(enabled) {
    autoRecordEnabled = enabled;
    localStorage.setItem('spj_auto_record', enabled ? '1' : '0');
    if (!enabled) cancelAutoRecord();
}

function cancelAutoRecord() {
    autoRecordActive = false;
    autoReadyFrames = 0;
    const label = document.getElementById('record-label');
    if (label) label.dataset.autoCountdown = '';
}

function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content || '';
}

// ── Init on page load ──
document.addEventListener('DOMContentLoaded', async () => {
    const perm = await navigator.permissions?.query({ name: 'camera' }).catch(() => null);
    if (perm && perm.state === 'prompt') {
        const loading = document.getElementById('camera-loading');
        loading.innerHTML =
            '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:white;padding:20px;text-align:center;">' +
            '<p style="font-size:40px;margin-bottom:12px;">📷</p>' +
            '<p style="font-size:16px;font-weight:600;margin-bottom:8px;">Potrebujeme prístup ku kamere</p>' +
            '<p style="font-size:14px;color:#9CA3AF;margin-bottom:16px;">Na nahrávanie posunkov používame prednú kameru. Video sa uloží až po vašom potvrdení.</p>' +
            '<button class="btn btn-blue" onclick="initCamera()" style="width:auto;padding:10px 24px;">Povoliť kameru</button>' +
            '</div>';
        await loadNextSign();
        return;
    }

    await initCamera();
    await loadNextSign();
    qualityChecker = new QualityChecker(
        document.getElementById('camera-preview'),
        updateQualityStatus
    );
    qualityChecker.start();
    restoreAutoRecord();

    // Cleanup camera and quality checker on page unload
    window.addEventListener('beforeunload', () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
        }
        if (qualityChecker) {
            qualityChecker.stop();
        }
        // Revoke any pending blob URL to prevent memory leaks
        const previewEl = document.getElementById('video-preview');
        if (previewEl && previewEl.src && previewEl.src.startsWith('blob:')) {
            URL.revokeObjectURL(previewEl.src);
        }
    });
});

function restoreAutoRecord() {
    const savedAuto = localStorage.getItem('spj_auto_record');
    if (savedAuto === '1') {
        autoRecordEnabled = true;
        const toggle = document.getElementById('auto-record-toggle');
        if (toggle) toggle.checked = true;
    }
}

// ── Camera: always 720p, always front, no user choice ──
async function initCamera() {
    try {
        // Let camera give native resolution — CSS handles framing
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false
        });

        cameraStream = stream;

        const preview = document.getElementById('camera-preview');
        preview.srcObject = cameraStream;
        await preview.play();

        document.getElementById('camera-loading').style.display = 'none';
        document.getElementById('camera-container').style.display = 'block';

        // Show framing guide
        if (!FramingGuide.hasSeenCard()) {
            FramingGuide.showInstructionCard(() => {
                FramingGuide.showOverlay(document.getElementById('camera-container'));
            });
        } else {
            FramingGuide.showOverlay(document.getElementById('camera-container'));
        }

        // Start quality checker if not yet running
        if (!qualityChecker) {
            qualityChecker = new QualityChecker(preview, updateQualityStatus);
            qualityChecker.start();
            restoreAutoRecord();
        }

    } catch (err) {
        document.getElementById('camera-loading').innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:white;padding:20px;text-align:center;">' +
            '<p style="font-size:16px;">⚠️ Nemôžem spustiť kameru.<br>Povoľte prístup ku kamere v nastaveniach prehliadača.</p></div>';
    }
}

// ── Load next sign to record ──
async function loadNextSign() {
    try {
        const params = new URLSearchParams();
        const themeEl = document.getElementById('current-theme-id');
        const themeSetEl = document.getElementById('theme-id-set');
        const signEl = document.getElementById('initial-sign-id');

        if (themeEl && themeSetEl && themeSetEl.value === '1') {
            params.set('theme_id', themeEl.value);
        }
        if (signEl && signEl.value !== '0') {
            params.set('sign_id', signEl.value);
            signEl.value = '0';
        }

        const qs = params.toString();
        const res = await fetch('/api/next_sign.php' + (qs ? '?' + qs : ''));
        const sign = await res.json();

        if (sign.error) {
            const isThemeDone = sign.error === 'all_done' && themeEl && themeSetEl && themeSetEl.value === '1';
            document.getElementById('word-display').textContent = '🎉 Hotovo!';
            document.getElementById('help-links').innerHTML = isThemeDone
                ? 'Všetky posunky v tejto téme sú nahrané. <a href="/themes.php">Späť na témy</a>'
                : 'Všetky posunky sú nahrané.';
            document.getElementById('record-btn').classList.add('disabled');
            return;
        }

        currentSign = sign;
        document.getElementById('word-display').textContent = sign.word_sk.toUpperCase();
        document.getElementById('help-links').innerHTML = 'Nepoznáte posunok? → ' + buildRefLinks(sign);

    } catch (err) {
        console.error('Failed to load next sign:', err);
    }
}

// ── Quality status callback ──
function updateQualityStatus(status) {
    // During recording: don't update UI, just wait for manual stop or 5s timer
    if (isRecording) return;

    const container = document.getElementById('camera-container');
    const banner = document.getElementById('camera-banner');
    const btnEl = document.getElementById('record-btn');
    const badgeHands = document.getElementById('badge-hands');
    const badgeFace = document.getElementById('badge-face');
    const badgeLight = document.getElementById('badge-light');

    // Hands
    if (status.hands) {
        badgeHands.className = 'status-badge status-ok';
        badgeHands.textContent = '✅ Ruky';
    } else {
        badgeHands.className = 'status-badge status-err';
        badgeHands.textContent = '❌ Ruky';
    }

    // Face
    if (status.face) {
        badgeFace.className = 'status-badge status-ok';
        badgeFace.textContent = '✅ Tvár';
    } else if (status.faceWarning) {
        badgeFace.className = 'status-badge status-warn';
        badgeFace.textContent = '⚠️ ' + status.faceWarning;
    } else {
        badgeFace.className = 'status-badge status-err';
        badgeFace.textContent = '❌ Tvár';
    }

    // Light
    if (status.light === 'ok') {
        badgeLight.className = 'status-badge status-ok';
        badgeLight.textContent = '✅ Svetlo';
    } else {
        badgeLight.className = 'status-badge status-warn';
        badgeLight.textContent = '⚠️ ' + (status.light === 'dark' ? 'Tmavo' : 'Svetlo');
    }

    // Overall status
    const blocked = !status.hands || !status.face;
    container.className = 'camera-container ' + (blocked ? 'border-red' : 'border-green');
    btnEl.className = 'record-btn' + (blocked ? ' disabled' : '');

    // Banner
    if (blocked) {
        banner.className = 'camera-banner error';
        banner.textContent = !status.hands ? '🔴 Ukážte ruky pred kameru' : '🔴 Tvár nie je viditeľná';
    } else if (status.light !== 'ok' || status.faceWarning || status.handsEdge || !status.contrast) {
        banner.className = 'camera-banner warn';
        const warnings = [];
        if (status.light === 'dark') warnings.push('Príliš tmavo');
        if (status.light === 'bright') warnings.push('Príliš svetlo');
        if (!status.contrast) warnings.push('Nízky kontrast');
        if (status.faceWarning) warnings.push(status.faceWarning);
        if (status.handsEdge) warnings.push('Ruky na okraji');
        banner.textContent = '⚠️ ' + warnings.join(' · ');
    } else {
        banner.className = 'camera-banner ok';
        banner.textContent = '✅ Pripravené na nahrávanie';
    }

    // Label
    const label = document.getElementById('record-label');
    if (!label.dataset.autoCountdown) {
        label.textContent = blocked ? 'Ukážte ruky a tvár' : (autoRecordEnabled ? 'Ukážte ruky pre auto nahrávanie' : 'Stlačte pre nahrávanie');
    }

    // Auto-record: require stable detection before starting
    if (autoRecordEnabled && !isRecording && !reviewingRecording && currentSign) {
        const ready = status.hands && status.face;

        if (ready) {
            autoReadyFrames++;
        } else {
            autoReadyFrames = 0;
            // Only cancel countdown if detection lost for multiple frames (not single flicker)
        }

        // Start countdown after stable detection
        if (!autoRecordActive && autoReadyFrames >= AUTO_READY_THRESHOLD) {
            autoRecordActive = true;
            label.dataset.autoCountdown = '1';
            autoStartRecording();
        }
    }
}

// ── Auto-start: fast 3-2-1 (~1 second total) then record ──
async function autoStartRecording() {
    const overlay = document.getElementById('countdown-overlay');
    const numEl = document.getElementById('countdown-num');
    overlay.style.display = 'flex';

    for (let i = 3; i >= 1; i--) {
        numEl.textContent = i;
        numEl.style.animation = 'none';
        void numEl.offsetHeight;
        numEl.style.animation = 'countdown-pulse 0.3s ease-in-out';
        await sleep(333);
    }

    overlay.style.display = 'none';
    cancelAutoRecord();

    // Only start if still valid
    if (!isRecording && !reviewingRecording && autoRecordEnabled && currentSign) {
        beginRecording();
    }
}

// ── Manual start: 3-2-1 countdown (1 sec each) then record ──
async function startRecording() {
    if (!currentSign || isRecording) return;
    if (document.getElementById('record-btn').classList.contains('disabled')) return;

    FramingGuide.hideOverlay();

    const overlay = document.getElementById('countdown-overlay');
    const numEl = document.getElementById('countdown-num');
    overlay.style.display = 'flex';

    for (let i = 3; i >= 1; i--) {
        numEl.textContent = i;
        numEl.style.animation = 'none';
        void numEl.offsetHeight;
        numEl.style.animation = 'countdown-pulse 0.5s ease-in-out';
        await sleep(1000);
    }

    overlay.style.display = 'none';
    beginRecording();
}

// ── Actually start MediaRecorder ──
function beginRecording() {
    if (isRecording) return;

    isRecording = true;
    recordedChunks = [];

    const mimeTypes = [
        'video/mp4;codecs=avc1',
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm',
    ];
    const mime = mimeTypes.find(t => MediaRecorder.isTypeSupported(t)) || '';

    mediaRecorder = new MediaRecorder(cameraStream, { mimeType: mime || undefined });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = onRecordingDone;

    mediaRecorder.start();
    recordingStartTime = Date.now();

    // Show recording UI
    const container = document.getElementById('camera-container');
    container.className = 'camera-container border-recording';
    document.getElementById('camera-banner').style.display = 'none';
    document.getElementById('recording-indicator').style.display = 'flex';
    document.getElementById('record-btn').className = 'record-btn recording';
    document.getElementById('record-btn').onclick = stopRecording;

    // Elapsed timer
    let elapsed = 0;
    const timerEl = document.getElementById('recording-timer');
    recordTimerInterval = setInterval(() => {
        elapsed++;
        timerEl.textContent = `0:${String(elapsed).padStart(2, '0')} / 0:05`;
    }, 1000);

    // Auto-stop after 5 seconds
    recordAutoStop = setTimeout(() => {
        stopRecording();
    }, 5000);
}

function stopRecording() {
    // Clear timers first
    if (recordTimerInterval) { clearInterval(recordTimerInterval); recordTimerInterval = null; }
    if (recordAutoStop) { clearTimeout(recordAutoStop); recordAutoStop = null; }

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}

async function onRecordingDone() {
    isRecording = false;
    reviewingRecording = true;
    recordedDurationMs = Date.now() - recordingStartTime;
    recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });

    // Hide camera, show preview
    document.getElementById('camera-container').style.display = 'none';
    document.getElementById('status-badges').style.display = 'none';
    document.getElementById('record-controls').style.display = 'none';
    document.getElementById('recording-indicator').style.display = 'none';
    document.getElementById('camera-banner').style.display = 'block';

    const previewEl = document.getElementById('video-preview');
    previewEl.src = URL.createObjectURL(recordedBlob);
    previewEl.style.display = 'block';

    // Quality gate
    const gateContainer = document.getElementById('quality-gate-container');
    gateContainer.innerHTML = '<div class="quality-gate-spinner">Kontrolujem kvalitu...</div>';
    gateContainer.style.display = 'block';
    document.getElementById('preview-controls').style.display = 'none';

    await QualityGate.loadModels();
    const result = await QualityGate.analyze(recordedBlob);

    if (result.skipped || result.passed) {
        if (result.skipped) {
            gateContainer.style.display = 'none';
        } else {
            gateContainer.innerHTML = renderScoreCard(result, true);
        }
        document.getElementById('preview-controls').style.display = 'block';
    } else {
        const reasons = [];
        if (!result.face.passed) reasons.push('Tvár musí byť viditeľná v aspoň 4 z 5 snímok.');
        if (!result.hands.passed) reasons.push('Ruky musia byť viditeľné v aspoň 3 z 5 snímok.');
        if (!result.brightness.passed) reasons.push('Osvetlenie musí byť dobré v aspoň 4 z 5 snímok.');

        gateContainer.innerHTML = renderScoreCard(result, false) +
            `<p style="font-size:13px;color:var(--gray);margin:12px 16px;">${reasons.join(' ')}</p>
             <div style="padding:0 16px 16px;"><button class="btn btn-gray" onclick="retryRecording()">Nahrať znova</button></div>`;
    }
}

function renderScoreCard(result, passed) {
    const cls = passed ? 'pass' : 'fail';
    const title = passed ? 'Video vyzerá dobre' : 'Skúste to znova';
    return `<div class="quality-gate-card ${cls}">
        <h3>${title}</h3>
        <div class="quality-gate-scores">
            <div class="quality-gate-score"><span class="label">Tvár</span><span class="value ${result.face.passed ? 'ok' : 'fail'}">${result.face.score}/5</span></div>
            <div class="quality-gate-score"><span class="label">Ruky</span><span class="value ${result.hands.passed ? 'ok' : 'fail'}">${result.hands.score}/5</span></div>
            <div class="quality-gate-score"><span class="label">Svetlo</span><span class="value ${result.brightness.passed ? 'ok' : 'fail'}">${result.brightness.score}/5</span></div>
        </div>
    </div>`;
}

// ── Shared helpers ──
function resetToCamera() {
    cancelAutoRecord();
    reviewingRecording = false;
    if (recordedBlob) {
        const previewEl = document.getElementById('video-preview');
        if (previewEl && previewEl.src && previewEl.src.startsWith('blob:')) {
            URL.revokeObjectURL(previewEl.src);
        }
        recordedBlob = null;
    }
    document.getElementById('video-preview').style.display = 'none';
    document.getElementById('preview-controls').style.display = 'none';
    document.getElementById('quality-gate-container').style.display = 'none';
    const vp = document.getElementById('variant-prompt');
    if (vp) vp.style.display = 'none';
    document.getElementById('camera-container').style.display = 'block';
    document.getElementById('status-badges').style.display = 'flex';
    document.getElementById('record-controls').style.display = 'block';
    document.getElementById('record-btn').className = 'record-btn';
    document.getElementById('record-btn').onclick = startRecording;
}

function isValidHttpUrl(url) {
    try {
        const u = new URL(url, window.location.origin);
        return u.protocol === 'https:' || u.protocol === 'http:';
    } catch { return false; }
}

function buildRefLinks(sign) {
    if (!sign) return '';
    const links = [];
    const w = encodeURIComponent(sign.word_sk);
    if (sign.link_posunky && isValidHttpUrl(sign.link_posunky)) {
        links.push(`<a href="${sign.link_posunky}" target="_blank" rel="noopener">Posunky.sk</a>`);
    } else {
        links.push(`<a href="https://posunky.sk/?s=${w}" target="_blank" rel="noopener">Posunky.sk</a>`);
    }
    if (sign.link_dictio && isValidHttpUrl(sign.link_dictio)) {
        links.push(`<a href="${sign.link_dictio}" target="_blank" rel="noopener">Dictio.info</a>`);
    }
    return links.join(' · ');
}

function retryRecording() {
    resetToCamera();
}

// ── Submit recording ──
async function submitRecording() {
    if (!recordedBlob || !currentSign) return;

    const submitBtn = document.getElementById('submit-btn');
    submitBtn.textContent = '⏳ Odosielam...';
    submitBtn.disabled = true;

    try {
        const formData = new FormData();
        const ext = recordedBlob.type.includes('mp4') ? 'mp4' : 'webm';
        formData.append('video', recordedBlob, `recording.${ext}`);
        formData.append('sign_id', currentSign.id);
        formData.append('duration_ms', recordedDurationMs);
        formData.append('csrf_token', getCsrfToken());

        const res = await fetch('/api/upload.php', { method: 'POST', body: formData });
        const result = await res.json();

        if (result.ok) {
            showToast('✅ Odoslané!', 'success');
            todayCount++;
            totalCount++;
            updateProgress();
            showVariantOption();
        } else {
            showToast('⚠️ ' + (result.error || 'Chyba'), 'error');
        }
    } catch (err) {
        await OfflineStore.save(recordedBlob, currentSign.id, recordedDurationMs);
        showToast('📶 Uložené offline, odošle sa neskôr', 'offline');
        goToNextSign();
    }

    submitBtn.textContent = '✓ Odoslať';
    submitBtn.disabled = false;
}

function showVariantOption() {
    if (recordedBlob) {
        const previewEl = document.getElementById('video-preview');
        if (previewEl && previewEl.src && previewEl.src.startsWith('blob:')) {
            URL.revokeObjectURL(previewEl.src);
        }
        recordedBlob = null;
    }
    document.getElementById('video-preview').style.display = 'none';
    document.getElementById('quality-gate-container').style.display = 'none';
    document.getElementById('preview-controls').style.display = 'none';
    document.getElementById('record-controls').style.display = 'none';

    const word = currentSign ? currentSign.word_sk.toUpperCase() : '';
    const links = buildRefLinks(currentSign);
    const refLinks = links ? `<p style="font-size:13px;margin-bottom:12px;">Nie ste si istý? Pozrite si: ${links}</p>` : '';

    let variantEl = document.getElementById('variant-prompt');
    if (!variantEl) {
        variantEl = document.createElement('div');
        variantEl.id = 'variant-prompt';
        variantEl.className = 'card';
        variantEl.style.cssText = 'text-align:center;margin-top:16px;';
        document.getElementById('preview-controls').parentNode.appendChild(variantEl);
    }
    variantEl.innerHTML = `
        <p style="font-size:15px;color:var(--gray);margin-bottom:12px;">
            Poznáte ďalší variant posunku <strong>${word}</strong>?
        </p>
        ${refLinks}
        <button class="btn btn-blue" onclick="recordVariant()" style="margin-bottom:8px;">
            ↻ Nahrať variant
        </button>
        <button class="btn btn-gray" onclick="skipVariant()">
            Ďalší posunok →
        </button>
    `;
    variantEl.style.display = 'block';
}

function recordVariant() {
    resetToCamera();
}

function skipVariant() {
    goToNextSign();
}

function goToNextSign() {
    resetToCamera();
    loadNextSign();
}

function updateProgress() {
    const goal = parseInt(document.getElementById('progress-count')?.textContent?.split('/')[1]) || 20;
    document.getElementById('progress-count').textContent = `${todayCount} / ${goal}`;
    document.getElementById('progress-fill').style.width = `${Math.min(100, (todayCount / goal) * 100)}%`;
    const totalEl = document.getElementById('total-count');
    if (totalEl) totalEl.textContent = totalCount;
}

function showToast(text, type) {
    const toast = document.getElementById('toast');
    toast.textContent = text;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 3000);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
