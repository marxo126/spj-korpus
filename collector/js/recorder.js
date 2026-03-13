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
let qualityChecker = null;
let isRecording = false;
let recordingStartTime = 0;
let recordedDurationMs = 3000;

function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content || '';
}

// ── Init on page load ──
document.addEventListener('DOMContentLoaded', async () => {
    // Camera permission priming — explain why we need camera
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
});

// ── Camera: always 720p, always front, no user choice ──
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false
        });

        // If device gave higher than 720p, downscale via canvas
        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();

        if (settings.height > 720) {
            const tempVideo = document.createElement('video');
            tempVideo.srcObject = stream;
            tempVideo.muted = true;
            tempVideo.playsInline = true;
            await tempVideo.play();

            const canvas = document.createElement('canvas');
            canvas.width = 1280;
            canvas.height = 720;
            const ctx = canvas.getContext('2d');

            function draw() {
                ctx.drawImage(tempVideo, 0, 0, 1280, 720);
                if (cameraStream) requestAnimationFrame(draw);
            }
            draw();

            cameraStream = canvas.captureStream(30);
        } else {
            cameraStream = stream;
        }

        const preview = document.getElementById('camera-preview');
        preview.srcObject = cameraStream;
        await preview.play();

        document.getElementById('camera-loading').style.display = 'none';
        document.getElementById('camera-container').style.display = 'block';

        // Start quality checker if not yet running
        if (!qualityChecker) {
            qualityChecker = new QualityChecker(preview, updateQualityStatus);
            qualityChecker.start();
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
        // Build query params for theme/sign filtering
        const params = new URLSearchParams();
        const themeEl = document.getElementById('current-theme-id');
        const themeSetEl = document.getElementById('theme-id-set');
        const signEl = document.getElementById('initial-sign-id');

        if (themeEl && themeSetEl && themeSetEl.value === '1') {
            params.set('theme_id', themeEl.value);
        }
        if (signEl && signEl.value !== '0') {
            params.set('sign_id', signEl.value);
            signEl.value = '0'; // clear after first use
        }

        const qs = params.toString();
        const res = await fetch('/api/next_sign.php' + (qs ? '?' + qs : ''));
        const sign = await res.json();

        if (sign.error) {
            const isThemeDone = sign.error === 'all_done' && themeEl && themeSetEl && themeSetEl.value === '1';
            document.getElementById('word-display').textContent = '🎉 Hotovo!';
            document.getElementById('help-links').innerHTML = isThemeDone
                ? 'Všetky znaky v tejto téme sú nahrané. <a href="/themes.php">Späť na témy</a>'
                : 'Všetky znaky sú nahrané.';
            document.getElementById('record-btn').classList.add('disabled');
            return;
        }

        currentSign = sign;
        document.getElementById('word-display').textContent = sign.word_sk.toUpperCase();

        // Build help links
        let links = 'Nepoznáte znak? → ';
        if (sign.link_posunky) links += `<a href="${sign.link_posunky}" target="_blank" rel="noopener">Posunky.sk</a>`;
        if (sign.link_posunky && sign.link_dictio) links += ' · ';
        if (sign.link_dictio) links += `<a href="${sign.link_dictio}" target="_blank" rel="noopener">Dictio.info</a>`;
        if (!sign.link_posunky && !sign.link_dictio) links = '';
        document.getElementById('help-links').innerHTML = links;

    } catch (err) {
        console.error('Failed to load next sign:', err);
    }
}

// ── Quality status callback ──
function updateQualityStatus(status) {
    if (isRecording) return; // don't update UI during recording

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
    document.getElementById('record-label').textContent =
        blocked ? 'Ukážte ruky a tvár' : 'Stlačte pre nahrávanie';
}

// ── Countdown → Record ──
async function startRecording() {
    if (!currentSign || isRecording) return;
    if (document.getElementById('record-btn').classList.contains('disabled')) return;

    // 3-2-1 countdown
    const overlay = document.getElementById('countdown-overlay');
    const numEl = document.getElementById('countdown-num');
    overlay.style.display = 'flex';

    for (let i = 3; i >= 1; i--) {
        numEl.textContent = i;
        numEl.style.animation = 'none';
        void numEl.offsetHeight; // force reflow
        numEl.style.animation = 'countdown-pulse 0.5s ease-in-out';
        await sleep(1000);
    }
    overlay.style.display = 'none';

    // Start recording
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

    // Timer
    let elapsed = 0;
    const timerEl = document.getElementById('recording-timer');
    const timerInterval = setInterval(() => {
        elapsed++;
        timerEl.textContent = `0:${String(elapsed).padStart(2, '0')} / 0:03`;
    }, 1000);

    // Auto-stop after 3 seconds
    setTimeout(() => {
        clearInterval(timerInterval);
        stopRecording();
    }, 3000);
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}

function onRecordingDone() {
    isRecording = false;
    recordedDurationMs = Date.now() - recordingStartTime;
    recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });

    // Show preview
    document.getElementById('camera-container').style.display = 'none';
    document.getElementById('status-badges').style.display = 'none';
    document.getElementById('record-controls').style.display = 'none';

    const previewEl = document.getElementById('video-preview');
    previewEl.src = URL.createObjectURL(recordedBlob);
    previewEl.style.display = 'block';
    document.getElementById('preview-controls').style.display = 'block';

    // Reset recording UI
    document.getElementById('recording-indicator').style.display = 'none';
    document.getElementById('camera-banner').style.display = 'block';
}

// ── Re-record ──
function retryRecording() {
    URL.revokeObjectURL(document.getElementById('video-preview').src);
    recordedBlob = null;

    document.getElementById('video-preview').style.display = 'none';
    document.getElementById('preview-controls').style.display = 'none';
    document.getElementById('camera-container').style.display = 'block';
    document.getElementById('status-badges').style.display = 'flex';
    document.getElementById('record-controls').style.display = 'block';
    document.getElementById('record-btn').className = 'record-btn';
    document.getElementById('record-btn').onclick = startRecording;
}

// ── Submit recording ──
async function submitRecording() {
    if (!recordedBlob || !currentSign) return;

    const submitBtn = document.getElementById('submit-btn');
    submitBtn.textContent = '⏳ Odosielam...';
    submitBtn.disabled = true;

    // Save to IndexedDB first (offline safety)
    const offlineId = await OfflineStore.save(recordedBlob, currentSign.id, recordedDurationMs);

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
            await OfflineStore.remove(offlineId);
            showToast('✅ Odoslané!', 'success');
            todayCount++;
            updateProgress();
            goToNextSign();
        } else {
            showToast('⚠️ ' + (result.error || 'Chyba'), 'error');
        }
    } catch (err) {
        showToast('📶 Uložené offline, odošle sa neskôr', 'offline');
        goToNextSign();
    }

    submitBtn.textContent = '✓ Odoslať';
    submitBtn.disabled = false;
}

function goToNextSign() {
    URL.revokeObjectURL(document.getElementById('video-preview').src);
    recordedBlob = null;
    document.getElementById('video-preview').style.display = 'none';
    document.getElementById('preview-controls').style.display = 'none';
    document.getElementById('camera-container').style.display = 'block';
    document.getElementById('status-badges').style.display = 'flex';
    document.getElementById('record-controls').style.display = 'block';
    document.getElementById('record-btn').className = 'record-btn';
    document.getElementById('record-btn').onclick = startRecording;
    loadNextSign();
}

function updateProgress() {
    const goal = parseInt(document.getElementById('progress-count')?.textContent?.split('/')[1]) || 20;
    document.getElementById('progress-count').textContent = `${todayCount} / ${goal}`;
    document.getElementById('progress-fill').style.width = `${Math.min(100, (todayCount / goal) * 100)}%`;
}

// ── Toast notification ──
function showToast(text, type) {
    const toast = document.getElementById('toast');
    toast.textContent = text;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 3000);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
