<?php
/**
 * SPJ Collector — Validate recordings from other users
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/auth.php';
require_login();

$user = get_user();
$can_validate = $user['total_recordings'] >= MIN_RECORDINGS_TO_VALIDATE;

$page_title = 'Overiť nahrávky — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<?php if (!$can_validate): ?>
<div class="card" style="text-align: center; margin-top: 40px;">
    <h2>🔒 Overovanie zatiaľ nie je dostupné</h2>
    <p style="color: var(--gray); margin: 16px 0;">
        Potrebujete aspoň <strong><?= MIN_RECORDINGS_TO_VALIDATE ?></strong> vlastných nahrávok.
    </p>
    <p style="font-size: 28px; font-weight: 900; margin-bottom: 8px;">
        <?= $user['total_recordings'] ?> / <?= MIN_RECORDINGS_TO_VALIDATE ?>
    </p>
    <div class="progress-bar" style="margin-bottom: 20px;">
        <div class="fill" style="width: <?= min(100, ($user['total_recordings'] / MIN_RECORDINGS_TO_VALIDATE) * 100) ?>%;"></div>
    </div>
    <a href="/record.php" class="btn btn-blue">⏺ Nahrať posunky →</a>
</div>

<?php else: ?>
<h2 style="text-align: center; margin-bottom: 16px;">Overenie nahrávky</h2>

<!-- Word display -->
<div class="word-display" id="validate-word">...</div>
<div class="help-link" id="validate-links"></div>

<!-- Video -->
<video class="video-preview" id="validate-video" autoplay playsinline loop
       style="margin-bottom: 16px;"></video>

<p style="text-align: center; font-size: 17px; font-weight: 600; margin-bottom: 16px;">
    Je tento posunok správny?
</p>

<div class="validate-btns">
    <button class="btn btn-green" onclick="vote(1)">✓ Áno</button>
    <button class="btn btn-red" onclick="vote(0)">✗ Nie</button>
</div>
<button class="btn btn-gray" id="skip-btn" onclick="loadNextValidation()">→ Preskočiť</button>

<div style="text-align: center; margin-top: 20px; color: var(--gray); font-size: 14px;">
    Overených dnes: <strong id="validated-count">0</strong>
</div>

<!-- No more to validate -->
<div class="card" id="no-more" style="display: none; text-align: center; margin-top: 40px;">
    <h2>✅ Všetko overené!</h2>
    <p style="color: var(--gray); margin: 12px 0;">Momentálne nie sú žiadne nahrávky na overenie.</p>
    <a href="/record.php" class="btn btn-blue">⏺ Nahrať posunky →</a>
</div>

<script>
let currentRecording = null;
let validatedCount = 0;

document.addEventListener('DOMContentLoaded', loadNextValidation);

async function loadNextValidation() {
    try {
        const res = await fetch('/api/next_validation.php');
        const data = await res.json();

        if (data.error) {
            document.getElementById('validate-word').style.display = 'none';
            document.getElementById('validate-links').style.display = 'none';
            document.getElementById('validate-video').style.display = 'none';
            document.querySelector('.validate-btns').style.display = 'none';
            document.getElementById('skip-btn').style.display = 'none';
            document.getElementById('no-more').style.display = 'block';
            return;
        }

        currentRecording = data;
        document.getElementById('validate-word').textContent = data.word_sk.toUpperCase();

        // Help links
        let links = '';
        if (data.link_posunky) links += `<a href="${data.link_posunky}" target="_blank">Posunky.sk</a>`;
        if (data.link_posunky && data.link_dictio) links += ' · ';
        if (data.link_dictio) links += `<a href="${data.link_dictio}" target="_blank">Dictio.info</a>`;
        if (links) links = 'Referencia: ' + links;
        document.getElementById('validate-links').innerHTML = links;

        // Video
        const video = document.getElementById('validate-video');
        video.src = '/api/video.php?file=' + encodeURIComponent(data.video_filename) + '&dir=pending';
        video.style.display = 'block';
    } catch (err) {
        console.error('Failed to load validation:', err);
    }
}

async function vote(value) {
    if (!currentRecording) return;

    try {
        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';
        const formData = new FormData();
        formData.append('recording_id', currentRecording.id);
        formData.append('vote', value);
        formData.append('csrf_token', csrfToken);

        await fetch('/api/vote.php', { method: 'POST', body: formData });
        validatedCount++;
        document.getElementById('validated-count').textContent = validatedCount;
        loadNextValidation();
    } catch (err) {
        console.error('Vote failed:', err);
    }
}
</script>
<?php endif; ?>

<?php require_once __DIR__ . '/includes/footer.php'; ?>
