<?php
/**
 * SPJ Collector — Validate recordings from other users
 */

require_once __DIR__ . '/includes/config.php';
require_once __DIR__ . '/includes/admin_auth.php';
require_researcher();

$page_title = 'Overiť nahrávky — ' . SITE_NAME;
require_once __DIR__ . '/includes/header.php';
?>

<h2 style="text-align: center; margin-bottom: 16px;">Overenie nahrávky</h2>

<!-- Word display -->
<div class="word-display" id="validate-word">...</div>
<div class="help-link" id="validate-links"></div>

<!-- Recording context -->
<div id="validate-meta" style="text-align: center; font-size: 13px; color: var(--gray); margin-bottom: 8px;"></div>

<!-- Video -->
<video class="video-preview" id="validate-video" autoplay muted playsinline loop
       style="margin-bottom: 16px;" aria-label="Nahrávka posunku na overenie"></video>

<!-- Vote progress -->
<div id="vote-progress" style="text-align: center; font-size: 13px; color: var(--gray); margin-bottom: 12px;"></div>

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
            document.getElementById('validate-meta').style.display = 'none';
            document.getElementById('validate-video').style.display = 'none';
            document.getElementById('vote-progress').style.display = 'none';
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

        // Recording context: contributor, hand, variant count
        let meta = [];
        if (data.contributor_name) meta.push(data.contributor_name);
        if (data.dominant_hand) meta.push(data.dominant_hand === 'left' ? 'ľavák' : 'pravák');
        meta.push(`${data.variants_approved} schválených / ${data.variants_total} celkom`);
        document.getElementById('validate-meta').textContent = meta.join(' · ');

        // Vote progress on this recording
        let vp = `👍 ${data.validations_up}/${<?= VOTES_TO_APPROVE ?>}`;
        vp += `  👎 ${data.validations_down}/${<?= VOTES_TO_REJECT ?>}`;
        document.getElementById('vote-progress').textContent = vp;

        // Video
        const video = document.getElementById('validate-video');
        video.src = '/video/pending/' + encodeURIComponent(data.video_filename);
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

<?php require_once __DIR__ . '/includes/footer.php'; ?>
