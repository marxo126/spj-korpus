/**
 * SPJ Collector — IndexedDB offline buffer
 * Saves recordings locally if upload fails. Retries on next visit.
 */

const OfflineStore = {
    DB_NAME: 'spj_collector',
    STORE_NAME: 'pending_uploads',
    DB_VERSION: 1,

    async getDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                    db.createObjectStore(this.STORE_NAME, { keyPath: 'id', autoIncrement: true });
                }
            };
        });
    },

    async save(blob, signId, durationMs) {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(this.STORE_NAME, 'readwrite');
            const store = tx.objectStore(this.STORE_NAME);
            const request = store.add({
                blob: blob,
                sign_id: signId,
                duration_ms: durationMs || 3000,
                created_at: new Date().toISOString(),
                retries: 0
            });
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    },

    async remove(id) {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(this.STORE_NAME, 'readwrite');
            const store = tx.objectStore(this.STORE_NAME);
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    },

    async getAll() {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(this.STORE_NAME, 'readonly');
            const store = tx.objectStore(this.STORE_NAME);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    },

    async count() {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(this.STORE_NAME, 'readonly');
            const store = tx.objectStore(this.STORE_NAME);
            const request = store.count();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    },

    async retryAll() {
        const pending = await this.getAll();
        if (pending.length === 0) return;

        console.log(`Retrying ${pending.length} offline uploads...`);

        for (const item of pending) {
            // Drop items that failed too many times or are older than 24h
            const age = Date.now() - new Date(item.created_at).getTime();
            if ((item.retries || 0) >= 3 || age > 24 * 60 * 60 * 1000) {
                await this.remove(item.id);
                console.warn(`Dropped stale offline recording ${item.id}`);
                continue;
            }

            try {
                const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';
                const formData = new FormData();
                const ext = item.blob.type.includes('mp4') ? 'mp4' : 'webm';
                formData.append('video', item.blob, `recording.${ext}`);
                formData.append('sign_id', item.sign_id);
                formData.append('duration_ms', item.duration_ms || 3000);
                formData.append('csrf_token', csrfToken);

                const res = await fetch('/api/upload.php', { method: 'POST', body: formData });
                const result = await res.json();

                if (result.ok) {
                    await this.remove(item.id);
                    console.log(`Uploaded offline recording ${item.id}`);
                } else {
                    // Increment retry counter
                    await this.updateRetries(item.id, (item.retries || 0) + 1);
                }
            } catch (err) {
                console.warn(`Still offline, keeping recording ${item.id}`);
                break; // stop retrying if still offline
            }
        }
    },

    async updateRetries(id, retries) {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(this.STORE_NAME, 'readwrite');
            const store = tx.objectStore(this.STORE_NAME);
            const getReq = store.get(id);
            getReq.onsuccess = () => {
                const item = getReq.result;
                if (item) {
                    item.retries = retries;
                    store.put(item);
                }
                resolve();
            };
            getReq.onerror = () => reject(getReq.error);
        });
    }
};

// Retry pending uploads when page loads and when coming back online
document.addEventListener('DOMContentLoaded', () => OfflineStore.retryAll());
window.addEventListener('online', () => OfflineStore.retryAll());
