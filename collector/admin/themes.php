<?php
/**
 * Admin — Themes management tab (included by admin/index.php)
 */

$pdo = get_db();
$themes = $pdo->query("
    SELECT t.id, t.name, t.emoji, t.sort_order, COUNT(s.id) as word_count
    FROM themes t
    LEFT JOIN signs s ON s.theme_id = t.id
    GROUP BY t.id, t.name, t.emoji, t.sort_order
    ORDER BY t.sort_order ASC
")->fetchAll();
?>

<!-- Add theme form -->
<details class="admin-form" open>
    <summary style="cursor:pointer;font-weight:700;font-size:15px;">+ Pridať tému</summary>
    <form method="POST" action="/admin/api/themes.php" style="margin-top:12px;">
        <input type="hidden" name="action" value="add">
        <?= csrf_field() ?>
        <div class="form-row">
            <div class="form-group">
                <label>Názov</label>
                <input type="text" name="name" placeholder="Jedlo a nápoje" required>
            </div>
            <div class="form-group" style="max-width:100px;">
                <label>Emoji</label>
                <input type="text" name="emoji" placeholder="🍞" maxlength="10">
            </div>
            <div class="form-group" style="max-width:100px;">
                <label>Poradie</label>
                <input type="number" name="sort_order" value="<?= count($themes) + 1 ?>">
            </div>
        </div>
        <button type="submit" class="btn btn-blue" style="width:auto;padding:10px 24px;">Pridať</button>
    </form>
</details>

<!-- Themes table -->
<table class="admin-table">
    <thead>
        <tr>
            <th>#</th>
            <th>Emoji</th>
            <th>Názov</th>
            <th>Slov</th>
            <th>Akcie</th>
        </tr>
    </thead>
    <tbody>
        <?php foreach ($themes as $t): ?>
        <tr>
            <td><?= $t['sort_order'] ?></td>
            <td style="font-size:24px;"><?= htmlspecialchars($t['emoji']) ?></td>
            <td style="font-weight:600;"><?= htmlspecialchars($t['name']) ?></td>
            <td><?= $t['word_count'] ?></td>
            <td class="admin-actions">
                <form method="POST" action="/admin/api/themes.php" style="display:inline;">
                    <input type="hidden" name="action" value="move_up">
                    <input type="hidden" name="theme_id" value="<?= $t['id'] ?>">
                    <?= csrf_field() ?>
                    <button type="submit">↑</button>
                </form>
                <form method="POST" action="/admin/api/themes.php" style="display:inline;">
                    <input type="hidden" name="action" value="move_down">
                    <input type="hidden" name="theme_id" value="<?= $t['id'] ?>">
                    <?= csrf_field() ?>
                    <button type="submit">↓</button>
                </form>
                <form method="POST" action="/admin/api/themes.php" style="display:inline;"
                      onsubmit="return confirm('Zmazať tému? Slová zostanú ako nekategorizované.')">
                    <input type="hidden" name="action" value="delete">
                    <input type="hidden" name="theme_id" value="<?= $t['id'] ?>">
                    <?= csrf_field() ?>
                    <button type="submit" class="delete">Zmazať</button>
                </form>
            </td>
        </tr>
        <?php endforeach; ?>
    </tbody>
</table>
