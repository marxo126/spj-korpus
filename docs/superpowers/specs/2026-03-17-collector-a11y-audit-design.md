# Collector Accessibility Audit Tool — Design Spec

**Date:** 2026-03-17
**Project:** SPJ Collector (zber.spj.sk)
**Status:** Approved
**Reference:** Adapted from fluentiagrant-app `tools/a11y-audit/`

---

## 1. Overview

A static + optional runtime accessibility audit tool for the SPJ Sign Language Collector app. Enforces WCAG 2.2 Level AA and EN 301 549 v3.2.1 (EU Web Accessibility Directive) compliance across the PHP/HTML/JS/CSS codebase.

### Goals

- Catch accessibility violations before deployment
- Produce actionable fix suggestions with disability impact info
- Track compliance over time via diffable JSON reports
- Audit the existing accessibility statement against EN 301 549 Annex A
- Include domain-specific checks for sign language video recording workflows

### Non-Goals

- Replacing manual testing with screen readers (NVDA, VoiceOver)
- LLM-based analysis — all checks are deterministic and repeatable
- Automated fixing — the tool reports, humans fix

---

## 2. Architecture

```
collector/tools/a11y-audit/
├── audit.py                    # CLI entry point
├── config.yaml                 # Thresholds, module toggles, paths
├── requirements.txt            # pyyaml, jinja2, colorama, beautifulsoup4, lxml
├── rules/                      # 16 rule modules
│   ├── base.py                 # BaseRule, Finding, Severity, Detection enums
│   ├── helpers.py              # Shared utilities (color parsing, contrast calc)
│   ├── contrast.py             # WCAG 1.4.3, 1.4.11
│   ├── color.py                # WCAG 1.4.1, 1.3.3
│   ├── typography.py           # WCAG 1.4.4, 1.4.8, 1.4.12
│   ├── layout.py               # WCAG 1.4.10, 2.5.8
│   ├── structure.py            # WCAG 1.3.1, 1.3.2, 2.4.1-6
│   ├── aria.py                 # WCAG 4.1.2
│   ├── focus.py                # WCAG 2.4.3, 2.4.7, 2.4.11
│   ├── keyboard.py             # WCAG 2.1.1, 2.1.2, 2.1.4
│   ├── interactive.py          # WCAG 1.4.13, 2.5.1-8
│   ├── forms.py                # WCAG 1.3.5, 3.3.1-7
│   ├── media.py                # WCAG 1.1.1, 1.2.1-5, 1.4.2
│   ├── language.py             # WCAG 3.1.1, 3.1.2
│   ├── motion.py               # WCAG 2.2.2, 2.3.1, 2.3.3
│   ├── cognitive.py            # WCAG 2.2.1, 3.2.x, 3.3.4, 3.3.7
│   ├── compliance.py           # EN 301 549 Annex A
│   └── collector.py            # Domain-specific (video, camera, quality, consent)
├── parsers/
│   ├── models.py               # ParseContext, FileContext, ElementNode, CSSContext
│   ├── php_parser.py           # BeautifulSoup HTML extraction from PHP files
│   ├── css_parser.py           # CSS variables, animations, media queries
│   └── js_parser.py            # Event handlers, ARIA manipulation, timers
├── reporters/
│   ├── terminal.py             # Colored CLI output (colorama)
│   ├── json_report.py          # Machine-readable JSON (timestamped, diffable)
│   ├── html_report.py          # Visual dashboard (Jinja2)
│   └── templates/
│       └── report.html.j2      # Self-contained HTML template
├── playwright_bridge.py        # Optional runtime checks (--runtime flag)
└── reports/                    # Generated output (.gitignored)
```

---

## 3. Parsers

### 3.1 PHP Parser (`php_parser.py`)

PHP files contain mixed `<?php ... ?>` blocks and HTML. The parser:

1. **Reads the raw file** preserving line numbers
2. **Strips PHP blocks** — replaces `<?php ... ?>` and `<?= ... ?>` with empty placeholder text of equal line count (preserves line numbers)
3. **Parses resulting HTML** with BeautifulSoup (lxml backend) to extract element trees
4. **Extracts PHP echo HTML** — detects `echo "<div..."`, `echo '<img...'`, and heredoc HTML inside PHP blocks via regex, parses those fragments with their source line numbers
5. **Builds `FileContext`** with `ElementNode` tree, raw content, and line array

For `includes/header.php` and `includes/footer.php` which are included in every page, the parser marks them as "shared includes" so violations are reported once, not per-including-page.

### 3.2 CSS Parser (`css_parser.py`)

Parses `css/style.css` (single file):

- **CSS custom properties** (`--red: #DC2626`) with resolved hex values
- **Dark mode variants** — properties inside `html.dark` or `@media (prefers-color-scheme: dark)`
- **Animations** — `@keyframes` with duration/iteration, transition properties
- **Media queries** — tracks `prefers-reduced-motion`, `prefers-contrast`, `prefers-color-scheme`
- **Font sizes** — all `font-size` declarations with computed px values
- **Focus styles** — `outline`, `box-shadow` on `:focus` / `:focus-visible` selectors
- **Target sizes** — `min-height`, `min-width`, `padding` on interactive elements

### 3.3 JS Parser (`js_parser.py`)

Regex-based extraction from `js/*.js`:

- **Event listeners** — `addEventListener('click', ...)` without corresponding `keydown`/`keypress`
- **ARIA manipulation** — `setAttribute('aria-*', ...)`, `setAttribute('role', ...)`
- **Timers** — `setTimeout(fn, ms)`, `setInterval(fn, ms)` with durations
- **DOM manipulation** — `innerHTML`, `insertAdjacentHTML` (potential a11y issues)
- **Focus management** — `.focus()`, `.blur()` calls
- **MediaPipe callbacks** — quality check result handlers (for collector-specific checks)

---

## 4. Rule Modules

### 4.1 Base Classes (`base.py`)

```python
class Severity(Enum):
    CRITICAL = "critical"   # Blocks users entirely
    SERIOUS = "serious"     # Significantly impairs access
    MODERATE = "moderate"   # Inconvenience, workaround exists
    MINOR = "minor"         # Best practice, cosmetic

class Detection(Enum):
    STATIC = "static"
    RUNTIME = "runtime"

@dataclass(frozen=True)
class Finding:
    rule_id: str          # "contrast"
    check_id: str         # "text-contrast-ratio"
    severity: Severity
    wcag: str             # "1.4.3"
    wcag_name: str        # "Contrast (Minimum)"
    message: str
    file: str             # relative path
    line: int
    element: str = ""     # code snippet
    fix: str = ""         # suggested remediation
    impact: tuple = ()    # ("low-vision", "color-blind")
    detection: Detection = Detection.STATIC
    suppressed: bool = False
    suppression_reason: str = ""

class BaseRule(ABC):
    id: str
    name: str
    wcag_criteria: tuple[str, ...]
    standards: tuple[str, ...]

    @abstractmethod
    def check(self, ctx: ParseContext, config: dict) -> list[Finding]: ...

    def _finding(self, **kwargs) -> Finding:
        """Factory method — auto-fills rule_id from self.id"""
```

### 4.2 Helpers (`helpers.py`)

Shared utilities:

- `parse_color(value: str) -> tuple[int,int,int]` — hex, rgb(), hsl(), named colors
- `relative_luminance(r, g, b) -> float` — WCAG luminance formula
- `contrast_ratio(fg, bg) -> float` — WCAG contrast ratio
- `is_large_text(font_size_px, bold) -> bool` — ≥18px or ≥14px bold
- `resolve_css_var(name, css_ctx) -> str` — follow var() references
- `strip_php(content: str) -> tuple[str, list[PhpBlock]]` — remove PHP, preserve lines
- `line_number_at_offset(content, offset) -> int` — byte offset to line number

### 4.3 Rule Details

#### contrast.py — WCAG 1.4.3, 1.4.11

| Check ID | What | Severity |
|----------|------|----------|
| `text-contrast` | Text color vs background ≥ 4.5:1 (normal) or 3:1 (large) | Serious |
| `ui-contrast` | UI component borders/icons vs background ≥ 3:1 | Serious |
| `dark-mode-contrast` | Same checks for `html.dark` variant colors | Serious |
| `placeholder-contrast` | Placeholder text contrast ≥ 4.5:1 | Moderate |

Resolves CSS custom properties (`var(--red)`) to hex values. Checks both light and dark mode.

#### color.py — WCAG 1.4.1, 1.3.3

| Check ID | What | Severity |
|----------|------|----------|
| `color-only-status` | Status badges using color without icon/text | Serious |
| `link-distinction` | Links distinguishable from text by more than color | Serious |
| `error-color-only` | Error states using red alone without icon/text | Serious |

#### typography.py — WCAG 1.4.4, 1.4.8, 1.4.12

| Check ID | What | Severity |
|----------|------|----------|
| `min-font-size` | No font-size below 14px | Moderate |
| `relative-units` | Prefer rem/em over px for font-size | Minor |
| `line-height` | line-height ≥ 1.5 for body text | Moderate |
| `letter-spacing` | letter-spacing not negative | Moderate |
| `text-justify` | No `text-align: justify` (dyslexia) | Minor |
| `line-length` | Max line length ~80ch | Minor |

#### layout.py — WCAG 1.4.10, 2.5.8

| Check ID | What | Severity |
|----------|------|----------|
| `reflow-320` | No horizontal scroll at 320px viewport | Serious |
| `target-size-aa` | Interactive elements ≥ 24×24px | Serious |
| `target-size-enhanced` | Interactive elements ≥ 44×44px (recommended) | Minor |
| `spacing` | Adequate spacing between interactive elements | Moderate |
| `viewport-meta` | `user-scalable=yes`, `maximum-scale` ≥ 2 | Serious |

#### structure.py — WCAG 1.3.1, 1.3.2, 2.4.1-6

| Check ID | What | Severity |
|----------|------|----------|
| `landmarks` | `<main>`, `<nav>`, `<header>`, `<footer>` present | Serious |
| `heading-hierarchy` | No skipped heading levels (h1→h3) | Moderate |
| `single-h1` | Exactly one `<h1>` per page | Moderate |
| `skip-link` | Skip-to-content link present and functional | Serious |
| `page-title` | `<title>` element present and descriptive | Serious |
| `nav-aria-label` | Multiple `<nav>` elements have unique `aria-label` | Moderate |
| `list-structure` | Lists use `<ul>`/`<ol>` + `<li>`, not styled divs | Minor |

#### aria.py — WCAG 4.1.2

| Check ID | What | Severity |
|----------|------|----------|
| `valid-role` | ARIA roles are valid WAI-ARIA values | Critical |
| `required-attrs` | Required ARIA attributes present (e.g., `aria-checked` on `role="checkbox"`) | Serious |
| `redundant-role` | No redundant roles (e.g., `<nav role="navigation">`) | Minor |
| `aria-hidden-focus` | `aria-hidden="true"` elements not focusable | Critical |
| `aria-label-empty` | `aria-label` / `aria-labelledby` not empty | Serious |
| `live-region-valid` | `aria-live` values are "polite", "assertive", or "off" | Moderate |

#### focus.py — WCAG 2.4.3, 2.4.7, 2.4.11

| Check ID | What | Severity |
|----------|------|----------|
| `outline-none` | `outline: none` / `outline: 0` without replacement style | Critical |
| `focus-visible` | `:focus` or `:focus-visible` styles exist for interactive elements | Serious |
| `focus-indicator-contrast` | Focus indicator has ≥ 3:1 contrast | Serious |
| `tabindex-positive` | No `tabindex` > 0 (disrupts natural tab order) | Moderate |
| `focus-trap` | Modals trap focus properly (checked via JS analysis) | Serious |

#### keyboard.py — WCAG 2.1.1, 2.1.2, 2.1.4

| Check ID | What | Severity |
|----------|------|----------|
| `click-no-key` | `onclick` / `addEventListener('click')` without keyboard equivalent | Serious |
| `mouse-only-handler` | `onmouseover`/`onmouseout` without `onfocus`/`onblur` | Serious |
| `accesskey-conflict` | Duplicate `accesskey` values | Moderate |
| `interactive-div` | `<div>` or `<span>` with click handler but no `role`/`tabindex` | Serious |

#### interactive.py — WCAG 1.4.13, 2.5.1-8

| Check ID | What | Severity |
|----------|------|----------|
| `hover-content` | Hover-triggered content must be dismissable and hoverable | Moderate |
| `touch-target` | Touch targets ≥ 48px on mobile | Serious |
| `drag-alternative` | Drag operations have non-drag alternative | Serious |

#### forms.py — WCAG 1.3.5, 3.3.1-8

| Check ID | What | Severity |
|----------|------|----------|
| `input-label` | Every `<input>`, `<select>`, `<textarea>` has associated `<label>` or `aria-label` | Critical |
| `label-for` | `<label for="id">` matches an input's `id` | Serious |
| `autocomplete` | Login/email/name inputs have `autocomplete` attribute | Moderate |
| `error-identification` | Error messages identify the field and describe the error | Serious |
| `required-indicator` | Required fields marked with more than just color | Moderate |
| `submit-button` | Forms have explicit submit button (not just Enter) | Moderate |
| `accessible-auth` | Authentication doesn't rely on cognitive function tests (WCAG 2.2 3.3.8) | Serious |

#### media.py — WCAG 1.1.1, 1.2.1-5, 1.4.2

| Check ID | What | Severity |
|----------|------|----------|
| `img-alt` | All `<img>` have `alt` attribute | Critical |
| `decorative-alt` | Decorative images use `alt=""` + `role="presentation"` | Minor |
| `video-track` | `<video>` elements have `<track kind="captions">` or ARIA text alternative | Serious |
| `video-autoplay` | Autoplay video must be muted | Moderate |
| `svg-accessible` | SVG elements have `<title>` or `aria-label` | Moderate |

#### language.py — WCAG 3.1.1, 3.1.2

| Check ID | What | Severity |
|----------|------|----------|
| `html-lang` | `<html>` has valid `lang` attribute | Critical |
| `html-lang-valid` | `lang` value is valid BCP 47 tag | Serious |
| `lang-change` | Foreign-language text spans have `lang` attribute | Minor |

#### motion.py — WCAG 2.2.2, 2.3.1, 2.3.3

| Check ID | What | Severity |
|----------|------|----------|
| `prefers-reduced-motion` | CSS animations/transitions respect `prefers-reduced-motion` | Serious |
| `animation-duration` | Animations < 5s or have pause/stop | Moderate |
| `flash-rate` | No content flashes > 3 times/second | Critical |
| `auto-scroll` | No auto-scrolling without user control | Moderate |
| `toast-duration` | Toast/notification animations last ≥ toast_min_ms (5000ms) | Moderate |

#### cognitive.py — WCAG 2.2.1, 3.2.x, 3.3.4, 3.3.7

| Check ID | What | Severity |
|----------|------|----------|
| `session-timeout` | Session timeouts ≥ 20 min or user can extend | Serious |
| `consistent-nav` | Navigation is consistent across pages | Moderate |
| `on-focus-change` | No context change on focus alone | Serious |
| `error-prevention` | Destructive actions (delete account) are reversible or confirmed | Serious |
| `redundant-entry` | Previously entered info not re-requested | Minor |

#### compliance.py — EN 301 549 Annex A

Parses `accessibility-statement.php` and checks for required elements:

| Check ID | Required Element | Severity |
|----------|-----------------|----------|
| `conformance-status` | "fully"/"partially"/"not" conformant statement | Critical |
| `non-accessible-content` | List of known accessibility issues | Serious |
| `preparation-date` | Date present and < 12 months old | Serious |
| `review-method` | Self-assessment or external audit noted | Moderate |
| `feedback-mechanism` | Contact email/form + response time commitment | Critical |
| `enforcement-link` | Link to national enforcement body | Serious |
| `scope-defined` | Which pages/features are covered | Moderate |
| `standards-reference` | WCAG 2.1/2.2 AA + EN 301 549 cited | Serious |

Outputs a gaps checklist with fix text for each missing element.

#### collector.py — Domain-Specific

| Check ID | What | Severity |
|----------|------|----------|
| `camera-fallback` | Camera permission flow must have accessible denial fallback with text explanation | Serious |
| `quality-aria-live` | MediaPipe quality badges need `aria-live="polite"` region | Serious |
| `consent-keyboard-trap` | Consent/GDPR modals must trap focus and be Escape-dismissable | Critical |
| `consent-focus-return` | After modal close, focus returns to trigger element | Moderate |
| `leaderboard-table` | Leaderboard/stats must use semantic `<table>` with `<th scope>` | Moderate |
| `recording-status` | Recording state changes need screen reader announcements | Serious |
| `timer-accessible` | Recording timer must have `aria-live` or `role="timer"` | Moderate |
| `offline-notification` | Offline/retry status must be announced to assistive tech | Moderate |
| `framing-guide-alt` | SVG framing guide overlay needs text alternative | Moderate |

---

## 5. Reporters

### 5.1 Terminal Reporter

Colored output using colorama:

- Severity badges: `[X]` critical (red), `[!]` serious (yellow), `[~]` moderate (blue), `[i]` minor (gray)
- Each finding shows: severity, rule_id, check_id, WCAG criterion, message, file:line, fix suggestion, code snippet
- Summary: counts by severity, PASS/FAIL verdict
- Exit code: 0 = no violations, 1 = violations found

### 5.2 JSON Reporter

Machine-readable, timestamped report:

```json
{
  "timestamp": "2026-03-17T10:00:00",
  "project": "SPJ Collector",
  "files_scanned": 28,
  "rules_checked": 16,
  "summary": {"critical": 2, "serious": 15, "moderate": 30, "minor": 12},
  "findings": [...],
  "suppressed": [...],
  "compliance_gaps": [...]
}
```

Supports `--compare` flag to diff against a previous report (new/fixed/unchanged).

### 5.3 HTML Reporter

Self-contained dashboard (Jinja2 template):

- Summary cards by severity
- Findings grouped by module, then by file
- WCAG criterion cross-reference table
- Compliance gaps checklist
- Filter/sort controls (JS in template)
- No external dependencies (CSS/JS inline)

---

## 6. Playwright Bridge (Optional)

Activated with `--runtime --base-url http://localhost:8080`:

- Launches headless Chromium via Playwright
- Navigates to each page (index, record, validate, themes, progress, thanks, terms, consent, accessibility-statement, forgot-password, reset-password, verify-email, admin pages)
- Runs axe-core for rendered DOM checks
- Tests actual focus visibility, contrast ratios, target sizes
- Merges runtime findings with static findings using composite key `(wcag, file, element)`
- Deduplicates overlapping findings (static finding takes precedence for file:line info)

---

## 7. CLI Interface

```bash
# Full static audit (default)
python collector/tools/a11y-audit/audit.py

# With runtime checks
python collector/tools/a11y-audit/audit.py --runtime --base-url http://localhost:8080

# Specific modules
python collector/tools/a11y-audit/audit.py --only contrast,forms,collector

# Specific files
python collector/tools/a11y-audit/audit.py --files "*.php"

# Output format
python collector/tools/a11y-audit/audit.py --format all
python collector/tools/a11y-audit/audit.py --format json
python collector/tools/a11y-audit/audit.py --compare reports/prev.json

# Verbosity
python collector/tools/a11y-audit/audit.py --verbose
```

---

## 8. Configuration (`config.yaml`)

```yaml
project:
  name: "SPJ Collector (zber.spj.sk)"
  root: "../.."
  directories:
    pages: "."
    admin: "admin"
    includes: "includes"
    api: "api"
  css_file: "css/style.css"
  js_dir: "js"

standards:
  wcag: "2.2"
  conformance: "AA"
  en_301_549: true

severity:
  critical: error
  serious: error
  moderate: error
  minor: error

thresholds:
  contrast_text: 4.5
  contrast_large: 3.0
  contrast_ui: 3.0
  target_size_aa: 24
  target_size_enhanced: 44
  min_font_size: 14
  min_line_height: 1.5
  max_line_length: 80
  animation_flash_hz: 3
  toast_min_ms: 5000

modules:
  contrast: true
  color: true
  typography: true
  layout: true
  structure: true
  aria: true
  focus: true
  keyboard: true
  interactive: true
  forms: true
  media: true
  language: true
  motion: true
  cognitive: true
  compliance: true
  collector: true

include:
  - "*.php"
  - "js/*.js"
  - "css/*.css"
  - "admin/**/*.php"
  - "api/**/*.php"
  - "includes/**/*.php"

exclude:
  - "uploads/**"
  - "tools/**"
  - ".git/**"
  - "sql/**"

suppressions: []

reports:
  output_dir: "tools/a11y-audit/reports"
  formats: [terminal, json, html]
  keep_history: 10
```

---

## 9. Dependencies

### requirements.txt

```
pyyaml>=6.0
jinja2>=3.1
colorama>=0.4
beautifulsoup4>=4.12
lxml>=5.0
```

### Optional (for --runtime)

```
playwright>=1.40
```

---

## 10. Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| BeautifulSoup over tree-sitter | PHP/HTML doesn't have a mature tree-sitter grammar; BS4+lxml handles mixed PHP/HTML well |
| Regex for JS analysis | Vanilla JS without build step — AST parsing overkill for 5 files |
| Single CSS file assumption | Collector has one `style.css` — no CSS-in-JS, no Tailwind |
| 16 modules (not 21) | Dropped fluentiagrant-specific: seizure (covered by motion.py flash-rate), coga (simpler app), notifications (toast checks in motion.py), preferences (dark mode covered in contrast.py), orientation (mobile-first already) |
| Zero tolerance severity | All levels = error, matching fluentiagrant policy |
| Static-first, runtime optional | PHP needs a running server for runtime — static catches ~80% without infra |
| EN 301 549 coverage | WCAG-mapped EN 301 549 clauses (§5, §7, §9, §11) are implicitly covered by the corresponding WCAG rule modules. The compliance module explicitly covers Annex A (accessibility statement requirements). |
| Own virtualenv | Tool has its own `requirements.txt` and `.venv/` — independent of the project's Python env |

---

## 11. Suppression Format

Suppressions in `config.yaml`:

```yaml
suppressions:
  - rule: "contrast"
    check: "text-contrast"
    file: "admin/index.php"
    line: 42
    reason: "Low-contrast text is decorative watermark, not content"
    approved_by: "marek"
    date: "2026-03-17"
```

Inline suppression in PHP/HTML via comment:

```html
<!-- a11y-suppress contrast:text-contrast — decorative watermark -->
<span class="watermark">Draft</span>
```

---

## 12. Known Limitations

### PHP Echo HTML Detection

The PHP parser strips `<?php ... ?>` blocks and parses the remaining HTML. It also extracts simple `echo "<tag..."` patterns. However, the following PHP patterns produce HTML that **cannot** be statically detected:

- String concatenation: `echo '<div class="' . $class . '">'`
- Variable interpolation: `echo "<span>$name</span>"`
- Multi-line echo with dot concatenation
- `printf()` / `sprintf()` with HTML format strings

These patterns are uncommon in the collector codebase (most HTML is inline, not echo'd). The `--runtime` Playwright bridge catches anything the static parser misses.

### Include Resolution

The parser does not follow PHP `include`/`require` statements. It treats each file independently. `includes/header.php` and `includes/footer.php` are audited as standalone files — violations are reported once per include file, not once per page that includes them.
