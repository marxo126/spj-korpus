# Data Protection Impact Assessment (DPIA)

**Project:** SPJ Collector — Video Collection for Slovak Sign Language Corpus
**Version:** 1.0
**Date:** 2026-03-13
**Review schedule:** Annually or when processing changes significantly

---

## 1. Processing Activity

Collection of video recordings of individual sign language signs from deaf, hard-of-hearing, and hearing signers of Slovak Sign Language (SPJ). Videos contain the signer's face (biometric data), hands, and upper body. The collected data is used to build a research corpus (SPJ Corpus) and to train AI models for sign language recognition.

---

## 2. Data Controller

**Innosign**
Contact: data@spj.sk

---

## 3. Data Types Processed

| Data type | Category | Source |
|-----------|----------|--------|
| Email address | Personal data | Registration form / Google OAuth |
| Display name | Personal data | Registration form / Google OAuth |
| School for the deaf | Personal data (sensitive context) | Registration form |
| City / region | Personal data | Registration form |
| Age range | Personal data | Registration form |
| Gender | Personal data | Registration form |
| Dominant hand | Personal data | Registration form |
| Video with face | **Special category — biometric (Art. 9)** | Camera recording |
| Pose landmarks | Derived data (skeletal points from video) | Automated extraction |
| Recording metadata | Personal data (linked to user) | Application |
| Validation votes | Personal data (linked to user) | Application |

---

## 4. Legal Basis

**Explicit consent** under Art. 6(1)(a) and Art. 9(2)(a) GDPR.

Three separate consents are collected at registration:

1. **Service/Account consent** — Processing of personal data (email, name, demographics) for account creation and research purposes.
2. **Biometric/Video consent** — Recording and processing of video containing the signer's face (biometric data under Art. 9).
3. **Retention consent** — Retention of anonymized videos and derived pose data in the research corpus after account deletion.

Each consent is tracked individually with a timestamp recording when it was given (GDPR audit trail).

---

## 5. Necessity and Proportionality

**Why video with face is essential:**

Sign language uses facial expressions as grammatical markers (non-manual signals). These include:
- Eyebrow raises/furrowing (questions, conditionals)
- Mouth patterns (mouthings of Slovak words, mouth gestures)
- Head tilts and nods (negation, affirmation)
- Eye gaze direction (pronominal reference, verb agreement)

Without facial data, sign recognition accuracy drops significantly because grammatical meaning is lost. Hands-only recording would produce an incomplete and scientifically unusable corpus.

**Data minimization measures:**
- Pose landmark extraction reduces video to 543 skeletal points (33 body + 42 hand + 468 face), removing visual identity while preserving linguistic information.
- Original video is needed for model training but pose-only data is used where possible.
- Only short clips (approximately 5 seconds per sign) are recorded, minimizing exposure.
- Demographic data collected is limited to research-relevant fields (school/dialect, region, age, gender, dominant hand).

---

## 6. Risk Assessment

| Risk | Impact | Likelihood | Overall | Mitigation |
|------|--------|------------|---------|------------|
| Video data leak — person identified from face | HIGH | MEDIUM | HIGH | HTTPS everywhere, access controls, server security hardening, database credentials isolated |
| Re-identification from pose landmarks | LOW | LOW | LOW | Skeletal data lacks sufficient biometric detail for identification; 543 coordinate points do not preserve facial identity |
| Unauthorized access to account data | MEDIUM | MEDIUM | MEDIUM | bcrypt password hashing, CSRF protection, session management with regeneration on login, rate limiting on login attempts |
| Data retained after account deletion without consent | MEDIUM | LOW | LOW | Explicit retention consent obtained separately; account deletion anonymizes user link (user_id set to NULL on recordings) |
| Google OAuth users bypass consent | HIGH | HIGH | CRITICAL | **Mitigated:** Consent guard in require_login() redirects users without consent to consent.php; consent page blocks access to all protected pages |
| Breach notification impossible (fake emails) | MEDIUM | MEDIUM | MEDIUM | Email verification implemented; unverified email banner shown to users |
| Minor (under 16) registers without parental consent | MEDIUM | LOW | LOW | Age range selection shows warning for under-18; terms require parental consent for minors under 16 |

---

## 7. Safeguards Implemented

### Technical measures
- **HTTPS everywhere** — all data in transit encrypted
- **bcrypt password hashing** — passwords stored as bcrypt hashes, never plaintext
- **CSRF tokens** — all state-changing operations protected against cross-site request forgery
- **Session regeneration** — session ID regenerated on login to prevent session fixation
- **Rate limiting** — login attempts rate-limited (10 per 15 minutes)
- **Input validation** — email validation, password minimum length, server-side form validation
- **Prepared statements** — all database queries use PDO prepared statements (SQL injection prevention)

### Organizational measures
- **3-part granular consent** — separate consent for account, biometric data, and retention
- **Per-consent timestamps** — each consent recorded with individual date for audit trail
- **Cookie consent banner** — only session cookies used, consent obtained before setting
- **Account deletion** (GDPR Art. 17) — user can delete account; personal data removed, recordings anonymized
- **Data export** (GDPR Art. 20) — user can download all personal data as JSON
- **Public accessibility statement** — EU Web Accessibility Directive compliance
- **Terms of service** — clearly state data processing purposes and user rights

---

## 8. Data Retention

| Data type | Retention period | Justification |
|-----------|-----------------|---------------|
| Account data (email, name, demographics) | Until user deletes account | Service provision |
| Videos | Indefinitely for research corpus | Explicit retention consent obtained; scientific research exemption (Art. 89) |
| Pose landmarks | Indefinitely for research | Derived from video; lower privacy impact |
| Recording metadata | Indefinitely (anonymized after account deletion) | Research corpus integrity |
| Session data | Until logout or session expiry | Service provision |
| Password reset tokens | 1 hour | Security (auto-expired) |

---

## 9. Data Subject Rights

| Right | Implementation |
|-------|---------------|
| **Right to access** (Art. 15) | Profile page displays all personal data; data export endpoint provides complete JSON download |
| **Right to rectification** (Art. 16) | Profile edit form allows updating name, school, location, age, gender |
| **Right to erasure** (Art. 17) | Account deletion button; personal data removed, recordings anonymized. Full video deletion available on request (data@spj.sk) |
| **Right to data portability** (Art. 20) | JSON export endpoint downloads all personal data, recordings metadata, and validations |
| **Right to withdraw consent** | Account deletion removes all consents; for video-only withdrawal, contact data@spj.sk |
| **Right to restrict processing** (Art. 18) | Contact data@spj.sk |
| **Right to object** (Art. 21) | Contact data@spj.sk |

---

## 10. Supervisory Authority and Consultation

**Supervisory authority:**
Úrad na ochranu osobných údajov SR (ÚOOÚ SR)
https://dataprotection.gov.sk

If residual risk remains high after implementing all safeguards listed above, formal consultation with the supervisory authority under Art. 36 GDPR will be initiated before processing begins.

---

## 11. Review Schedule

This DPIA will be reviewed:
- **Annually** (next review: March 2027)
- When processing activities change significantly (new data types, new purposes, new data sharing)
- When a data breach or security incident occurs
- When technical infrastructure changes materially

---

## Approval

| Role | Name | Date |
|------|------|------|
| Data Controller | | |
| DPO / Legal | | |

*To be signed before production deployment.*
