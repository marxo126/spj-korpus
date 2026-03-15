-- Per-consent date tracking for GDPR compliance
ALTER TABLE users
  ADD COLUMN consent_service_date DATETIME DEFAULT NULL AFTER consent_service,
  ADD COLUMN consent_biometric_date DATETIME DEFAULT NULL AFTER consent_biometric,
  ADD COLUMN consent_retention_date DATETIME DEFAULT NULL AFTER consent_retention;

-- Backfill existing users: copy consent_date to individual dates where consent was given
UPDATE users SET consent_service_date = consent_date WHERE consent_service = 1 AND consent_date IS NOT NULL;
UPDATE users SET consent_biometric_date = consent_date WHERE consent_biometric = 1 AND consent_date IS NOT NULL;
UPDATE users SET consent_retention_date = consent_date WHERE consent_retention = 1 AND consent_date IS NOT NULL;
