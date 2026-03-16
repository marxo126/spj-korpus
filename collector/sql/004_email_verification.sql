-- Email verification support
ALTER TABLE users
  ADD COLUMN email_verified TINYINT(1) DEFAULT 0 AFTER email,
  ADD COLUMN email_verify_token VARCHAR(64) DEFAULT NULL AFTER email_verified,
  ADD COLUMN email_verify_expires DATETIME DEFAULT NULL AFTER email_verify_token;

-- Existing users are already verified (they registered before this feature)
UPDATE users SET email_verified = 1;
