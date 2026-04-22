
# Current Request Resolution Runbook

For customer replay requests from third parties:
- Start from access-policy v2.1.
- Prefer the narrowest compliant path.
- If the request can be met with a sandbox-only redacted export plus privacy/security approval, do that instead of a raw token.
- If a freeze notice or incident rollback suspends replay exceptions, fall back to the sanitized sample path.
