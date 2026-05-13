
# Access Policy v2.1

Scope: third-party access to customer replay traces.

Rules:
1. Raw production replay access for third parties is forbidden.
2. A redacted replay package in the vendor sandbox may be approved only if:
   - privacy review signs off,
   - security review signs off,
   - the ticket includes an audit log reference,
   - the exception expires within 24 hours,
   - the path is least-privilege and does not mint a direct production token.
3. During any release freeze that explicitly suspends vendor replay exceptions, no new replay exception may be approved.
