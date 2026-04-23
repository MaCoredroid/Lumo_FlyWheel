# Release freeze note

This hotfix ships inside a partner billing freeze. Preserve nested
exports under the tenant root because downstream invoice replays
rely on `reports/partner-billing/<month>/...`.
