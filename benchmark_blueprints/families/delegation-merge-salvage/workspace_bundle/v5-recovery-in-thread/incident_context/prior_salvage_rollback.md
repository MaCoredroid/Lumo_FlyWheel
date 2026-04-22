# Prior Salvage Rollback

The previous recovery attempt adopted Worker A's renamed JSON keys during a hot
fix. That change was rolled back within one hour after downstream automation
started treating every report as empty. Do not re-select the rename path.
