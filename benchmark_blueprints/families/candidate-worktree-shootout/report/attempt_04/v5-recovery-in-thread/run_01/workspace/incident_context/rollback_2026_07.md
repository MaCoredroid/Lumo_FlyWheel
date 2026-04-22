# Rollback 2026-07-14

The previous attempt localized normalization in `cli.py`.
CLI tests passed, but the importer and saved-view repair job kept
emitting unnormalized keys. The hotfix was rolled back during
incident INC-7421 after customer-visible drift in report IDs.
