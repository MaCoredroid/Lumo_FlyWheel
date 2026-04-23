# Migration Notes

During the codex-native migration we temporarily kept the legacy helper around for diffing outputs.

That did not make it the live entrypoint. The live workflow stayed on the Make target while the helper remained rollback-only.
