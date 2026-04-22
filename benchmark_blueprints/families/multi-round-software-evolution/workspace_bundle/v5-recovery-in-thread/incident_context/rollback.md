# Rollback Incident

- The previous round selected `streaming-watermark-alignment`.
- Deployment was rolled back when the watermark fix hit unstable snapshot restore ordering and corrupted recovery state on replay.
- Incident note: "Do not retry watermark alignment until snapshot ordering is stable; otherwise the same replay corruption path remains live."

This is the critical recovery-in-thread signal for the variant.
