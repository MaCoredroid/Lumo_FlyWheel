# Open Threads

- The watermark round was rolled back because unstable snapshot restore ordering corrupted recovery state.
- Adaptive batching is still blocked on replay determinism.
- Retrying watermark work before the prerequisite invariant is fixed would repeat the same incident path.
