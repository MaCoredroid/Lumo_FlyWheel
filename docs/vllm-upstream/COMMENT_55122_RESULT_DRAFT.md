# #55122 low-smem fallback result comment — v2 (Codex text + head note + one bridge sentence after jschmied's 2026-09-12 replies)

> Funded P4′ run (Mark, 2026-09-11). Codex GO on this text; NEEDS MARK GO TO POST as a PR comment on #55122. Artifacts: https://github.com/MaCoredroid/Lumo_FlyWheel/tree/a4c38b9c1d3830ad9128186932497edf8d613e58/results/upstream/55122

---

At `7cfd04a3` (kernel sources unchanged at `a7188289e`; only the test file moved), a standalone GB10 harness using the unmodified PR kernel header and a transcribed launcher exercised the low-shared-memory overflow fallback (48 SMs, 101,376 B opt-in shared memory). This ran alongside another workload.
For contiguous float32 rows with stride=length, rows {1,4}, k {512,1024,2048}, and random, tie-heavy and all-equal inputs, widths {355588,400000,474112} logged `force_single_cta=1`. The launch parameters and source select the uncached `det_select_row` path. All 324 fallback launches matched a stable value-descending/index-ascending reference, with selected indices sorted ascending, and were identical across six repeats per case; no output poison remained. Width 355584 added 108 passing cooperative-control launches; 474116 produced 18 expected pre-launch >64-CTA rejections.
Harness, source hashes and logs: https://github.com/MaCoredroid/Lumo_FlyWheel/tree/a4c38b9c1d3830ad9128186932497edf8d613e58/results/upstream/55122. This supports exactness and repeatability for these cases on one device; it does not validate torch/vLLM operator registration, CUDA graphs, other streams or broader determinism. No performance claim.

This is the routing case in the note above: at these widths the launcher took the `force_single_cta` path rather than `top_k_per_row_decode`, and it was exact and repeatable there.
