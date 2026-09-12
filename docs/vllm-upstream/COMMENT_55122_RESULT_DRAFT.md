# #55122 low-smem fallback result comment — v3 (Codex GO; nits applied; logs added to artifact, URL re-pinned)

> Funded P4′ run (Mark, 2026-09-11). Codex GO on this text; NEEDS MARK GO TO POST as a PR comment on #55122. Artifacts: https://github.com/MaCoredroid/Lumo_FlyWheel/tree/a4c38b9c1d3830ad9128186932497edf8d613e58/results/upstream/55122

---

At `7cfd04a3` (kernel sources unchanged at `a7188289e`; only the test file changed), a standalone GB10 harness using the unmodified PR kernel header and a transcribed launcher exercised the low-shared-memory overflow fallback (48 SMs, 101,376 B opt-in shared memory). This ran alongside another workload.
For contiguous float32 rows with stride=length, rows {1,4}, k {512,1024,2048}, and random, tie-heavy and all-equal inputs, widths {355588,400000,474112} logged `force_single_cta=1`. The launch parameters and source select the uncached `det_select_row` path. All 324 fallback launches matched a stable value-descending/index-ascending reference, with selected indices sorted ascending, and were identical across six repeats per case; no output poison remained. Width 355584 added 108 passing cooperative-control launches; 474116 produced 18 expected pre-launch >64-CTA rejections.
Harness, source hashes and logs: https://github.com/MaCoredroid/Lumo_FlyWheel/tree/1536dae35436ec11de78367cccbf14e5dd36950c/results/upstream/55122. This supports exactness and repeatability for these cases on one device; it does not validate torch/vLLM operator registration, CUDA graphs, other streams or broader determinism. No performance claim.

These measurements exercise the stock fallback conditions you described, using this PR's `force_single_cta`/uncached `det_select_row` path at our tested widths.
