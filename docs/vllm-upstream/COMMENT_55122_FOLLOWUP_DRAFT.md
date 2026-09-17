# #55122 follow-up (funded item A) — port re-run on 85f61e24b + static launcher-args question — v1 (Codex GO verbatim, 2026-09-17)
> Codex verdict (A_review.md): GO. Placement: ONE inline COMMENT on the added
> `csrc/libtorch_stable/persistent_topk.cuh:1693` call in PR #55122 (the
> `det_select_row` call), containing both paragraphs; no separate general post;
> no merge verdict. Evidence: results/upstream/55122/port_85f61e24b @19510f357.
> Report corrections applied to the agent report, not to this text: §10 is
> "54 fallback cells + 18 control cells + 18 rejection cells"; smem sizing still
> affects allocation ("dead" too broad); no process census → no exclusivity
> claim; the 279-line region hash is not cited.
> AWAITING MARK GO. Body below the separator.

---

Re-ran the standalone GB10 harness against `85f61e24b`, using the unmodified kernel header and transcribed launcher: 324 fallback launches plus 108 cooperative-control launches passed the exact-reference/repeatability checks; 18 above-window cases produced the expected >64-CTA rejection. Separate probes confirmed `force_single_cta=1`, `CACHED=0`. [Harness, hashes and logs](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/19510f3573a4e4568248bc4471a9584e8d818e12/results/upstream/55122/port_85f61e24b). This covers the previous float32 grid, not operator integration or performance.

One source-reading question about the port: `csrc/libtorch_stable/persistent_topk.cuh:1664–1671` now takes six parameters and reads the length without the former row-bound clamp, while `:1755–1760` still supplies eight arguments. At `:1693`, `det_select_row` receives fixed 128 KiB rather than the launcher's computed `smem_size`. Could the wrapper retain the prior `max_seq_len`/`smem_bytes` inputs and clamp? This path is untested here: GB10's 101,376 B opt-in limit cannot select FilteredTopK.

AI assistance was used.
