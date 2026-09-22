# E8 conditional timing harness: independent final review

Date: 2026-09-22. Reviewer: paper_redteam_round1. CPU/source only; no real Docker, model, GPU, inference, or campaign launch by this reviewer.

**Disposition: PASS for the conditional six-cell harness. No unresolved material source/control finding.** This is not a real qualification verdict or permission to skip the parent's separate timing handoff. The two actual qualification receipts from `out-20260922T213142Z-e8-single-logits-v2` must pass the unchanged gates before timing. Their raw evidence is a separate review.

## Exact reviewed snapshot

Stage: `experiments/e8-single-logits-timing-v1`. Local and DGX manifest SHA-256 both `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`; all 33 manifest payload hashes passed. All 23 payloads plus manifest copied from qualification-v2 remain byte-identical to that stage. Required qualification manifest: `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`. The original failed qualification attempt and its launcher bytes remain preserved.

| File | SHA-256 |
|---|---|
| timing_driver.py | 9bd49f360224ceaebd9423c2403d0be18c6b7b14febd53162562804ca29677d6 |
| timing_verify.py | c21dbac3c8feaf0f9dee75a0f6cd2a4d5d83e1ef7cf68332a885c5de79e823be |
| ownership.py | b839a908c1621b1caeb331d4956ab59c2a0bd2973a2e9ba34f6081974a29d659 |
| aggregate.py | 3f62f9b1136bcf23b58b546b40a53b42afa9739c9f037d0ce4be77e5cf167a29 |
| timing_launch.v1.sh | 8458cb6d1e83b93a88bbb6367cb8936b5f325883e3f5fddc463f437b794783d3 |
| prepare.py | 63d03c56680e01009db487e7843aad319f785f772f3c4cbb82b749b5845697f6 |
| test_cpu.py | 33b029d8a50134f0a9a8b5510621d6c6484b1067e01a830dd7ae57e7134d0f53 |
| test_launcher_execution.py | 0a578c8d9b926208042bbb97b0d8db21188220acdbbc30be19462d0d11464e9a |
| README.md | e6899f8ecdd57e1d912024098f8b92c5bce916dab2591deb9345fdb0ca13744d |

## Checked obligations

- **Actual qualification dependency:** `timing_verify.py:26–41` requires both arms, exact qualification manifest, final receipt hashes, reruns the unchanged qualification verifier on each raw arm and compares every raw/source evidence hash. Missing, failed, one-arm, wrong-source or altered evidence cannot authorize timing. `timing_driver.py:124–140` copies qualification evidence and timing sources into a fresh root, revalidates both and re-execs the copied driver. Each cell revalidates qualification before launch and before sealing.
- **Isolated clean computation:** `prepare.py:26–41` derives the launcher from the corrected qualification-v2 launcher; the head helper, shim, candidate assembly and four other loaded modules remain pinned. `timing_verify.py:53–84` checks actual clean eagle hash and engine/route configuration, not only the requested arm flag. Selfcheck, qualification cloning/dispatch and heavy captures are disabled. `timing_verify.py:44–50,92–94` requires closed owned head evidence with ON `5P+0` and OFF `5P+5P` head evaluations for `P` completed proposals; all qualification counters must be zero. Tiny symmetric census instrumentation remains and must be disclosed.
- **API/interval support:** `timing_driver.py:98–111` runs the exact frozen API mapper and E1 joiner after graceful cleanup, then seals only after validation. `timing_verify.py:96–144` binds all ten direct-ID requests, phase/seed/token budgets/prefix hashes, unique engine IDs, exact frozen request order, sequential B1 request times and complete API join. Eight timed prefixes each need at least one retained interval; total floor is eight. Numerator is API-bound emitted tokens on the joined unique pure-decode physical support, not all structural tokens or whole-service throughput. All valid slow intervals remain; the old 1.5-second diagnostic never filters primary support. Insufficient support is preserved without a rate.
- **Ownership:** the early draft only checked idle state; this was repaired before freeze. `ownership.py:14–25,28–70,73–85` preserves before/during/after workload query output, uses `docker top` host PIDs rather than container PID assumptions, requires the exact labeled container and rejects foreign GPU PIDs/containers, absent GPU ownership and failed queries. Driver failure stops the campaign without replacement. Samples are roughly five seconds apart plus query time; the receipt reports the measured largest gap. They cannot prove the absence of short-lived contention between samples.
- **Prospective estimator and stopping:** six immutable cells are OFF/ON, ON/OFF, OFF/ON across three paired blocks. `aggregate.py:12–38` computes the mean of the three relative differences `(ON-OFF)/OFF`, with 10,000 seeded paired-block resamples and the frozen percentile rule. Ratio of arm means is diagnostic. The pre-data supersession of the disabled qualification-manifest provisional primary is explicit. Any missing, failed or insufficient cell suppresses primary estimate/interval; no subset estimate, favorable retry, replacement or precision extension. Three blocks support only coarse uncertainty. `timing_verify.py:147–168` and `aggregate.py:59–76` reject unsealed or changed evidence and campaign failure. Stream comparison is descriptive and cannot change retention.

## Independent execution evidence

1. Re-executed `test_cpu.py` on a temporary exact stage copy, with the archived E1 fixture linked read-only: **41/41 PASS**. Result `/tmp/e8-timing-independent-cpu-20260922.json`, SHA-256 `d8a53399c81929be005e22e197070ea6f82f3508319bb9812f90ef3a3993db1b`. The archived E1 fixture checks reducer compatibility only; it is not fresh E8 performance or qualification evidence. The test explicitly stubs E8 configuration/ownership only for that archived reducer fixture.
2. Independently exercised **37 additional CPU controls** on the settled functions: estimator differing from ratio of means; each of six missing cells; failed/insufficient/NaN/zero/negative-rate suppression; extra block refusal; positive ON/OFF clean census; each arm's wrong counts, qualification flag, open receipt, dispatch/checked counter and missing owner; host-PID positive/foreign GPU/foreign container/empty GPU/wrong header/query failure; positive telemetry bracketing and nonbracketing refusal. All passed. These are synthetic control tests, not measurements.
3. Executed the exact Linux outer and captured inner launchers for both clean arms using strict subprocess stubs. Both passed. Remote report `/tmp/e8-timing-independent-launcher-20260922.json`, SHA-256 `e71bb00a5c5005ee7b844d91c814dbda42943c571de465d26e354f9a89b22e28`. This runs executable shell plumbing rather than only `bash -n`, including clean shim arguments and actual vLLM argument construction. The absent live qualification is explicitly stubbed only in this shell test after real stage validation; real qualification refusal has separate CPU controls.

Exact independent Linux smoke command:

```sh
ssh mark@100.103.10.122 'env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits-timing-v1/test_launcher_execution.py --output /tmp/e8-timing-independent-launcher-20260922.json'
```

## Execution boundary and claim scope

After actual two-arm qualification passes and the parent supplies a fresh timing root, the reviewed command form is:

```sh
python3 -B /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits-timing-v1/timing_driver.py --repo /home/mark/lumo-paper-v2-20260921 --run <fresh-timing-root> --qualification /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T213142Z-e8-single-logits-v2 --execute
```

One existing DGX GPU, serial B1 boots, GPU-memory utilization 0.6, model length 16,384, pinned Qwen3.6-27B FP8 model/image, synchronous eager TREE_ATTN, cache off, policyB 1/0/1, engine and API seed 20260921. Each cell has two 32-token warmups and eight requests of up to 128 tokens; EOS behavior is unchanged. Six boots are expected to take approximately 45–70 minutes based on prior comparable boot/workload durations, not a measured E8 runtime. Per-cell launch/health/workload deadlines are 300/1500/900 seconds.

A passing campaign can support a narrowly instrumented single-logits head ablation on this fixed route/workload. The qualification's same-input full-logit and ordered-candidate checks, unchanged continuation source and clean head census are the relevant causal controls. They do not establish identical cross-boot continuations, full-model state equivalence, other batches/backends, or gains for composed optimizations. No additional experiment is required by this harness review beyond the already agreed qualification and, conditionally, six timing cells. Any qualification invariant failure stops timing; any timing contamination or execution failure is preserved and aborts the campaign without a favorable replacement.
