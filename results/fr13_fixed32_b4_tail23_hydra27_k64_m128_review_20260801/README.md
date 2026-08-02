# B4 Tail23/Hydra27 K64 M128 route review

This is a static code-review artifact. No Docker container, GPU kernel,
SWE-Verified task, timing arm, synthetic probe, TPS measurement, or acceptance
gate was launched during this review.

## Verdict

The original tip `aac82d0b19ffce16a1d69490147c2fa8697be02e` was not safe to
launch because its B4 phase summary mixed per-event and per-step units. The
corrected route is on branch
`agent/fixed32-b4-tail23-hydra27-k64-m128-review` at code commit
`9f30d84dc68f97bfd871862db829b7048e921847`.

Launch only the review branch:

```bash
cd /home/mark/lumoFlyWheel-b4-tail23-hydra27-k64-m128
bash scripts/fr13_run_b4_tail23_hydra27_k64_m128_stack.sh
```

## Findings fixed

### F1 - High: B4 SFWD used a per-event value as milliseconds per step

`fr13_measure.py` defines `s_per_fwd_gpu` per request-event and
`s_per_fwd_gpu_per_forward` per physical pure-decode step. The old summary
multiplied `s_per_fwd_gpu` by 1000 and labeled it `sfwd_gpu_ms_per_step`, while
DFWD and CFWD were already per step. At B4 this understated SFWD and inflated
the residual bucket.

A pre-existing historical B4 deploy-speed record demonstrates the error without
running hardware:

- events/step: `2.607565319121279`
- SFWD per event: `67.47178125284147 ms`
- correct SFWD per step: `175.9370768142467 ms`
- DFWD: `84.12520546273292 ms/step`
- CFWD: `7.6153079912114805 ms/step`
- wall: `276.5625328535598 ms/step`
- corrected other wall: `8.88494258536872 ms/step`

The fixed reducer uses the per-forward field and fail-closes unless SFWD units,
wall units, accepted/committed tokens, full-wall TPS, floor ratio, and the phase
sum all reconcile.

### F2 - High: all-parent production was not bound to its exact4 verdict

The timing runner validated the source/mode/batch production bundle but did not
bind it to the canonical exact4 gate verdict. The review route now requires the
verdict and binds its SHA-256, source commit, topology, physical geometry, K64
block map, task set and marker, mismatch counts, and production-bundle hash.
Both timing arms also require every measured work-census event to report the
actual `fixed32_native_precompute_production_candidate_return` route.

### F3 - High: remote Qwen stdout could be truncated before pipe drain

The prior Hydra-only M128 gate completed all four agents, but one canonical
trace ended at exactly 258,048 bytes without a final newline. The candidate
comparator was byte-exact across 320 real records, but the campaign correctly
withheld its production pass because the task trace was incomplete.

Fixed32 remote launches now precreate a private, single-link regular file and
redirect Qwen stdout to `/out/qwen_trace.jsonl` inside the container. Docker
stdout is discarded, the remote file identity and digest are observed before
and after transfer, and exact local bytes are installed before strict JSONL
validation. Malformed evidence remains intact and non-fixed32 launch behavior
is unchanged.

### F4 - High: B4 timing rejected the mandatory terminal census record

The old timing parser treated every v9 work-census record as an event and
required a TAW route on each one. A valid final flush always appends one
terminal record without a TAW section, so every otherwise valid timing arm
would have been rejected.

The timing route now revalidates events and the mandatory terminal from the raw
census bytes, rederives and exact-compares the persisted arm report, binds raw
SHA-256 and byte counts, and requires one mode-neutral physical-work signature
across Tail23/Hydra27 and stock/M128. The invalid launch source was stopped
before any timing arm and is documented by a reduced abort artifact.

## Reviewed contracts

- Tail23: `tail6_fixed32`, mask `0x7a9ce7ff`, 23 active draft rows.
- Hydra27: `hydra27_fixed32`, mask `0x7abdffff`, 27 active draft rows.
- Physical geometry: 31 draft slots plus root, B4 projection `M=128`.
- Draft vocabulary: root enabled, `K=65536`, block-map SHA-256
  `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`.
- Verifier vocabulary: full.
- Qwen Code: `0.19.4`, pinned turn tool-call cap `256`, bundle-tree SHA-256
  `594cac41e2d5ed505e0646f318b263ff70e200bcffe97326fe1c042fdc220516`.
- Candidate binary SHA-256:
  `895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f`.
- Canonical real SWE-Verified exact4 tasks: Astropy `12907`, `13033`, `13236`,
  and `13398`.
- Timing arms share the K64 settings, topology, tasks, stock FA2, all-parent
  committer credential, timers, graph mode, and harness. The intended runtime
  delta is stock CUTLASS versus persistent M128.
- Each child stage owns attested container teardown. The following stage
  requires an empty Docker state, and the timing runner refuses to continue
  after a non-clean stock arm.

Exact4 remains a tuning screen. It is not formal exact16/U95 hardware-floor
acceptance, and this artifact contains no new performance number.

## Static verification

- focused suite: `72 passed`
- broader provenance, ingress, floor, exact-commit, Qwen, CUTLASS, and
  all-parent suite: `313 passed, 1 skipped`
- post-capture integration suite: `322 passed, 1 skipped`
- post-census integration suite: `322 passed, 1 skipped`
- B4 timing-math regression tests: pass
- Bash syntax and four embedded Python blocks: pass
- Ruff, Python byte compilation, and `git diff --check`: pass
- fixed32 work-census self-test: pass, 167 tamper tests
- depth-acceptance self-test: pass
- runtime-manifest canonical SHA-256:
  `8996529e514caf31584c0701a8661c0cb4d3fece6666352e0c09d3a85ea40b2c`
- external-manifest canonical SHA-256:
  `c42fb16d2ea932f0e819c81172c49b69b312b75dd446146b6d7b6ef78af00a9e`
