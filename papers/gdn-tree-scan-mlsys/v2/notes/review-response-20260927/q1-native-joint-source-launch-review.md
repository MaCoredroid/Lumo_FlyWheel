# Prospective joint-source launch preparation review

2026-09-29. Source/CPU review only; no network, Docker, model, GPU, memory action, campaign launch or gate edit. The accepted joint-capture implementation is treated as a reviewed dependency; this pass covers its new source-plan/CLI/config/patch-apply/launch connections. All writes were review artifacts or temporary synthetic source/receipt copies.

**Final disposition: PASS for the bounded source/CPU launch-preparation scope. All three reported findings are closed below. This is not launch authorization or qualification evidence.**

## Scope and preparation checks

- `tools/q1_joint_source_plan_v1.py:17–30` starts from the unchanged 84 calibration cases and deterministically selects the three `root-only` cases, exactly one per prefix, A/r0. Every one of the 84 cases has an explicit same-prefix/root-token/root-position mapping. The source inventory is three requests at R1, explicitly **not** the 84 × A/B × R2 qualification denominator. Failure stops the source inventory; no outcome-based subset/retry is introduced. JOB canonical hashing includes MTP policy and plan binding.
- `q1_spec_off_joint_source_config_v1.arm_config` preserves the accepted target serving argv, environment, model/image, resources and existing read-only mounts. Only the applied runner-patcher path and declared instrumentation change; target scheduling stays spec-off while the reviewed separate native MTP owner participates. No common-O0 import or old-source run mount is carried into this natural joint-source inventory.
- `q1_apply_native_mtp_joint_source_v1` connects the reviewed `q1_patch_native_mtp_joint_source_v1.patch_text` to the actual launch CLI. Generated worker source compiles and has SHA `205799b54ec8bbae7641f8f5e47f88d78d43adbe8b4a512c0866bc604274db10`. The five reference anchors plus reviewed owner/history metadata hooks are preserved.
- The new launcher uses `Q1-NATIVE-JOINT-SOURCE`, its own output namespace/snapshots, exactly one process A / three requests / R1, source-plan reconstruction, merged dependency hashes, the new job/config/driver, and inherited exact-CID cleanup. The driver's raw target/MTP/joint checks remain connected and stop before the next request on invalid evidence. A completed source inventory alone cannot qualify either common-source continuation or full Q1.
- Independent execution of the supplied nine CPU tests passes on temporary copies (9/9, 0.096 s). It includes the actual plan/job/config constructors, generated patch readback/no-reuse and foreign-source refusal, shell syntax, and the launcher's early no-action `--dry-run` branch. No post-gate launcher action was executed.

## Findings and closure

1. **Closed: one reachable runtime dependency initially unbound.** `q1_native_mtp_same_input_v1.py:10` imports `raw, ints` from `q1_native_mtp_history_v1`, initially omitted from the new source plan's hash inventory. Parent added it. Independent transitive AST traversal now finds zero unbound local modules across the actual plan/config/hook/driver/patch chain.
2. **Closed: actual patch receipt initially only required existence/JSON syntax.** Parent expanded the pre-request validator to require the exact joint schema, source/patched SHA, in-place flag, all anchor counts, five marks and full expected native-MTP patch detail, recomputed from the pinned source/patcher. Independent execution of this extracted pure JSON validator accepts the matching receipt and refuses `{}`, wrong source SHA, wrong patched SHA, false in-place flag and four marks (all rc6). An initial reviewer call omitted the newly added TOOLS argument and failed before validation; that harness error is retained and is not a finding against the repaired validator.
3. **Closed in final recheck below: CPU `--out` could overwrite its own source.** On the original adapter SHA `1d35832f59c01ef8f009f14e9ba5cc5229cc9890d2f4e5f2a3a4667ab43b680e`, `apply(source, receipt, out=source, in_place=False)` overwrites a temporary copy of the pinned original while recording `applied_in_place=false`. CLI `--source X --out X` bypasses the `--apply` container guard. `:10–14` checks existing output only when `target != source`. Refuse a non-in-place output resolving to the source (including alias paths) before any write, then retain the explicit container-only apply route. This does not affect the current launch's correct `--apply` arguments; it repairs the new wrapper's copy/apply contract.

## Evidence currently retained

- `p0/monitor/review-response-20260927/joint-source-launch-review/INITIAL-CPU-CHECKS.json`: `6dca77fb8246af833c5492c734eceb0614bc2997ea862a0ced84592a50c2a8fb`
- `p0/monitor/review-response-20260927/joint-source-launch-review/CPU-OUT-ALIAS-CONTROL.json`: `3a0cffbea54e4305b57e7ce06d70f60417e355203eafaf7221458908cdca0a1c`
- `p0/monitor/review-response-20260927/joint-source-launch-review/RECEIPT-AND-DEPENDENCY-CLOSURE.json`: `caef9f4239b42d3da81a754cc90b5f118cdff75a45ce1e15db5388176fba3644`

Initial source copies are retained under the same review directory. Final corrected hashes and disposition follow; no source-plan freeze or launch authority is created by this review.

## Final narrow repair closure — 2026-09-29

The adapter now rejects non-in-place outputs when resolved paths match or an existing destination is the same file, and refuses all other existing copy destinations before writing. Independent controls on temporary pinned-source copies show same-path, symbolic-link and hard-link aliases all refuse, preserve source bytes, and create no receipt. The full supplied suite independently passes **10/10** (exit 0), including positive patch-copy/receipt generation and no-action shell dry-run. Only the adapter dependency changed since the preceding dependency/receipt closure; all other bound implementation files stayed unchanged. No new scope, model setting, acceptance criterion or generated worker byte changed.

Corrected adapter SHA `571182665a140bbd8ed12d040d0e46ca8504b3bf304d7871ae9bcdd4fcc93264`; test SHA `98d1b77ed6a0d698e0766b6fd9556a6061cdc039f1a6f629a12aa1aac89f1dc2`. Final evidence: `p0/monitor/review-response-20260927/joint-source-launch-review/FINAL-CPU-CLOSURE.json`, SHA `4d0848458c8ed5a25b1b5133482a13e30400ddc5479e6872fb26602ad22a7efe`. The earlier alias reproduction and initial controls remain preserved.

| Bound source role | Final SHA-256 |
| --- | --- |
| `corpus_tool` | `039809d05865eb38756a595728b1bc285f453bc5b46b4e283e2d52836794aa4b` |
| `launcher` | `dd41001579d994348d09ea3e2c72e307cd7764661efeadfad66c04cad42618b9` |
| `patcher` | `571182665a140bbd8ed12d040d0e46ca8504b3bf304d7871ae9bcdd4fcc93264` |
| `config` | `4fb40c97adbbb5c25b5a8b025a1c596f89599690774cda0b25df4632a7c99950` |
| `hooks` | `22edc9f985203a9a1f4c76744297ebfbfc475f06b0b678133a3264076c798414` |
| `driver_v2` | `3ea559dd482a09862be15afcf611b3853adf8dca83ce48abd3821e30179a9c8f` |
| `q1_patch_native_mtp_joint_source_v1` | `cc37c6cae13512f736af8444f0948db9a8920ecf36c0c8213b7561201e85f427` |
| `q1_native_mtp_history_v1` | `d531fea5376ec263217c45636fb1661bfaf000792d9a18db8512c407adfa2d13` |
