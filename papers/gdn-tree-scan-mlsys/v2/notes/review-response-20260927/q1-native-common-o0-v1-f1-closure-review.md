# Native common-O0 F1 closure

**PASS for the bounded prepared-metadata repair**, on adapter SHA `d79e35b6a082181e307ce8229a2927a90e17707f99cf1032c74c32cc958c8620`. This closes F1 in the prior implementation review. Source/CPU readiness only; no launch, GPU result, numerical qualification, MTP qualification, or gate approval is implied. The unchanged H4 implementation and previously accepted source mapping were not re-audited.

The adapter now checks actual `runner.seq_lens` (a direct Tensor in the pinned native image), active value P+1; CPU/GPU query starts [0,1]; all 48 GDN metadata counts for exactly one native decode/no prefill/no speculation; and both per-layer non-spec query arrays. `q1_native_common_o0_v1.py:79–86,116–120` enforces these before H4 planning/mutation. Lines 140–159 include prepared values/tensor identities and metadata counts/query identities in the saved identity; lines 167–170 rerun the same validation after copying.

The first displayed repair had `runner.seq_lens.gpu`, inconsistent with the pinned `gpu_model_runner.py:684–686`, which constructs seq_lens using `torch.zeros` directly. This was reported and corrected before the independent control run; the fixture was also corrected to a direct tensor. The initial copy's provisional filename `PRE-CORRECTION-SNAPSHOT.json` is misleading: its actual adapter hash already equals the corrected d79e35b6 bytes because the copy overlapped the quick repair. No snapshot of the transient `.gpu` source is claimed; the original displayed source and message retain that finding.

25 focused independent controls executed actual AST-extracted adapter/import/hook functions with metadata facades, without Torch, tensors containing model data, GPU, or H4 copying. The valid exact-native interface passes. Missing/stale sequence, invalid CPU/GPU query intervals, missing/wrong GDN query arrays, and each of the six decode/prefill/spec counts refuse. The earlier hook-level false admissions now fail in the connected root hook's actual import path before the simulated mutation callback, then seal invalid, latch unusable, and raise. Valid root import succeeds. After the simulated copy, changed sequence/query/count values refuse; replacing the sequence tensor with a different object of identical values also refuses the identity comparison.

The helper `_check_prepared` itself is unchanged and does not independently enforce these sequence/query checks. Closure is specifically the root common-O0 import path and its post-copy identity recheck. No new claim is made about independent metadata attestation on later continuation steps. Existing actual-token/all-position checks and base continuation evidence are unchanged.

The parent's supplied attempt2 log reports six native-shape Torch CPU methods passing; it is parent-performed evidence, not a local rerun. The independent controls use the corrected actual runner interface and include the previously missed direct-Tensor positive control. Historical F1 evidence is retained separately.

## Evidence hashes

- `p0/monitor/review-response-20260927/native-common-o0-v1-f1-closure-review/corrected-q1_native_common_o0_v1.py`: `d79e35b6a082181e307ce8229a2927a90e17707f99cf1032c74c32cc958c8620`
- `p0/monitor/review-response-20260927/native-common-o0-v1-f1-closure-review/corrected-q1_reference_hooks_common_o0_v1.py`: `9eaedce27d00ac6206bb4ee2f1a3e4efa172828457bb5fcb174ec3dc0ea1fb21`
- `p0/monitor/review-response-20260927/native-common-o0-v1-f1-closure-review/f1_closure_controls.py`: `a6608af16ff44edcd11fdffc7d1c406464709b3c032243a6d004309ab543313d`
- `p0/monitor/review-response-20260927/native-common-o0-v1-f1-closure-review/INDEPENDENT-F1-CONTROLS.json`: `c7e359ef8c8913b9e930ed1b69ed07dcb436e180e91a89326f6bc3778e093ad9`
- `experiments/review-response-20260927/tools/tests/test_q1_native_common_o0_v1.py`: `c6defad5d86db0395804b7c43e2b5faa48b74034d89be939829102c153652128`
- `experiments/review-response-20260927/tools/test_log.native_common_o0_v1.attempt2.txt`: `47046e7ae53d7d6b59023cc5bb5fd8e8cb2e7a9fde310611645e6c5c2b8d01aa`
