# Q1 replay publication observer — bounded independent review

Disposition: **PASS for the corrected source/CPU observer scope**. One concrete cross-witness gap was found and repaired during this review. No live candidate run, numerical SSM qualification, MTP qualification, or launch approval is established.

## Finding and closure

The initial raw-audit call supplied only observation ID and forced nodes to `RW.audit`. Its expected staged SSI was derived exclusively from its own replay `source_before.spec_idx`; no named layer order or convolution/O0/O1 bridge existed. Two independent stdlib counterexamples consistently rehashed (a) a changed running-row SSI and matching staged SSI, and (b) swapped layer rows and matching staged SSI. Both passed the standalone replay audit despite disagreeing with unchanged same-case convolution/O0 rows. These counterexamples remain executable in `controls.py`.

The repaired helper `q1_replay_publication_witness_v1.py:137` adds `bind_convolution`: it compares exact names/order and observation IDs, both complete 48×1×32 SSI byte snapshots against authenticated convolution `after_inputs`, and each named running row against both O0 and O1. The independent controls now show both original counterexamples refused by this binder. Foreign observation, layer-order swap, and changed O0/O1 rows are also refused. The standalone audit explicitly returns `running_row_binding_checked=False`; callers must retain the separate binding result.

The connected raw audit calls the binder immediately after the replay audit at `q1_candidate_raw_audit_v4.py:189`, after snapshot/convolution authentication. Both live replay seams call `_replay_operands` (`q1_candidate_hooks_v5.py:505`), which requires the completed same-case conv witness, identical registry order, full live SSI equal to conv after-inputs, named O0 rows, and the identical SSM-bank tuple from the production fast route. Thus this closure applies to the actual collection path and its offline interpretation; it does not rely on intended configuration values.

## Production route and buffer checks

- Kernel preseed initializes `reference_graph=None` at 15164. With `layer_batch=False`, the native captured graph is assigned to `state['graph']` at 15360–15370 while `reference_graph` stays None. The observer's admission check is correct for this route. It rejects alternatives and does not change flags.
- Kernel 15067–15138 allocates the observed fixed16 staging shapes. Graph-body 14602–14655 copies root plus the accepted path into k/v/a/b buffers and neutralizes the tail: BF16 `a=-9984` (rounded -10000, bytes `1c c6`), other inputs zero. SSI repeats running column zero. The pure byte oracle matches this source, including the root-only case.
- Generated GDN 10243 constructs dt_bias through the default parameter dtype and explicitly creates FP32 A_log at 10246. The observer accepts and preserves observed BF16 or FP32 dt_bias, with FP32 A_log; it does not cast evidence. Independent raw positives cover both allowed dt_bias types. Actual dtype still needs observation at a live admitted run.
- Source rings, SSI, previous lengths, and parameters are compared across the original replay. Every source/staging tensor's identity must remain stable. The only permitted source mutation is the explicit `flags[:,0]` clear; flags column one is checked unchanged. Source/staging captures force CPU readback, so the observed post bytes are not merely an enqueue counter.
- The patcher places the before callback immediately before `_fixed_replay` and the after callback immediately after the existing flags clear. An independent AST comparison removed only the four conv/replay observer calls and recovered the original production route exactly. The actual replay call and its arguments are unchanged. Existing production fast-state validation continues to bind captured inputs, while the observer also checks actual fast-route object identity and graph stability.

The coverage claim remains bounded: full B1 ring capacity and fixed16 logical staging tensor elements, source preservation during replay, and running-row binding. It does not prove earlier scan production of the rings, SSM recurrence numerics, entire allocator storage including padding, or deferred MTP publication.

## Controls and retained evidence

Independent configured-Python run: **34 stdlib controls**, 8 PASS and 26 expected REFUSED. The PASS count deliberately includes the two reproduced standalone counterexamples; the repaired binder refuses both. Controls include raw positives, authenticated/rehashed source and staging corruption, missing source, wrong replay counts, changed tensor identity, wrong route/observation, the repaired cross-witness cases, actual AST-extracted hook admission, and production-route AST preservation. No Torch/model/GPU import was used by these controls.

The configured local interpreter lacks Torch. The attempted parent unittest run failed at import, preserved in `parent-tests.txt`; this is a dependency limitation, not an implementation test failure. The parent separately supplied settled `test_log.q1_replay_publication_witness_v1.attempt3.txt`: 3 methods pass in 1.200 s, including real updated hooks around AST-extracted production staging with the native SSM operation mocked and the consistent wrong-SSI counterexample. I read and retained this log but did not independently rerun its Torch tests or treat it as numerical evidence. The earlier independent control draft stopped because parent repairs added new hook prerequisites during the review; `controls.attempt1.py` is preserved, and the updated final controls completed.

No remaining concrete blocker was found in this bounded corrected observer scope. Source-freeze/launcher wiring and live collection remain parent-owned, unobserved work.

## Exact source bindings

- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_replay_publication_witness_v1.py` — `3d7d9d56310442058816b04494b105eb7339ae1620eee4a9b2bb1890275bce63`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_hooks_v5.py` — `cb0f5bb6ab6476d69204709e6f28b2c85189a44e14ebb17388b3cb12dff0ad77`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_patch_candidate_v5.py` — `92e2743ac4f3c85daefe752901e3815edb7fa0d09da92cea8cfaec79b9c643d0`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_raw_audit_v4.py` — `48f895ecd22af6652787570e47f4ded793a6c23482af17bfb9e25aa606679786`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_job_v5.py` — `5bdd1035a449c7fec3365a0a8da2c2e9de139f6680cc60e44608a5113f97b026`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_target_kv_witness_v1.py` — `311b047bcc1bc59809d89f71baf27dc0f178c065d20d48358e408332a081d0d8`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_conv_publication_witness_v1.py` — `f6c62159852257ee2ff915209e1e46dd39eabb379cccb2a8fddcb054f01dcd23`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_q1_replay_publication_witness_v1.py` — `e1ee28c97b6e574804b90d7b96a4494433c243554a0575de0826beca8c0ee819`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/test_log.q1_replay_publication_witness_v1.attempt3.txt` — `4fd548e045b626274a43b1d401d5a11fe405d1154c9330d19dc6e561b4f4e68e`
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` — `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/generated_source/probe-20260928T035847Z/logs/generated/rejection_sampler.patched.py` — `7e4691a6e48fdd9426117beb9b91ef1fdd1053578f792b4d670234efba7b0e52`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/generated_source/probe-20260928T035847Z/logs/generated/gdn_linear_attn.patched.py` — `23df7748f02a742753e586487e3e905e0ffb23815a9c5aaf87f5cf2c4a2ff91a`

All reviewed bytes are copied under `p0/monitor/review-response-20260927/replay-witness-independent-20260929T0400Z/sources/`; `SOURCE-BINDINGS.json` maps original paths and `MANIFEST.json` seals the review payload.
