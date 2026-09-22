# Round 18: corrected p072 compact-commit diagnostic

The bounded three-boot experiment now provides a real publication/consumption witness and a small downstream numerical effect. None and the matched commit sham are bitwise equal on the recorded tensors. The compact-commit arm changes the authoritative recurrent state, the next scan reads that exact published state, and two aligned forward calls show downstream logit differences. All 32 emitted token IDs, API logprobs, accepted counts and all recorded per-row argmaxes remain unchanged.

These observations support this one-prefix, one-layer diagnostic. They do not establish general numerical harmlessness, whole-model E2, convolution/KV publication correctness, or service performance. No further compact-method experiment is necessary to report this bounded result. Sequential E2→E1 remains a separate route.

## Scope and completion

Remote root: `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T071945Z-e7b-r16-p072-commit-batch`.

| Arm | Directory |
|---|---|
| None | `e7b_001_p072_arm1_none-B_fs_ieee-all` |
| Matched B commit-only sham | `e7b_002_p072_arm2_sham-B_fs_ieee-all-commit_only` |
| B commit-only | `e7b_003_p072_arm3_commit_only-B_fs_ieee-all` |

All three boots completed; `loop.log` ends with `E7B LOOP DONE: 3 boots`. Each has 51/51 recorded provenance checks and 8/8 operand-gate checks passing. The independent raw audit below also verifies payload identity, topology/path structure, state publication/consumption, and pairwise recorded inputs instead of relying only on gate summaries.

The selected layer is `language_model.model.layers.62.linear_attn`; the route is p072, Qwen3.6-27B, B=1, eager TREE_ATTN, ten real nodes/padding 16, B forward substitution with IEEE dot precision, fp32 recurrent state. All arms actually load `capture_operands=true` and `capture_state=true`. Requested specs equal the runtime's `e7b_spec_loaded.spec`; the sham declares `shadow=commit_only`. All 16 script-snapshot files, including runtime/driver/kernels, have identical hashes across arms.

I performed CPU-only reads/reductions with CUDA hidden and bytecode writes disabled. No inference, container launch, tmux action, source edit, or additional experimental cell was performed. Boot 3 completed while the initial raw audit was being prepared; it was included only after both gate artifacts existed and passed.

## Independent raw comparison

There are 13 operand files, 13 state files, 13 hidden captures and 13 logits captures per arm. Each operand/state pair is correctly bound by layer/request row/step. Recorded parents match the expected ten-node topology. Every accepted path is a valid real-node child chain; the first twelve accepted counts agree with the sampler trace. The initial prompt/expected route, draft offset, actual draft IDs, all ten row IDs, full 3×10 mrope positions and input embeddings match across arms for the aligned forwards.

The common sampler trace has 12 entries:
`[1,3,1,0,3,3,1,2,0,0,4,5]`.

Its cumulative emitted-token boundary calculation (including the initial token and each event's correction/bonus) is:
`[1,3,7,9,10,14,18,20,23,24,25,30,36]`.

**None versus matched sham:** every recorded operand tensor, all 13 replay/published states, all hidden/residual tensors at layers 0–63, final-normalized hidden tensors, and logits are bitwise equal. The API's 32 token IDs and their logged logprobs are also exactly equal. The sham executes 13 factor-verifier and 13 scratch compact-commit events, with zero applied verifier/commit replacements. Its saved replay-before and published-after tensors are bitwise equal at every step. Thus the matched diagnostic workload has no observed numerical effect on this prefix; this is one matched pair, not a noise distribution.

**Authoritative publication:** every arm records running/authoritative row 21. The accepted-copy rows vary with accepted length; their recorded consistency-before flag is true for every event. Independently comparing actual saved tensors establishes all **12/12 next-h0 links per arm are bitwise identical to the preceding published-after state**, and their row IDs agree. The commit arm has 13 applied compact-commit events and zero verifier replacements. Its before/after state differs at every event. In particular, step 1 accepts three nodes, records destinations [21,23], and the subsequent scan's row-21 h0 equals that changed publication. This is the distinct-column case that the old implementation mishandled.

All before/after deltas below are max absolute values. “Published pair delta” compares the matched sham's and commit arm's actual published states, so it includes prior live substitutions. “Scan output delta” is the captured layer-62 production bf16 tree output difference.

| Call | Accepted length | Publication destinations | Commit before→after delta | Published pair delta | Scan output delta | Logit delta | Post-event boundary |
|---|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | 21 | 2.384185791015625e-7 | 2.384185791015625e-7 | 0 | 0 | complete |
| 1 | 3 | 21,23 | 4.76837158203125e-7 | 7.152557373046875e-7 | 0.000003814697265625 | 0 | complete |
| 2 | 1 | 21 | 2.384185791015625e-7 | 9.5367431640625e-7 | 0.0000152587890625 | 0 | complete |
| 3 | 0 | 21 | 7.450580596923828e-8 | 9.5367431640625e-7 | 3.725290298461914e-9 | 0 | complete |
| 4 | 3 | 21,23 | 2.384185791015625e-7 | 9.5367431640625e-7 | 0.0000019073486328125 | 0 | complete |
| 5 | 3 | 21,23 | 2.384185791015625e-7 | 0.0000011920928955078125 | 1.862645149230957e-9 | 0 | complete |
| 6 | 1 | 21 | 4.76837158203125e-7 | 0.0000011920928955078125 | 0.00006103515625 | 0.0625 | complete |
| 7 | 2 | 21,22 | 4.76837158203125e-7 | 9.5367431640625e-7 | 1.4901161193847656e-8 | 0 | complete |
| 8 | 0 | 21 | 1.1920928955078125e-7 | 9.5367431640625e-7 | 5.960464477539063e-8 | 0 | complete |
| 9 | 0 | 21 | 2.384185791015625e-7 | 9.5367431640625e-7 | 9.5367431640625e-7 | 0.15625 | complete |
| 10 | 4 | 21,24 | 9.5367431640625e-7 | 0.0000011920928955078125 | 0.0000019073486328125 | 0 | complete |
| 11 | 5 | 21,25 | 4.76837158203125e-7 | 0.0000011920928955078125 | 0.0000019073486328125 | 0 | terminal event |
| 12 | 2 | 21,22 | 3.5762786865234375e-7 | 0.0000011920928955078125 | 2.384185791015625e-7 | 0 | partial capture |

No q/k/v/a/b, A_log or dt_bias differences were observed between sham and commit at any recorded call. From call 1 onward their h0 differs because of the applied state replacement; their production scan outputs consequently differ. All layers 0–61 remain bitwise equal.

Downstream differences are confined to two identity-bound forwards:

| Call | Layer-62 hidden / residual max delta | Layer-63 hidden / residual max delta | Final norm max delta | Logits max / mean delta | Per-row argmax changes |
|---|---|---|---:|---|---:|
| 6 | 0 / 0.00390625 | 0 / 0.0078125 | 0.00244140625 | 0.0625 / 0.0000025056021968339337 | 0 |
| 9 | 0.03125 / 0.03125 | 1 / 0.25 | 0.140625 | 0.15625 / 0.002095431787893176 | 0 |

The maximum log-softmax difference at the reference argmax over all captured rows is 0.0000015497207641601562 at call 6 and 0.0000274985795840621 at call 9. These are per-row diagnostics, not changes in API-emitted token logprobs; the latter are unchanged. The largest live sham/commit recurrent-state difference is 0.0000011920928955078125. The largest same-event compact-versus-replay replacement delta is 0.00000095367431640625.

## CPU continuation, explicitly separate from the live arm

Executed the settled `experiments/e2/e7b_state_continuation.py` (SHA-256 `9aaa30503c0d7c114c1e9492aab7ac253a8b5a840b21cda9426f81b62226301d`, unchanged after execution) twice, using `--inject-step 1`, default fp64 arithmetic and explicitly reported fp32 carry:

1. None as reference, matched commit sham as arm: fidelity passes; injected state difference is zero; all reported state/output differences are zero.
2. Matched commit sham as reference, compact commit as arm: fidelity passes; inject the **actual** difference between their step-0 published states, max absolute **2.384185791015625e-7**.

The second run carries that one state difference through the reference's later captured operands and accepted paths. It does **not** repeatedly reapply compact commit as the live arm did. It holds one layer's future inputs fixed and exercises no full-model forward, logits or convolution/KV continuation.

Independently checked actual commit-arm self-fidelity as well: all 13 steps satisfy the declared frozen T9/T10 output class and T6 replay-state class; all twelve recorded h0 links are bitwise equal. Its maximum oracle/output absolute error is 0.0002337333386441265; maximum replay-before/oracle state error is 0.0000007152557373046875. The replay check uses the saved production replay-before state when replacement was applied.

For the real-state injection, every reported max-state delta remains exactly the injected 2.384185791015625e-7; the max-norm ratio is 1.0. This is persistence of the observed max error, not contraction of every component or a stability theorem. Through the fully API-bound future intervals, oracle-output max deltas range from 1.788227024790423e-9 to 4.853118927561351e-9.

| Reference call | Structural replay tokens since injection | Implied total count from base 3 | Oracle output max delta | Carried state max delta | Logged bf16 output bitwise fraction | Scope |
|---|---:|---:|---:|---:|---:|---|
| 1 | 4 | 7 | 3.5968239983641404e-9 | 2.384185791015625e-7 | 0.9999837279319763 | complete |
| 2 | 6 | 9 | 1.788227024790423e-9 | 2.384185791015625e-7 | 0.9999186396598816 | complete |
| 3 | 7 | 10 | 1.9690096150770664e-9 | 2.384185791015625e-7 | 1 | complete |
| 4 | 11 | 14 | 2.1340341993525413e-9 | 2.384185791015625e-7 | 1 | complete |
| 5 | 15 | 18 | 4.853118927561351e-9 | 2.384185791015625e-7 | 0.9999837279319763 | complete |
| 6 | 17 | 20 | 3.6270334721755226e-9 | 2.384185791015625e-7 | 0.9999837279319763 | complete |
| 7 | 20 | 23 | 2.054451116173528e-9 | 2.384185791015625e-7 | 0.999951183795929 | complete |
| 8 | 21 | 24 | 2.390557699710749e-9 | 2.384185791015625e-7 | 0.9999837279319763 | complete |
| 9 | 22 | 25 | 2.7627086694737013e-9 | 2.384185791015625e-7 | 0.9999674558639526 | complete |
| 10 | 27 | 30 | 3.808177599651241e-9 | 2.384185791015625e-7 | 1 | complete |
| 11 | 33 | 36 | 3.872883649197467e-9 | 2.384185791015625e-7 | 0.9999837279319763 | structural only |
| 12 | 36 | 39 | 2.746986594789891e-9 | 2.384185791015625e-7 | 0.9999674558639526 | structural only |

The fractions are retained as the fixture's floating-point summary values. They are not claims of exact integer bit-match counts.

## Terminal capture and valid horizon

There are **12 identity-bound forward calls (0–11), but only 11 complete post-event/API boundaries (calls 0–10)**. Call 11 begins at 30 known emitted tokens; its full accepted-plus-bonus event would reach 36, while the API ends at 32. Call 12 has no corresponding sampler-trace entry and no API-bound root token.

The terminal step-12 state is nevertheless a genuine captured internal record: all three arms record accepted length 2/path [1,2], a valid child chain under the captured topology, authoritative/running row 21, accepted-copy row 22, and the actual next-h0 link from step 11. Its operands/state satisfy the numerical fidelity checks. Its state file is written before the API response record (commit arm: 07:37:37.376536Z versus 07:37:37.473420Z). The depth-position record is present, but `tree_path_lcp.jsonl` is empty and the 12-entry per-request sampler trace supplies no independent terminal accepted event. Treat this as a structurally valid partial capture, not a fully emitted continuation boundary.

For the offline fixture's injection **after step 0**, the main-paper bound is **ten future intervals, calls 1–10, adding 27 structural replay tokens from base count 3 to count 30**. The fixture's later counts 33 and 36 are structural replay-token counts, implying totals 36 and 39 from that base. They are not 33/36 API-emitted continuation tokens. Report calls 11/12 separately, as above. This accounting closes the potential horizon overstatement without changing raw files or rerunning inference.

## Findings and minimum next work

No material blocker remains for reporting the narrow corrected commit diagnostic: the matched control is numerically clean, real publication is consumed, and downstream effects have an aligned input/state explanation. Preserve both the changed logits and unchanged token decisions; neither licenses a broad harmlessness claim.

The paper can report the completed one-prefix verifier/commit stage-isolation observations and the bounded one-layer continuation. It must retain the failed E7a freeze, no whole-model continuation qualification, no conv/KV qualification from this fixture, and no performance conclusion from these instrumented boots. Do not expand compact-method cells merely to strengthen this already defensible scoped observation. Continue the separately required sequential E2 qualification before E1.

## Evidence and reproducibility

Key raw paths under each arm:

- `e7b_spec.json`, `capture_expected.json`, `capture_provenance.json`, `operands_gate.json`, `capture_request.json`.
- `logs/e7b_substitute.jsonl`: `event`, `applied`, `mode`, `shadow`, `h0_row`, `authoritative_row`, `rows_written`, `state_max_abs_delta`.
- `logs/e7b_state/language_model_model_layers_62_linear_attn.row0.stepK.pt`: `accepted_len`, `path`, row metadata, `replay_before`, `published_after`.
- `logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.stepK.pt`: actual parents, q/k/v/a/b, h0 row/tensor, production output.
- `logs/layer_hidden.callK.pt`: `rows`, complete `positions`, `input_hidden`, per-layer hidden/residual, `final_norm_hidden`.
- `logs/final_logits.callK.pt`: `rows`, `capture_call_index`, `logits`.
- `logs/per_req_spec_trace.jsonl`, `logs/fr10_mtp_draft_trace.jsonl`: accepted counts and provenance-offset draft identity.

Bitwise tests compare equal-shaped/equal-dtype tensors viewed as contiguous uint8 bytes. The report's deltas use max absolute float64 differences of actual saved tensors. Forward identity requires equal token prefix before the call, expected topology, actual draft tokens, row IDs, full 3×10 positions and input embeddings. Post-event completeness additionally requires the entire accepted-plus-bonus boundary to lie within both API token arrays.

A reproducible CPU continuation invocation is:
```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/e2/e7b_state_continuation.py \
  <matched-sham-directory> --arm <commit-only-directory> --inject-step 1 --carry float32
```
The actual review captured the JSON in memory instead of writing remote output. Main-paper continuation summaries must select reference calls 1–10 as explained above. None/sham uses the same invocation with those two explicit directories.

The audit hashed 76 input artifacts per arm: all 52 per-call/per-step tensor files (13 each of operands, states, hidden, logits), the eight listed JSON/JSONL files below, and all 16 script-snapshot files. Each index is the SHA-256 of sorted `sha256 + two spaces + path-relative-to-arm + newline` records.

| Arm | Hashed files | Sorted-index SHA-256 |
|---|---:|---|
| none | 76 | `744a06c5f608561b7430fd8507a0d68cbac1522b0b686d30452b32ad980faa13` |
| commit sham | 76 | `187626e332413f15b0b5665ab92bcab70ead662150b571584180ec7f3c955be2` |
| commit only | 76 | `2ed33be567b0bc94abcc32b661241653e30e51038bce6c442b90e4fd81883e53` |

Individual non-tensor input hashes:
```text
ARM e7b_001_p072_arm1_none-B_fs_ieee-all
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  capture_expected.json
36bb3ae5c84746199b12717c540a3597c7308c26165c8a1b9597806eeca51259  capture_provenance.json
7e1276f4e14ba1b5e0c3d71d21411af5c13ac11dd1fde95b339c7ef3005c64e5  capture_request.json
8627e1b3a6602f1a9425667d8c00081c365da91b97870a30e23bec30642aece9  e7b_spec.json
ce8bb1835f73e466db58c5a8085a88bdf1c36952f7ed3c14c4cfb652d0663abf  logs/e7b_substitute.jsonl
67b57e4d21a1da5d81ff798c458ab23b24940656e326701368c1822d4a38d681  logs/fr10_mtp_draft_trace.jsonl
0f28c66e4fff3d9929b43b743b24a654bb20741149e5360a0101558043c09f67  logs/per_req_spec_trace.jsonl
518826826094fb33a5dc5445e8d13c5b888c64935b7b1233b675992755394408  operands_gate.json

ARM e7b_002_p072_arm2_sham-B_fs_ieee-all-commit_only
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  capture_expected.json
26d1dc817cb41f6ceef6d2cc1f2fbb2b816393430a31281f00eb0f2925a6704c  capture_provenance.json
11c178bff651db3408cda4b16aa2e08bc5afca6c35b8eb5f3e5efcc8d93fc896  capture_request.json
3800fc9030211055bc95ecc976993d796c9dd09982684153be4b15c5a34ea787  e7b_spec.json
ba415125c7297a8485e97a8f45856e1a3d2b9591c97be51acebb7585c514cffa  logs/e7b_substitute.jsonl
e71093f9cbaaa71770d007aa5e79a6b86355f9aebffe42ef349a58a98a8f7992  logs/fr10_mtp_draft_trace.jsonl
d8c7332101381c1e9af077ed0b0f6865140722d13643c21d6aee6ca5ec3f9dca  logs/per_req_spec_trace.jsonl
3a6173b6dfbc4097b62894bbcaeff95f77b004a081930eece8c1708278af6c5c  operands_gate.json

ARM e7b_003_p072_arm3_commit_only-B_fs_ieee-all
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  capture_expected.json
85bda469110b1eb288e329f7463d5c573e01cd3e3eb289bb9ff9a94cfe2c555f  capture_provenance.json
8c2cfb9ecd09dcc2f2fa5b4ece397bfbccd1092894f8f5d4a76d8ed79f1c4b96  capture_request.json
ef062c20409c9ab7cf0236ebef227130fa897b533257419177f06391b1860b5f  e7b_spec.json
1842b3d6e045d5eccf4bb54ae231a8831a5e3827fc92a35251010c801c8b698f  logs/e7b_substitute.jsonl
27c23d16714230a75c147bada294b8d5195f1099aef23c7d4a99b1d684ab5010  logs/fr10_mtp_draft_trace.jsonl
35a0f71cbf4e8a45b0a2901723afd6dbfa491bd83e6d080282153dfe14a32312  logs/per_req_spec_trace.jsonl
16d76ff3a087d47d8d99856774c2404bae95d9a29e5e1e7d46aef89389810f83  operands_gate.json
```

Shared script-snapshot hashes (identical across all three arms):
```text
acff98c93cabc3254e05dc49e0ed3fc114a881f03b307d48879954c2f54a646f  script_snapshot/SHA256SUMS
25f05bdb10088646978bd33d7e1eefa032d42ede1be71e4d71c05b0e3393a0e4  script_snapshot/capture_loop.v4.sh
aaec8b3dc6c5d854c0d6c16b219599225e751d65ee56e31c962f1b7547752c8f  script_snapshot/e7a_capture_launch.v3.sh
1d1b511acbe8b5ab218c016d61a89328fdd2cd30907e54efa25b6bff738d788b  script_snapshot/e7a_capture_shim.py
15c3536252183281c6ff1dd5fc9fcc068e5d056bf6fa5ec8b199cc302ac21a70  script_snapshot/e7a_kernels.py
45f261fa95fcd29eaa7d02add0dd6db713d0505538d2b8cfc33d8cd295726896  script_snapshot/e7a_native_launch.v2.sh
a5b47e094e293b38b2c887de0de8a1a6d93fd0304801945931b00761082705b0  script_snapshot/e7b_loop.v1.sh
8c2a086253ee707d181e924ca04e03e22c043e5c92994ba7148bf400688cb0c4  script_snapshot/e7b_operands_gate.py
ebfaa9ef86fdd6a8f8dd4cb551e4c24bfdd177e757ecc9f3b3e36054f2fbd1c1  script_snapshot/e7b_reduce.py
9c3a6d94b2e1c6ccb07e4aedbbc29d812b9cbf4d102789b5f8a262b04b99b5ca  script_snapshot/e7b_runtime.py
40063932a729d22eb12370c185b2f01b4979db013a5872e539c26e739eff3c5e  script_snapshot/e7b_substitute_shim.py
6148206a024c17746bce91c6c3e065275a8d47c24d660c665d7e3d8e29ad265b  script_snapshot/gpu_oom_guard.sh
6e71a24c5b7ea82932b219b7502a5c19d3b367e37105451a306b7afb3c95ed04  script_snapshot/inspect_capture.py
b70a3165f03b143131311ea4cc2766d8796e320c6fbc4e588e3cde21e08d7954  script_snapshot/prefix_pool.py
5d1dafa4c2ccf51eb65a2c6e840490927c5f546e0d7382d79cf5971df655cb26  script_snapshot/serve_drivers.v7.sh
a24c110c9c6edf8a9011a7df3f0c9667474179aa5c5e3569765d9effbbf3f804  script_snapshot/test_inspector_controls.py
```

Additional terminal-evidence hashes, not included in the 76-file index:
```text
763bf206a7a4bc659ceacdff96d450aabc0cd0cad58ce9daa448e77e26218018  logs/fr10_tree_depth_positions.jsonl [all arms]
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/tree_path_lcp.jsonl [all arms, empty]
e5da95ae01a1a9b53cd091a30a32505ad6b0cf7056b275d966467b01e3047230  none/logs/tree_sampler_debug.jsonl
df5824f29f18b096ebc5b47b8d71d02899a2aa60a87668167bbf6690481a6724  sham/logs/tree_sampler_debug.jsonl
ba718a4b71e792448d9508825317cd9e76281d48a3e72e0ec5493e159d4996f6  commit/logs/tree_sampler_debug.jsonl
```

## Manuscript evidence check: stage-isolated model diagnostics

Reviewed only the added subsection at `main.tex:359–367` (`sec:e7b-diagnostic`). Local and remote manuscript SHA-256 are both `be61c9192292d006eab7326cc091d1fe135c35fee037d1cf542a98899f1d99cd`. The source matches the independently verified verifier, corrected-commit and offline-continuation evidence. No canonical manuscript edit, inference or PDF rebuild was performed in this check.

- Lines 361–363 correctly distinguish the twelve aligned verifier calls/120 candidate rows from the thirteenth partial capture; the step-3/7 logit differences, unchanged argmax/token observations, and local bf16 figures match round15. The freeze explicitly lists p072 under `pilot_evidence[0]`, supporting its pilot-prefix description.
- Line 365 correctly records the corrected commit result, thirteen actual publications and twelve next-read byte links. It preserves the crucial distinction between twelve input-aligned forwards and eleven complete postcommit/API boundaries, without promoting the extra capture to an emitted continuation interval.
- Line 367 correctly restricts the once-injected recurrence to **ten fully API-bound future intervals**, declares fp64 arithmetic with fp32 carry, reports the verified error range, and separates it from repeated live substitution and cross-layer feedback. “No amplification” is explicitly confined to the observed maximum-state-error measure and horizon; this is supported by the constant measured max error and is not a general stability claim.
- The subsection leaves convolution history, attention KV, sampler-law and full-route qualification open, consistent with the separate E2/E1 pending status.

**Disposition: no material evidence mismatch or claim-scope blocker in the new subsection.** One nonblocking precision improvement is available at line 365: qualify the historical accepted-copy publication defect with “when accepted length exceeded one,” since zero/one acceptance already aliases the accepted copy with column zero. The verified corrected evidence and present experiment conclusions are unchanged by that wording clarification.
