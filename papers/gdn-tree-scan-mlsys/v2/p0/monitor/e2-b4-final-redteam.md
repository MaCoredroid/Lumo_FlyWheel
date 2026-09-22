# B4 final bounded E2 red-team — 2026-09-22

**PASS for the observed policy-B B4 route.** The original five gate findings and the additional nonterminal premature-stop defect are closed against the exact settled sources below. CPU reanalysis passes for all four requests, 53 recorded layer-62 steps, 49 exact state handoffs, and 32 captured request blocks. No new GPU boot or J3 cell is needed to close these bounded findings. This is sufficient for the selected tree-route qualification defined in `e2-closure-decision.md`, together with its already-reviewed helper composition and B1 evidence. It is **not** full-model sequential/native equivalence, universal route correctness, or a retained E1 timing result. Native-arm preflight and campaign qualification remain separate.

## Evidence and identity

Remote repository: `/home/mark/lumo-paper-v2-20260921`. Below, `E` means `papers/gdn-tree-scan-mlsys/v2/experiments`; `R` means `E/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all`.

The boot used stock TREE_ATTN with KV remap/reorder/syncfree **1/0/1**, replay=1, eager-pack=1, fused-conv=1, synchronous scheduling, max-num-seqs=4, and enforce-eager. `R/docker_inspect.json` hashes to `5b489f3b456750910ddd4af7595aa7f9450d8aa552a044d2dde2151948e2e793`; `R/docker_logs.txt` hashes to `daa444abfe4c33a598f12f011250f9aa81c1970f96368c1a6dc2f1f4a200c4fe`. Loaded fused-conv engagement is logged at line 281 and KV remap at line 333. These tie the observed route to the previously reviewed source/helper contract; they are not a live bytewise oracle for every convolution/KV bank.

I derived the request timelines independently from recorder output-row order and per-tree-row counters, then compared them with `e7b_ledger_align.py:49`. Every state/operand filename and payload layer/row/step agrees. Every authoritative published row is the running col0 row, all declared replay rows agree, and all 53 mode-none `replay_before`/`published_after` pairs are bitwise equal. All 49 next-h0 links match both row identity and tensor bytes (uint8 views, not floating-point numeric equality).

| Request | Prompt tokens | Layer steps | Exact handoffs | Ledger / API tokens | Max output abs. error | Max replay-state abs. error |
|---|---:|---:|---:|---:|---:|---:|
| p072 | 256 | 12 | 11 | 34 / 32 | 2.40523414e-4 | 7.15255737e-7 |
| p095 | 4096 | 14 | 13 | 32 / 32 | 4.63376856e-4 | 7.15255737e-7 |
| p017 | 256 | 13 | 12 | 34 / 32 | 4.72719456e-4 | 3.57627869e-7 |
| p085 | 4096 | 14 | 13 | 34 / 32 | 4.78956074e-4 | 9.53674316e-7 |

The request order for `--request 0,1,2,3` is **p072, p095, p017, p085**, which differs from cohort-directory order. The row3→row0 compaction and duplicate-sibling witness concern **p085**, not p095 (reply 23 mislabels both). p085 uses row3/steps0–7 at forwards5–12 and row0/steps12–17 at forwards13–18. Its 13 handoffs pass, including the compaction boundary; no cross-request row chain is substituted.

The sealed ledger has 59 contiguous events, 19 forwards/output records, 14 pure physical steps, five prefill/mixed records, and no recorded errors or sink failures. All 53 state and 53 operand files are used exactly once in the four request timelines. State and spec-trace accepted lengths agree at all 53 points.

## Captured path and API binding

Eight 40-row final-logit/hidden capture pairs bind to forwards5–12, physical steps0–7, giving **32 request blocks**. I checked all three position axes for every captured tree node against prompt length, previously emitted count, and tree depth—not just the root position. Captures contain the expected 64 layers, input/final-norm tensors, row IDs, and call/saved IDs.

Each published path is a valid parent chain. Every accepted node's actual draft token equals its parent's full-vocabulary target argmax; every emitted accepted/bonus token equals the argmax at the corresponding published-path node. All 32 blocks pass; none stops at a selected node with a still-matching child. The duplicate-token p085 path [1,3] is legal. First-child and last-child walk agreement are only diagnostics (31/32 and 32/32), not contracts.

**85 structural ledger tokens** are bound in those 32 blocks, of which **83 are API-visible**. p072's last captured forward structurally produces five tokens but its API budget retains only the first three; the two extra tokens must not be counted as API output. Across the entire boot, 134 structural ledger tokens reconcile to 128 API-visible tokens, with terminal trims 2/0/2/2 in the table's request order.

API token identity uses the independent exact-formatter inverse from `p0/monitor/e2-b4-api-inversion-redteam.md`, SHA-256 `f3b37263801f0faeaf4ecd4faeaeb6382b0e014fac54501109e4367e1e0a388a`. That review derives all 128 IDs without using ledger/draft/output IDs and covers the pinned formatter's complete model-ID domain and contextual replacement handling. I then independently mapped each API response ID to exactly one engine request and verified all four 32-token sequences against the complete ledger, allowing truncation only within that request's final row. The pilot's original API captures contain strings; this remains **derived exact ID evidence**, not a claim that the client directly requested IDs or that the original raw HTTP bodies were preserved.

For gate execution I created a temporary overlay: original logs and capture metadata were read-only symlinks, while temporary copies of the API capture records replaced token strings with the independently recovered `token_id:` values. C0–C5 all passed, rc0. Raw captures and all previous failure files remain untouched. This bypasses the old provisional generic tokenizer inverse using the separately certified formatter proof; future clients must still request direct IDs.

## Frozen numerical classes and support

I reran `e7b_state_continuation.v2.py:107` on all 53 independently derived timeline records with fp64 arithmetic and the recorded fp32 h0. The complete per-step dictionaries exactly reproduce the four saved `fixture_none_fidelity.req{0,1,2,3}.json` files, including the thresholds:

- T9 significant bf16 ULP max ≤1: observed max 1 in every request.
- T10 served-output absolute error ≤7.275056663253038e-4: overall observed max **4.789560742476662e-4**.
- T6 replay-state absolute error ≤2.5938975340977777e-6: overall observed max **9.5367431640625e-7**.
- Next-h0 bitwise fraction =1: all 49 links exact.

Minimum oracle-rounded-bf16/served-output bitwise fraction is **0.9998534917831421** (p085); the output fidelity claim is numerical, not bitwise. The original `PILOT_FREEZE.json` remains SHA-256 `21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce`. These are **one-layer fixed-input fidelity** readings. Zero-perturbation continuation in the saved files is a self-consistency control; it does not establish cross-layer or full-model equivalence.

The exact-ID joiner rerun returns rc0 with seven complete four-request intervals, 28 request rows, 73 API-bound tokens, and 8.432447447441518 seconds of unique physical-step wall time. Eight full forwards produce seven retained intervals. Prespecified exclusions are three successor-cohort changes, three wrong-occupancy intervals, and the terminal interval. The 3.762523009441793-second first interval stays in the primary support. This boot is a capture/qualification pilot with no warmup phase, so none of these numbers is a confirmation speedup or confidence interval.

## Gate repair closure

All source bytes were copied to a temporary directory before execution and verified unchanged afterward. Owner controls: **15/15 pass**. Independent negatives reject capture call/saved=999, a duplicated hidden payload, wrong operand identity, wrong state layer, altered draft tokens, a nonterminal partial emission, ledger/physical request-order mismatch, unmappable API IDs, and a corrupted per-request h0 chain. The positive synthetic B4 request/compaction fixture passes.

The first recheck of gate `978105f9a16c5ad367ce499468675fe0fe8d27a64df54163c1ffdd77f57dc427` still allowed a local premature stop as a diagnostic. Reproduction: from the positive synthetic fixture, set draft record3/request-row3/node1 to final-logits call2/row30 argmax, leaving the published root-only path unchanged. That request continues at forward5. Old result: rc0 with a matching child not taken. **Final v3e result: rc1, C4 alone fails; C0/C1/C2/C3/C5 still pass.** Thus the rejection is causally attributable to the repaired stop check at `e7b_b4_capture_gate.py:152`, not an unrelated broken fixture. The real data has zero such stops.

The loop's final evidence block at `e7b_loop.v4.sh:120` also passes isolated CPU shell tests: positive rc0; missing ledger, mapper refusal, join rc2, join INVALID despite rc0, join REFUSED despite rc0, and wrong mapped-request count all exit16. No server-launch code was executed.

No material bounded E2 blocker remains in these observed B4 captures. Preserve the original failed `operands_gate.json`, `capture_gate.v3.json`, and all subsequent versioned results; the boot-time loop failed and was qualified by repaired offline analysis, not retrospectively declared successful.

## Exact settled analysis sources

Paths relative to E. Reply 23's earlier gate/helper/test hashes have been superseded; use these for the analysis companion and future gate snapshots.

| File | SHA-256 |
|---|---|
| `e2/e7b_b4_capture_gate.py` | `0f0aef4f7880427a3afefc4a42dc2000526b79afa833cfff9ea77b887b78be13` |
| `e2/e7b_ledger_align.py` | `3983b57e0bb0e52c4abf6ba6a31155febd96632f583597cd5ff7df6bafdd997e` |
| `e2/e7b_operands_gate.py` | `f5f5ec29fe04c0238deff807e09c302a80328318aeea51278f864c35b5ac754f` |
| `e2/test_e7b_b4_capture_gate_cpu.py` | `fc7e12f6f206512b19257623ef31f2f86d3eb703b0007fdee9b82ba024aa9bf1` |
| `e2/e7b_state_continuation.v2.py` | `e00e8684facfb819e12d16ff0e5d649315cdaf3ef14df73417f5a5b87001e809` |
| `e2/e7b_loop.v4.sh` | `5684d3e67699a9502c6df61b21ea71d866b78cf8ebc90cdc5103703f27893a0d` |
| `e7a/e7a_core.py` | `89feb49d2bf17b670fb8b5daf1032fd55467f51de3bf33a84f08a1ed555cabd7` |
| `e1/e1_join.py` | `cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766` |

Also preserve the **actual loaded boot sources**, separately from later analysis tools:

| Path relative to R | SHA-256 |
|---|---|
| `loaded_backend/gpu_model_runner.py` | `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40` |
| `loaded_backend/tree_attn.py` | `a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97` |
| `loaded_backend/gdn_linear_attn.py` | `5175f7f42a1fa2157e8be6423a69c71e800d415bbb2d45a3aeb065394c6ce9ed` |
| `loaded_backend/rejection_sampler.py` | `110371ca5ca623d04d5e47ac2f3227592880daf4544e54913d3d6016c68b2bb0` |
| `script_snapshot/e7b_runtime.py` | `9c3a6d94b2e1c6ccb07e4aedbbc29d812b9cbf4d102789b5f8a262b04b99b5ca` |
| `script_snapshot/serve_drivers.v12.sh` | `a1a3490de13e8fb8494c0bb32ac31b79076aa40ac699344c26464368cae48102` |
| `script_snapshot/e7b_loop.v3.sh` | `99c7d3117f5aab60656235f94d8f09391f57318bb7ae92cb306bbb5345256d94` |
| `script_snapshot/e1_event_recorder_shim.py` | `06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df` |
| `script_snapshot/e1_recorder.py` | `1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0` |

The appendix retains 136 individual artifact hashes and exact request/forward/path ledgers. Its sorted compact JSON hash map digest is `de4a816242f1a11ba8bde7504aafbed94f8bcff0f8d4f489488d267a031bdcdd`; tensor paths are relative to R, and `fixture_none_fidelity.req*.json` paths are relative to R's parent.

<details>
<summary>Exact CPU reduction results and artifact hashes</summary>

```json
{
  "source_stable": true,
  "independent_assertions": 456,
  "all_pass": true,
  "requests": [
    {
      "prefix": "p072",
      "engine_id": "cmpl-a33355be5168a83b-0-9ae2ab44",
      "prompt_tokens": 256,
      "timeline": [
        [
          1,
          0,
          0
        ],
        [
          2,
          0,
          1
        ],
        [
          3,
          0,
          2
        ],
        [
          4,
          0,
          3
        ],
        [
          5,
          0,
          4
        ],
        [
          6,
          0,
          5
        ],
        [
          7,
          0,
          6
        ],
        [
          8,
          0,
          7
        ],
        [
          9,
          0,
          8
        ],
        [
          10,
          0,
          9
        ],
        [
          11,
          0,
          10
        ],
        [
          12,
          0,
          11
        ]
      ],
      "state_links_bitwise": 11,
      "api_tokens": 32,
      "ledger_tokens": 34,
      "terminal_trim": 2,
      "fidelity_max_abs": 0.0002405234139283563,
      "fidelity_max_ulp16_sig": 1,
      "replay_max_abs": 7.152557373046875e-7,
      "bf16_fraction_min": 0.9999023675918579
    },
    {
      "prefix": "p095",
      "engine_id": "cmpl-bba5676034b3072d-0-8771797d",
      "prompt_tokens": 4096,
      "timeline": [
        [
          3,
          1,
          0
        ],
        [
          4,
          1,
          1
        ],
        [
          5,
          1,
          2
        ],
        [
          6,
          1,
          3
        ],
        [
          7,
          1,
          4
        ],
        [
          8,
          1,
          5
        ],
        [
          9,
          1,
          6
        ],
        [
          10,
          1,
          7
        ],
        [
          11,
          1,
          8
        ],
        [
          12,
          1,
          9
        ],
        [
          13,
          1,
          10
        ],
        [
          14,
          1,
          11
        ],
        [
          15,
          1,
          12
        ],
        [
          16,
          1,
          13
        ]
      ],
      "state_links_bitwise": 13,
      "api_tokens": 32,
      "ledger_tokens": 32,
      "terminal_trim": 0,
      "fidelity_max_abs": 0.00046337685564543096,
      "fidelity_max_ulp16_sig": 1,
      "replay_max_abs": 7.152557373046875e-7,
      "bf16_fraction_min": 0.9999023675918579
    },
    {
      "prefix": "p017",
      "engine_id": "cmpl-be7c5fc6026dc3ca-0-b99e1ca1",
      "prompt_tokens": 256,
      "timeline": [
        [
          3,
          2,
          0
        ],
        [
          4,
          2,
          1
        ],
        [
          5,
          2,
          2
        ],
        [
          6,
          2,
          3
        ],
        [
          7,
          2,
          4
        ],
        [
          8,
          2,
          5
        ],
        [
          9,
          2,
          6
        ],
        [
          10,
          2,
          7
        ],
        [
          11,
          2,
          8
        ],
        [
          12,
          2,
          9
        ],
        [
          13,
          2,
          10
        ],
        [
          14,
          2,
          11
        ],
        [
          15,
          2,
          12
        ]
      ],
      "state_links_bitwise": 12,
      "api_tokens": 32,
      "ledger_tokens": 34,
      "terminal_trim": 2,
      "fidelity_max_abs": 0.0004727194563886872,
      "fidelity_max_ulp16_sig": 1,
      "replay_max_abs": 3.5762786865234375e-7,
      "bf16_fraction_min": 0.9999186396598816
    },
    {
      "prefix": "p085",
      "engine_id": "cmpl-822fbbe30317f121-0-9234a7af",
      "prompt_tokens": 4096,
      "timeline": [
        [
          5,
          3,
          0
        ],
        [
          6,
          3,
          1
        ],
        [
          7,
          3,
          2
        ],
        [
          8,
          3,
          3
        ],
        [
          9,
          3,
          4
        ],
        [
          10,
          3,
          5
        ],
        [
          11,
          3,
          6
        ],
        [
          12,
          3,
          7
        ],
        [
          13,
          0,
          12
        ],
        [
          14,
          0,
          13
        ],
        [
          15,
          0,
          14
        ],
        [
          16,
          0,
          15
        ],
        [
          17,
          0,
          16
        ],
        [
          18,
          0,
          17
        ]
      ],
      "state_links_bitwise": 13,
      "api_tokens": 32,
      "ledger_tokens": 34,
      "terminal_trim": 2,
      "fidelity_max_abs": 0.0004789560742476662,
      "fidelity_max_ulp16_sig": 1,
      "replay_max_abs": 9.5367431640625e-7,
      "bf16_fraction_min": 0.9998534917831421
    }
  ],
  "capture_blocks": [
    {
      "capture": 0,
      "forward": 5,
      "row": 0,
      "row_step": 4,
      "request": "p072",
      "path": [
        1,
        3
      ],
      "emitted": [
        6971,
        6813,
        12333
      ],
      "terminal": false,
      "root_position": 264
    },
    {
      "capture": 0,
      "forward": 5,
      "row": 1,
      "row_step": 2,
      "request": "p095",
      "path": [
        1,
        2
      ],
      "emitted": [
        264,
        2972,
        588
      ],
      "terminal": false,
      "root_position": 4100
    },
    {
      "capture": 0,
      "forward": 5,
      "row": 2,
      "row_step": 2,
      "request": "p017",
      "path": [
        1
      ],
      "emitted": [
        332,
        21853
      ],
      "terminal": false,
      "root_position": 259
    },
    {
      "capture": 0,
      "forward": 5,
      "row": 3,
      "row_step": 0,
      "request": "p085",
      "path": [],
      "emitted": [
        5303
      ],
      "terminal": false,
      "root_position": 4096
    },
    {
      "capture": 1,
      "forward": 6,
      "row": 0,
      "row_step": 5,
      "request": "p072",
      "path": [
        1
      ],
      "emitted": [
        36349,
        788
      ],
      "terminal": false,
      "root_position": 267
    },
    {
      "capture": 1,
      "forward": 6,
      "row": 1,
      "row_step": 3,
      "request": "p095",
      "path": [],
      "emitted": [
        61384
      ],
      "terminal": false,
      "root_position": 4103
    },
    {
      "capture": 1,
      "forward": 6,
      "row": 2,
      "row_step": 3,
      "request": "p017",
      "path": [
        1
      ],
      "emitted": [
        421,
        11693
      ],
      "terminal": false,
      "root_position": 261
    },
    {
      "capture": 1,
      "forward": 6,
      "row": 3,
      "row_step": 1,
      "request": "p085",
      "path": [
        1,
        3
      ],
      "emitted": [
        506,
        62987,
        854
      ],
      "terminal": false,
      "root_position": 4097
    },
    {
      "capture": 2,
      "forward": 7,
      "row": 0,
      "row_step": 6,
      "request": "p072",
      "path": [
        1,
        2,
        4,
        6
      ],
      "emitted": [
        6971,
        6813,
        12333,
        36349,
        2135
      ],
      "terminal": false,
      "root_position": 269
    },
    {
      "capture": 2,
      "forward": 7,
      "row": 1,
      "row_step": 4,
      "request": "p095",
      "path": [
        1
      ],
      "emitted": [
        332,
        220
      ],
      "terminal": false,
      "root_position": 4104
    },
    {
      "capture": 2,
      "forward": 7,
      "row": 2,
      "row_step": 4,
      "request": "p017",
      "path": [
        1,
        3
      ],
      "emitted": [
        279,
        220,
        16
      ],
      "terminal": false,
      "root_position": 263
    },
    {
      "capture": 2,
      "forward": 7,
      "row": 3,
      "row_step": 2,
      "request": "p085",
      "path": [
        1,
        3
      ],
      "emitted": [
        13,
        561,
        8057
      ],
      "terminal": false,
      "root_position": 4100
    },
    {
      "capture": 3,
      "forward": 8,
      "row": 0,
      "row_step": 7,
      "request": "p072",
      "path": [
        1,
        3
      ],
      "emitted": [
        6971,
        6813,
        12333
      ],
      "terminal": false,
      "root_position": 274
    },
    {
      "capture": 3,
      "forward": 8,
      "row": 1,
      "row_step": 5,
      "request": "p095",
      "path": [],
      "emitted": [
        19
      ],
      "terminal": false,
      "root_position": 4106
    },
    {
      "capture": 3,
      "forward": 8,
      "row": 2,
      "row_step": 5,
      "request": "p017",
      "path": [],
      "emitted": [
        21
      ],
      "terminal": false,
      "root_position": 266
    },
    {
      "capture": 3,
      "forward": 8,
      "row": 3,
      "row_step": 3,
      "request": "p085",
      "path": [
        1,
        2
      ],
      "emitted": [
        2099,
        369,
        23185
      ],
      "terminal": false,
      "root_position": 4103
    },
    {
      "capture": 4,
      "forward": 9,
      "row": 0,
      "row_step": 8,
      "request": "p072",
      "path": [
        1,
        3
      ],
      "emitted": [
        36349,
        736,
        6971
      ],
      "terminal": false,
      "root_position": 277
    },
    {
      "capture": 4,
      "forward": 9,
      "row": 1,
      "row_step": 6,
      "request": "p095",
      "path": [],
      "emitted": [
        18295
      ],
      "terminal": false,
      "root_position": 4107
    },
    {
      "capture": 4,
      "forward": 9,
      "row": 2,
      "row_step": 6,
      "request": "p017",
      "path": [
        1,
        2,
        4
      ],
      "emitted": [
        13,
        19,
        20,
        85123
      ],
      "terminal": false,
      "root_position": 267
    },
    {
      "capture": 4,
      "forward": 9,
      "row": 3,
      "row_step": 4,
      "request": "p085",
      "path": [],
      "emitted": [
        1518
      ],
      "terminal": false,
      "root_position": 4106
    },
    {
      "capture": 5,
      "forward": 10,
      "row": 0,
      "row_step": 9,
      "request": "p072",
      "path": [],
      "emitted": [
        6813
      ],
      "terminal": false,
      "root_position": 280
    },
    {
      "capture": 5,
      "forward": 10,
      "row": 1,
      "row_step": 7,
      "request": "p095",
      "path": [],
      "emitted": [
        23014
      ],
      "terminal": false,
      "root_position": 4108
    },
    {
      "capture": 5,
      "forward": 10,
      "row": 2,
      "row_step": 7,
      "request": "p017",
      "path": [
        1,
        2,
        4,
        7
      ],
      "emitted": [
        198,
        429,
        279,
        220,
        17
      ],
      "terminal": false,
      "root_position": 271
    },
    {
      "capture": 5,
      "forward": 10,
      "row": 3,
      "row_step": 5,
      "request": "p085",
      "path": [
        1,
        2,
        4,
        6,
        8
      ],
      "emitted": [
        279,
        8057,
        8213,
        26,
        279,
        62987
      ],
      "terminal": false,
      "root_position": 4107
    },
    {
      "capture": 6,
      "forward": 11,
      "row": 0,
      "row_step": 10,
      "request": "p072",
      "path": [
        1,
        2
      ],
      "emitted": [
        12333,
        36349,
        1824
      ],
      "terminal": false,
      "root_position": 281
    },
    {
      "capture": 6,
      "forward": 11,
      "row": 1,
      "row_step": 8,
      "request": "p095",
      "path": [
        1
      ],
      "emitted": [
        318,
        1719
      ],
      "terminal": false,
      "root_position": 4109
    },
    {
      "capture": 6,
      "forward": 11,
      "row": 2,
      "row_step": 8,
      "request": "p017",
      "path": [
        1,
        2
      ],
      "emitted": [
        24,
        13,
        16
      ],
      "terminal": false,
      "root_position": 276
    },
    {
      "capture": 6,
      "forward": 11,
      "row": 3,
      "row_step": 6,
      "request": "p085",
      "path": [
        1,
        2,
        4,
        7
      ],
      "emitted": [
        33877,
        513,
        23185,
        1518,
        62987
      ],
      "terminal": false,
      "root_position": 4113
    },
    {
      "capture": 7,
      "forward": 12,
      "row": 0,
      "row_step": 11,
      "request": "p072",
      "path": [
        1,
        2,
        4,
        6
      ],
      "emitted": [
        6971,
        6813,
        12333,
        36349,
        1469
      ],
      "terminal": true,
      "root_position": 284
    },
    {
      "capture": 7,
      "forward": 12,
      "row": 1,
      "row_step": 9,
      "request": "p095",
      "path": [
        1
      ],
      "emitted": [
        220,
        21
      ],
      "terminal": false,
      "root_position": 4111
    },
    {
      "capture": 7,
      "forward": 12,
      "row": 2,
      "row_step": 9,
      "request": "p017",
      "path": [
        1
      ],
      "emitted": [
        16,
        15686
      ],
      "terminal": false,
      "root_position": 279
    },
    {
      "capture": 7,
      "forward": 12,
      "row": 3,
      "row_step": 7,
      "request": "p085",
      "path": [
        1,
        2
      ],
      "emitted": [
        8213,
        13,
        1061
      ],
      "terminal": false,
      "root_position": 4118
    }
  ],
  "capture_gate": {
    "run": "/tmp/b4-actual-redteam-xvg_fprt/derived_api_run",
    "cohort": 4,
    "gate_sha256": "0f0aef4f7880427a3afefc4a42dc2000526b79afa833cfff9ea77b887b78be13",
    "helper_sha256": "3983b57e0bb0e52c4abf6ba6a31155febd96632f583597cd5ff7df6bafdd997e",
    "checks": [
      {
        "check": "C1_capture_identity_and_content",
        "pass": true,
        "detail": {
          "n_captures": 8,
          "violations": []
        }
      },
      {
        "check": "C0_ledger_spec_trace_draft_trace_aligned",
        "pass": true,
        "detail": {
          "forwards": 19,
          "pure": 14
        }
      },
      {
        "check": "C2_captures_bound_to_forwards_and_requests_by_num_tokens_and_root_positions",
        "pass": true,
        "detail": {
          "bound_forward_indices": [
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
          ],
          "physical_step_ids": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7
          ],
          "violations": []
        }
      },
      {
        "check": "C3_operands_state_payload_identity_and_spec_trace_binding_per_request",
        "pass": true,
        "detail": {
          "n_states_bound": 32,
          "violations": []
        }
      },
      {
        "check": "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request",
        "pass": true,
        "detail": {
          "bound_tokens": 85,
          "terminal_partial_prefix": [],
          "terminal_stops_with_matching_child_allowed_by_clipping": [],
          "diagnostic_walk_agreement_informational_only": {
            "first_child_agrees": 31,
            "last_child_agrees": 32,
            "blocks": 32
          },
          "violations": []
        }
      },
      {
        "check": "C5_api_responses_mapped_and_token_sequences_reconciled",
        "pass": true,
        "detail": {
          "domain": {
            "cmpl-822fbbe30317f121": "token_ids",
            "cmpl-a33355be5168a83b": "token_ids",
            "cmpl-bba5676034b3072d": "token_ids",
            "cmpl-be7c5fc6026dc3ca": "token_ids"
          },
          "map": {
            "cmpl-822fbbe30317f121": "cmpl-822fbbe30317f121-0-9234a7af",
            "cmpl-a33355be5168a83b": "cmpl-a33355be5168a83b-0-9ae2ab44",
            "cmpl-bba5676034b3072d": "cmpl-bba5676034b3072d-0-8771797d",
            "cmpl-be7c5fc6026dc3ca": "cmpl-be7c5fc6026dc3ca-0-b99e1ca1"
          },
          "string_domain_identity": null,
          "violations": []
        }
      }
    ],
    "all_pass": true,
    "verdict": "B4 CAPTURE GATE v3e PASS"
  },
  "join": {
    "n_events": 59,
    "invalid": null,
    "refused": null,
    "phases": {
      "warmup": [],
      "timed": [
        "cmpl-822fbbe30317f121-0-9234a7af",
        "cmpl-a33355be5168a83b-0-9ae2ab44",
        "cmpl-bba5676034b3072d-0-8771797d",
        "cmpl-be7c5fc6026dc3ca-0-b99e1ca1"
      ]
    },
    "n_forwards": 19,
    "n_output_records": 19,
    "n_nonpure_output_records": 5,
    "n_physical_steps": 14,
    "n_chain_breaks": 5,
    "n_usable": 7,
    "n_excluded": 7,
    "excluded": {
      "7": {
        "reason": "cohort change at the successor (ramp/drain)",
        "wall_s": 0.7794493734836578
      },
      "8": {
        "reason": "occupancy 3 != expected 4 (ramp/drain)",
        "wall_s": 0.5490235462784767
      },
      "9": {
        "reason": "occupancy 3 != expected 4 (ramp/drain)",
        "wall_s": 0.518454228527844
      },
      "10": {
        "reason": "cohort change at the successor (ramp/drain)",
        "wall_s": 0.52860994823277
      },
      "11": {
        "reason": "cohort change at the successor (ramp/drain)",
        "wall_s": 0.42711280286312103
      },
      "12": {
        "reason": "occupancy 1 != expected 4 (ramp/drain)",
        "wall_s": 1.708567876368761
      },
      "13": {
        "reason": "terminal: no successor physical step id+1",
        "wall_s": null
      }
    },
    "excluded_wall_s_total": 4.511217775754631,
    "truncated_tail_tokens": {
      "cmpl-a33355be5168a83b-0-9ae2ab44": 2,
      "cmpl-be7c5fc6026dc3ca-0-b99e1ca1": 2,
      "cmpl-822fbbe30317f121-0-9234a7af": 2
    },
    "over_cap_diagnostic": {
      "cap_s": 1.5,
      "n_intervals_over_cap": 1,
      "wall_s_over_cap": 3.762523009441793,
      "physical_step_ids": [
        0
      ],
      "tokens_per_wall_second_trimmed_DIAGNOSTIC_ONLY": 13.70471853446374
    },
    "phase_summary": {
      "warmup": {
        "requests": 0,
        "ledger_tokens": 0,
        "api_tokens": 0
      },
      "timed": {
        "requests": 4,
        "ledger_tokens": 134,
        "api_tokens": 128
      }
    },
    "token_evidence": {
      "cmpl-a33355be5168a83b-0-9ae2ab44": 32,
      "cmpl-be7c5fc6026dc3ca-0-b99e1ca1": 32,
      "cmpl-822fbbe30317f121-0-9234a7af": 32,
      "cmpl-bba5676034b3072d-0-8771797d": 32
    },
    "ledger_tokens_all_forwards": 134,
    "sum_wall_s_unique_physical_steps": 8.432447447441518,
    "sum_emitted_tokens_api_bound_pure_support": 73,
    "n_request_rows": 28,
    "tokens_per_wall_second": 8.657035867107464
  },
  "owner_controls_count": 15,
  "owner_controls_pass": true,
  "negative_cases": {
    "independent_positive": {
      "rc": 0,
      "failed_checks": []
    },
    "capture_identity": {
      "rc": 1,
      "failed_checks": [
        "C1_capture_identity_and_content"
      ]
    },
    "duplicate_hidden": {
      "rc": 1,
      "failed_checks": [
        "C1_capture_identity_and_content",
        "C2_captures_bound_to_forwards_and_requests_by_num_tokens_and_root_positions"
      ]
    },
    "wrong_operand_identity": {
      "rc": 1,
      "failed_checks": [
        "C3_operands_state_payload_identity_and_spec_trace_binding_per_request"
      ]
    },
    "wrong_state_layer": {
      "rc": 1,
      "failed_checks": [
        "C3_operands_state_payload_identity_and_spec_trace_binding_per_request",
        "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request"
      ]
    },
    "wrong_draft_tokens": {
      "rc": 1,
      "failed_checks": [
        "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request"
      ]
    },
    "nonterminal_partial": {
      "rc": 1,
      "failed_checks": [
        "C3_operands_state_payload_identity_and_spec_trace_binding_per_request",
        "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request"
      ]
    },
    "wrong_request_order": {
      "rc": 1,
      "failed_checks": [
        "C0_ledger_spec_trace_draft_trace_aligned",
        "C2_captures_bound_to_forwards_and_requests_by_num_tokens_and_root_positions",
        "C3_operands_state_payload_identity_and_spec_trace_binding_per_request",
        "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request",
        "C5_api_responses_mapped_and_token_sequences_reconciled"
      ]
    },
    "unmappable_api": {
      "rc": 1,
      "failed_checks": [
        "C2_captures_bound_to_forwards_and_requests_by_num_tokens_and_root_positions",
        "C5_api_responses_mapped_and_token_sequences_reconciled"
      ]
    },
    "correct_B4_per_request_trace": {
      "rc": 0,
      "all_pass": true,
      "failed_checks": []
    },
    "wrong_request_state_chain": {
      "rc": 1,
      "all_pass": false,
      "failed_checks": []
    },
    "premature_stop_despite_matching_child": {
      "rc": 1,
      "failed_checks": [
        "C4_published_path_bound_to_actual_drafts_and_full_vocab_argmax_per_request"
      ]
    }
  },
  "loop_tail": {
    "loop_sha256": "5684d3e67699a9502c6df61b21ea71d866b78cf8ebc90cdc5103703f27893a0d",
    "cases": {
      "positive": {
        "rc": 0,
        "pass": true,
        "output": [
          "  e1 joiner OK (pilot evidence; see /tmp/b4-loop-tail-_rlkkkj9/positive.e1_join.log)"
        ]
      },
      "missing_ledger": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 LEDGER MISSING for test arm test (recorder on) — stopping"
        ]
      },
      "mapper_refused": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 API TOKEN MAP REFUSED for test arm test (see /tmp/b4-loop-tail-_rlkkkj9/mapper_refused.e1_join.log) — stopping"
        ]
      },
      "join_rc2": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 JOIN FAILED for test arm test (see /tmp/b4-loop-tail-_rlkkkj9/join_rc2.e1_join.log) — stopping"
        ]
      },
      "invalid_rc0": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 JOIN INVALID/INCOMPLETE for test arm test — stopping"
        ]
      },
      "refused_rc0": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 JOIN INVALID/INCOMPLETE for test arm test — stopping"
        ]
      },
      "wrong_count": {
        "rc": 16,
        "pass": true,
        "output": [
          "E1 JOIN INVALID/INCOMPLETE for test arm test — stopping"
        ]
      }
    },
    "all_pass": true
  },
  "artifact_manifest_sha256": "de4a816242f1a11ba8bde7504aafbed94f8bcff0f8d4f489488d267a031bdcdd",
  "artifact_hashes": {
    "cohort/req_0_p072/capture_request.json": "ebe977a2189c4d3c6a6b0eca59598c76f7b77ddcca8a132546372e81a4ee5980",
    "cohort/req_1_p017/capture_request.json": "38f65eaf163434a5b0b0420cfb88d923275e449ddc5df1c0c4b174c30d2b4afe",
    "cohort/req_2_p085/capture_request.json": "a970cd4a6612eb466700697f644b1db997647f4ac6b901525ad1080d7ca9e57e",
    "cohort/req_3_p095/capture_request.json": "24f741a9a232290868b48155db928e64c37d08df2ae8f22b2b67ac546a13d8cf",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step0.pt": "b741371ff8d72a9ffd969df04061f787ddd5f9a657257830064ca541a3cbed39",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step0.pt": "864ef693d97ffe137503cb505614054a3ede30b40be9b7bfe8b8fbf922e364bf",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step1.pt": "e9b4b29a04b4e508939791b60c72f65580bc22aaee6948c02896d7e8d5293d3b",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step1.pt": "2a3ba448722f29a4e2d4e2f466aaa9e8aa65b4bf1175ca8924f81dbe47103112",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step2.pt": "14721394ce3e1496492dd301444b7d9323a29b5fc3421365d8b1cac842c98476",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step2.pt": "9b0f49ae7088395e9ea6b3967dd6eb18e29c782744021ca2d7f1a68ae73559e4",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step3.pt": "9dff2af780fa8e793713a39ca45dad2bea750e1dca6b0486d2e583cc77365cb5",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step3.pt": "3652d147074955f40b07fcc2a7bdf6f66d6989106da3ce3bd1898a4f3015bdf2",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step4.pt": "fa239134257a1c6ecbf14bbf3f8a3598f262e26824452aa5e195461b921cae68",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step4.pt": "34b9a9f2e79d81cb5d869ffc68ffcd10f1f1c08ec943111f1a424082366a8cc7",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step5.pt": "9d55f9fbab3cc99a45bc847ce097600c6c6cca43212e195848f743be0a0a1057",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step5.pt": "e7359d644bf7eafffd41d1ca70d9a5af3fe50b2aff56623a93f9aa5d971ff8f3",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step6.pt": "15d4c6a72e233bddc1875585d81d757035c2ad4d8d6abbdfc6d333875777c834",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step6.pt": "d0bed4b1b3393dee5bce575dc5f3f1c78c90096b41e69bda1dfd4864433d2132",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step7.pt": "896ebece16218035af50888bdd23804aaceeb1a0588ae4fb078575ec4b3a4e23",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step7.pt": "e6e32322f95c67c135a650ea9cc6290aa5958fd345c61a5eb68118df2c175864",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step8.pt": "ab943ea68c5b09ac0763e89d742ab3d49ffddc4f9a50eddb1138c633fecb3202",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step8.pt": "91b438f6a77d75623320a7cdd37da7fb998bcedfdae0f313589c2314835d3eb3",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step9.pt": "8e1dc0078a1e0c44d1bf0cf6cb0ea1327d9772da1016625ec41deaf94cf25f6d",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step9.pt": "6e4a8359f0d77f309a3445002e30dfec2178d98e7cdf5d3e2681de4fe5023ab1",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step10.pt": "463e7ecaa14177873bbc9f5b1830f000e55a315b9c05485ddf310113791522ff",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step10.pt": "4c98efbfb6454147f25493391a5d0b16504fafee181a5c0babafea1bd6614904",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step11.pt": "cb0c7427487b5f2d74dcd7647ec0ec86a772020564d2138b53ecb24a44c14a6a",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step11.pt": "2598bf0e1af1d536677b39f66dbf7d9e3f8214d0ef27da920e50ed2cd0192568",
    "fixture_none_fidelity.req0.json": "353fff81e8fd27015c60606b697d95217b7539f16c2aeffe634d6a257cb3ec25",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step0.pt": "7c82740cafe8060b03a52f9e16d92da824b92dd82fcabbcbe0d36b91c63f6213",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step0.pt": "14c369753ab28e035acab62aa9aa673d02d020ac6d86f80535593d6c60014ae6",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step1.pt": "2b30a0bb1f2fdea5a70e556e4912b8bceb33e4b37d58c7b8c6064bfd4d587f3a",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step1.pt": "e7a898f667d12bb157993b571f676b903c62177a3cb4153f97d6afb977583a48",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step2.pt": "899c83ad57a6971a43f3441e79703ae4eb8a781e38d114c7aff76af27b7e5a77",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step2.pt": "2f16b6a2406390d92739a0be9e88b42598d265dfcf6dbb790fbfe8ac6617af7b",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step3.pt": "521fc4a9c50f0f3c9459e39f242a401a5ceed70fe4c4bf820108cf571d0ea312",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step3.pt": "5854c6e130501b85933ea853a858ea0a48456df6af236ede91f74f990c902652",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step4.pt": "a8a27104c60c6052d7e7fe94c8677a747f646dbc0f02266169a130c853df292c",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step4.pt": "5641df21d1baa8ac288b7fa56aaac36881c4c05e173d670b155e8b28671d7f7e",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step5.pt": "6218eb264e97df93f88f06ec22d1d2f8f6aa898039294a5e414aaee8482cde57",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step5.pt": "8151d2e4854d8e109f2a168ada16576a7651efa2f214b7ed08440cdc06f6aa4d",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step6.pt": "b0f4f0e41271c60f48b613a0bfae1243efc7e5757151da35a9341e493f6d39e3",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step6.pt": "3c59a8f0793576bcaf8dad4ef94027f449ed9f49fa6068854019ace53c1f90fb",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step7.pt": "13510479eeca629f111c2857fa07d4135a9f1ec25e51d406d603a614d543f44f",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step7.pt": "7e7c2ab8211c051e75087162731dcdaa3becf39b54f2389f9a6d2e19a6fb3ba7",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step8.pt": "62147934a848512bece44f359ab46f99b0187bf6e06def95a4f8fdc325721442",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step8.pt": "f9f8beac554e4e749fde591117f9ef98494c4d920129a5fb8baa504a7787cb0f",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step9.pt": "7a2f6d771d779c80128e777102c38a376dddd6c4395627f37d00935cc6503a15",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step9.pt": "9c9aa829b43bc41af49e03a38d6a2cccaba27f9e3a37643feaba264d01685075",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step10.pt": "516353de4557d0487e8d2aa13ce6e8ca68fc0dd5aaa268043d12e0e59c0831e2",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step10.pt": "1ac7a8742446c0944370ff812552cdd066481ef82568ee8cbf5c6ea5185b7908",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step11.pt": "69b354c0c01b4459ec4377441cb916c56a352f3463d929ce1552683fdcde8b81",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step11.pt": "d88c0ea4b3d7f21ec3dc397f7938704138d650152fe41954f38ccecc38d7c205",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step12.pt": "5727994ea25d077e46f0c1aab245fd6705c7f9fc2a1165d4210336b4de12bf78",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step12.pt": "15d375dd9c37b2e65e2d79a2095d0b0d681601185d26ac4ee7daa60ba23894be",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row1.step13.pt": "e3dbb41e65b9412d743318b0affb2309650170481ea5e615c7f5d80c306c4789",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row1.step13.pt": "1b206c8c358514e87eb59a15ad4d7d2dc72e6c621f12a942060c13139f836e4f",
    "fixture_none_fidelity.req1.json": "259784c4b54ebd362c35158f17a03dcc0d0fcf0c6b38e0d48dd876618b3e4671",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step0.pt": "07ff43e13c54979893a5210aff910d61fdc6d3156c80c0e59e0249e202537164",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step0.pt": "24df9750952a78b48c4f20054ccc2acf9044282364a2825c87dc44e86cf9a44d",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step1.pt": "3b40a8ec9562d306062ed0f16db97a1f14d29737ee240039e91a9704c84638e6",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step1.pt": "c5258a0ec40323591968bc00d8e549f88443bcace1e2e2655168c0094fede1c1",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step2.pt": "ade7392a9fb168b908edc22402063bb6ebd27f3e1e80a88245cca43b60217432",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step2.pt": "7f03b358cba8d918ee03c9aabe554be22660f75bddafdcfdc0622899bbfbcc68",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step3.pt": "a98f8e1eae48170fb7c2fcb9069634c9af9c23284a05bd26c3d9cfcab17eb961",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step3.pt": "e4a733f646745fea16ed7e0c764b932eaedcea251f31dfa8c5d4f8ed8199a60b",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step4.pt": "fca841cfb89465122b3a2ce430a02948f3a9b6d628183fdfc9fc2086e17011d8",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step4.pt": "d4492f56a7210cd2189bce97cbee1edeb7f7d667f91e37e8c024954dbbdd2a70",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step5.pt": "47039cab322139968b820ff436e7f20eb5c8920d45b26dd955b0cecef62f11ff",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step5.pt": "6139da4cde1bf259a74ec1bf7d74c7c3ff1dec8bdebb45359d841179d9b9c558",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step6.pt": "5bf9196b4271fa018c8ba8be9c88e75658450e927d87fee6cc3300620dafda73",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step6.pt": "d1d2912451de4716a041fb0a496f7555b1cc651bda7093269bab5a85b35e57fd",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step7.pt": "e6163c0916c3ce92cf1529da9ffae3e6b76257bd3686db2aab09f48aa99c2ccc",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step7.pt": "d87ae030ff0f0663012a1de9f6371a1367658c44636daa17e8ffe951daa86515",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step8.pt": "8ae077428fce23eb258cf2b881db2c08dae4b5fb225f9519566f261f9c2b3798",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step8.pt": "8d76c3c04381ad06e13eb82066378adb827a37a2c8b26d27df9b7900eb5e18fa",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step9.pt": "b56250c8e93eeb1b615f3466a59730198276909fe845b91d1360aef85879c865",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step9.pt": "625c9888325d831ba220fef85106d0efb4681bc229f717505dc0bb9b146f60dd",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step10.pt": "52381acc15b1a70ac78dcff5619a00a0ba93dc6e4ad63c9c0ced53a4137ba1f1",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step10.pt": "c560b7b663eeefba0ce6a863daa744d66c076869bcf6d2a0b895d78a27c995b5",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step11.pt": "e783e069bb73eb9b7b0513b027a4c8965d5255f6c27e27a02d49d977bab91484",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step11.pt": "16c8652e4c8347b36f6ebf9882f4a69e69d1544a436875275ec5da8db189d753",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row2.step12.pt": "ff36d2f7c556238362dec12c99d279bc9453ed3dff76ef2745ac66f8c6f77948",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row2.step12.pt": "08bdbbc7247e360e14ac9c2fc77740bdc7164d2fd983b2cc16a557605827b713",
    "fixture_none_fidelity.req2.json": "83de4006a62673f3e3283d5778073bfd2cd5c1f228eac7a07c68613eb970ad8c",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step0.pt": "c2ae5e26450a77ad05d6a5dafd914efe6edc7351b6d977bc1cca47d2244de717",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step0.pt": "adde376bc1100ac91beb5c884c1398cb194b88b63c077a350a84b2c892c83b20",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step1.pt": "c91da5db5c9f79d97ecb1ccdbff1e10e36b902ec20ef5132eaba5e312befae5a",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step1.pt": "0351f1e22c1af62ea216dbaa2d08ed4e98470e33b7915cec077c9b907a5e89fc",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step2.pt": "0662ce92d571c57272b3054ea8114e5dfb5ead47dfaab45f1d744e0358ff5369",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step2.pt": "c3fe7d0085d0960e6fe852f400c2bfb13031a185930b49c2e5e5fdf99349af82",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step3.pt": "65805ffb60a98161c798530efb7eea338b810180cbe5c63ed807f81dc2f15bbf",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step3.pt": "c139fa3e5248815ce7dcf33dfa3bae9e3300b230c98e8d1f4444eb75e65a33d7",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step4.pt": "ffbd5508bc0c1c2ae63baa55fb6c697c4b832178d6c4d925eacff29e9a6188c6",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step4.pt": "4f514307381e8714e6f626dde1a3ec88854f49587f14c4343f6021cd21b02a00",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step5.pt": "a6094086b5c059fa0f521b3c4d635ff63361ece33b5eeef1705960716181036a",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step5.pt": "0b1c0e0b96103fa501e0a322196149a61e031c318b4ec13a88258a28d08f3730",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step6.pt": "118e6a19776d7d7959c31700b655f4f449ae5aa3f1735f84cc1baad09c7e8778",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step6.pt": "ee2b169d0272dd088dd30437b4f07d4999abf4dc29e3c7f57e424f96cfcb038d",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step7.pt": "79616eff9c2024ffc336113789d61db783d2ca140e4beb670727ad57ec4ad9c2",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step7.pt": "02ca999169cc3356af6b3b1255beacc5103e06cddf9d015ed6a40bd84ff7f5bb",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step12.pt": "afb8ff82c7c3f22ec845dc5baeeb9fb7f232e4694842eee093c320e934fd7de2",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step12.pt": "1174738674bafdd59bdbc0930a265d826d4fc1de588d412bc82098500c88a15c",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step13.pt": "e8064fbbf487924d1881a76296134ce65111bc8442dec34a567cad1d12a5c2e9",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step13.pt": "8f917f6dcfd73b71291b717615287ef385fff7b29460de975b7ec9b02b1de682",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step14.pt": "e88586af60d51d51262c50f22860f5d24889e4aacee7ffdbcfcdb0fbee03cfb0",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step14.pt": "2aead0b40d0ad53aedf805d378dca08ddd6a4354a73f5240784228299167956a",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step15.pt": "9cacafc77bb0630607a8953009de50bd2434faaed1e0d1114005300f64607f68",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step15.pt": "d6898eb3c0613571f970f6c48d1c14d781ecab5a49e0c3bd2c4d79688fd76cad",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step16.pt": "412505c38dd4dd74dd1efe343243a5818864dbff80aba6841c3723454c8c7ab6",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step16.pt": "4ff3a94c9e219d3c23d46e8cbab085345b67270726f5ebb124759f3f83e53298",
    "logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step17.pt": "170f1e6b8e172a8fb3f7b51d022671069c8197b102dabd498e4dab41bdb9b867",
    "logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step17.pt": "c87118aea28a50a9adbbb3bf31f1f79d17d608529d49d20032638a2f34a773a0",
    "fixture_none_fidelity.req3.json": "8a042fdd22db37d6c9429cb906eccf223b9e6a432b0c15717b663f20a868b2d0",
    "logs/final_logits.call0.pt": "fe8ffced380516e90d5e1c12d96a7d7c5c1c88d5d56ffa7db6c00a0518384a8a",
    "logs/layer_hidden.call0.pt": "58f05a0e2bc9e25e7cb2a4361e0e3411a3ab019f1ee6609739a898215e7394dd",
    "logs/final_logits.call1.pt": "d0afb1a36734e06e3bd5b2c25b17ce9b384c1bef4bb36be98cf7aaf9c9d0cc2f",
    "logs/layer_hidden.call1.pt": "2aaa1fb8085d454aa65c900ae08b3410ff6187fe906b4b2aa65ef772e5af89af",
    "logs/final_logits.call2.pt": "279474c5323637feaa13457c1abe42ff63a4e3aa897eaa425c767c54ecbd79dd",
    "logs/layer_hidden.call2.pt": "6bc8b75556a5bbc47abb16bcde5cd900c66e52f01217a5a734931d9aa1e5a554",
    "logs/final_logits.call3.pt": "bac455925736a1c328f16caaa74ba35d3ac3bb309baa6dc4c486dcadd6ae6e66",
    "logs/layer_hidden.call3.pt": "24413f52f3ba05fc4510aab14616755af514e5756324025cc4c8d2694ed05574",
    "logs/final_logits.call4.pt": "637b5263c0405d55500502776bd9e53441d8fb29a91920ad6628ae7c2d27d4d2",
    "logs/layer_hidden.call4.pt": "3088b23a45c645934e562eb2ddec51515ced0ef8946fccfcf064f6dcdaad6f81",
    "logs/final_logits.call5.pt": "63bd0ad6183d09984a8cd5abb99674401b370b709c83b9b079417a57cee58f4b",
    "logs/layer_hidden.call5.pt": "edca3829091c3633727c085e6d36a1ceff972a66f17a101031395058fda741cb",
    "logs/final_logits.call6.pt": "5c66461570913948a5f719e86903bb70305467cc5878d5fdb8e37b6eec75712f",
    "logs/layer_hidden.call6.pt": "9e0c49da74842169da45251931b826af1277a07db963e68dfb22b6a0a58a59fc",
    "logs/final_logits.call7.pt": "7bef6318a82464cba0745f5df76119cc507fd0f78520e4d5d16005696f87d8bb",
    "logs/layer_hidden.call7.pt": "7fa01206ec7d641f86e51b710912d9df78a77b8c52afa32a63a16222ab81304b",
    "logs/e1_events.jsonl": "cf410506441426904020ac756f60daa23f9d42ba5fca236b37499b16f26ae40b",
    "logs/fr10_mtp_draft_trace.jsonl": "e2f3b8db9234e35afe23f822545edd629e24d06a2ffa14fe236e93f457779f0e",
    "logs/per_req_spec_trace.jsonl": "cf13b1e21f9689724fdf9482257f71557be7744bd0a3271e6478ebb63c622a42",
    "capture_expected.json": "3ebfd35df5ff4426c4c1f8c254b5257bd8934ea19161126c655abae4ec1395f7",
    "operands_gate.json": "ae63cbcd494b95b87e418032bb3d872b9635d7119260c325adb6c3129c25e568",
    "capture_gate.v3.json": "69907abdaa6174e4fade1bac99f8d8d8a2da64b258b91bd50edb537f8ec6e2d9"
  }
}
```

</details>

<details>
<summary>Reproduction: independent real-capture CPU audit</summary>

Run the Python below from the remote repository root with `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -B`. It writes only temporary files and preserves original sources/artifacts. It derives timelines independently, reruns numerical fidelity and gates, and prints the result JSON. Expected: 456 assertions pass; sources stable; both gates and join rc0.

```python
from pathlib import Path
import sys,json,hashlib,tempfile,subprocess,os,importlib.util,re
import torch
torch.set_num_threads(1)
ex=Path.cwd()/"papers/gdn-tree-scan-mlsys/v2/experiments";base=ex/"e2"
run=ex/"out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all";layer="language_model_model_layers_62_linear_attn"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
names=["e2/"+x for x in ("e7b_b4_capture_gate.py","e7b_ledger_align.py","e7b_operands_gate.py","test_e7b_b4_capture_gate_cpu.py","e7b_state_continuation.v2.py","e7b_loop.v4.sh")]+["e7a/e7a_core.py","e1/e1_join.py"]
raw={p:(ex/p).read_bytes() for p in names}
R={"source_hashes":{p:hashlib.sha256(v).hexdigest() for p,v in raw.items()},"run":str(run),"checks":{},"requests":[],"capture_blocks":[],"artifacts":{}}
def require(n,b):R["checks"][n]=bool(b);assert b,n
report=ex.parent/"p0/monitor/e2-b4-api-inversion-redteam.md"
recovered=json.loads(re.search(r"\x60\x60\x60json\n(.*?)\x60\x60\x60",report.read_text(),re.S).group(1))
R["api_recovery_report_sha256"]=sha(report)
ev=jl(run/"logs/e1_events.jsonl");fwd=[x for x in ev if x.get("event")=="output_rows"];fe={x["seq"]:x for x in ev if x.get("event")=="forward_entry"};ps={x["seq"]:x for x in ev if x.get("event")=="physical_step"}
require("seal",ev[-1]["event"]=="run_close" and ev[-1]["n_events"]==len(ev) and [x["n"] for x in ev]==list(range(1,len(ev)+1)))
require("event_health",all(x.get("event")!="error" and not x.get("errors") and not x.get("sink_failures") for x in ev))
api={}
for p in sorted((run/"cohort").glob("req_*/capture_request.json")):
 j=json.loads(p.read_text());aid=j["response_id"];matches={r["request_id"] for f in fwd for r in f["rows"] if r["request_id"]==aid or r["request_id"].startswith(aid+"-")}
 require("unique_api_map_"+j["prefix_id"],len(matches)==1);rid=matches.pop();api[rid]=j;R["artifacts"][str(p.relative_to(run))]=sha(p)
require("api_cohort_four",len(api)==len(recovered)==4)
counter={};timeline={};filekeys=set()
for i,f in enumerate(fwd):
 rows=[r for r in f["rows"] if r.get("num_draft_tokens") not in (None,0)]
 if f["kind"]=="pure":require("physical_order_"+str(i),[r["request_id"] for r in f["rows"]]==ps[f["seq"]]["request_ids"])
 for s,r in enumerate(rows):
  k=counter.get(s,0);counter[s]=k+1;timeline.setdefault(r["request_id"],[]).append((i,s,k));filekeys.add((s,k))
require("all_files_exactly_used",all(len(list((run/"logs"/kind).glob(layer+".row*.step*.pt")))==len(filekeys)==53 for kind in ("e7b_state","e7b_operands")))
spec=jl(run/"logs/per_req_spec_trace.jsonl");sp={};cursor=0
for i,f in enumerate(fwd):
 ids=[r["request_id"] for r in f["rows"] if r.get("num_draft_tokens") not in(None,0)];g=spec[cursor:cursor+len(ids)];cursor+=len(ids)
 require("spec_group_"+str(i),sorted(x["rid"] for x in g)==sorted(ids));sp[i]={x["rid"]:x for x in g}
require("all_spec_rows",cursor==len(spec)==53)
draft=[d for d in jl(run/"logs/fr10_mtp_draft_trace.jsonl") if d.get("event")=="mtp_draft"]
require("draft_forward_count",len(draft)==len(fwd)==19)
with tempfile.TemporaryDirectory(prefix="b4-actual-redteam-") as td:
 t=Path(td)
 for p,b in raw.items():(t/p).parent.mkdir(parents=True,exist_ok=True);(t/p).write_bytes(b)
 sys.path.insert(0,str(t/"e2"));sys.path.insert(0,str(t/"e7a"))
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
 LA=module("e7b_ledger_align",t/"e2/e7b_ledger_align.py");C=module("fixture_v2",t/"e2/e7b_state_continuation.v2.py");fw=LA.forwards(ev)
 require("independent_timelines_match_helper",all(timeline[rid]==LA.request_timeline(fw,rid) for rid in timeline))
 stcache={};opcache={}
 for idx,(rid,tl) in enumerate(timeline.items()):
  ap=api[rid];ids=recovered[ap["response_id"]];out=[v for f in fwd for row in f["rows"] if row["request_id"]==rid for v in row["emitted_ids"]]
  require("api_exact_prefix_"+ap["prefix_id"],len(ids)==ap["usage"]["completion_tokens"]==32 and out[:32]==ids)
  lastrow=[row for f in fwd for row in f["rows"] if row["request_id"]==rid][-1]
  require("api_truncation_terminal_only_"+ap["prefix_id"],len(out)-len(lastrow["emitted_ids"])<=32<=len(out))
  ops={};sts={};prev=None;links=0
  for q,(i,s,k) in enumerate(tl):
   for kind,cache in (("e7b_operands",opcache),("e7b_state",stcache)):
    p=run/"logs"/kind/f"{layer}.row{s}.step{k}.pt";R["artifacts"][str(p.relative_to(run))]=sha(p);z=torch.load(p,weights_only=False,map_location="cpu")
    require(f"identity_{kind}_{s}_{k}",z["layer"].replace(".","_")==layer and z["row"]==s and z["step"]==k);cache[(s,k)]=z
   op,st=opcache[(s,k)],stcache[(s,k)]
   require(f"col0_path_state_{s}_{k}",op["h0_col"]==0 and op["parents"]==[-1,0,1,1,2,2,4,4,6,6] and st["authoritative_row"]==st["running_row"] and st["authoritative_row"] in st["rows_written"] and st["rows_consistent_before"] is True and st["mode"]=="none" and not st["applied"] and bits(st["replay_before"],st["published_after"]))
   require(f"spec_acc_{s}_{k}",st["accepted_len"]==sp[i][rid]["acc"])
   if prev is not None:require(f"bitwise_handoff_{s}_{k}",op["h0_row"]==prev["authoritative_row"] and bits(op["h0"],prev["published_after"]));links+=1
   ops[q]=dict(op,step=q);sts[q]=dict(st,step=q);prev=st
  C.check_chain_identity(ops,sts);fid=C.run_fidelity(ops,sts,torch.float64)
  savedp=run.parent/f"fixture_none_fidelity.req{idx}.json";saved=json.loads(savedp.read_text());R["artifacts"][savedp.name]=sha(savedp)
  require("saved_fidelity_reproduced_"+ap["prefix_id"],fid==saved["fidelity"] and [list(x) for x in tl]==saved["timeline_forward_row_step"] and all(x["pass"] for x in fid) and C.FIDELITY==saved["fidelity_thresholds"])
  R["requests"].append({"prefix":ap["prefix_id"],"engine_id":rid,"prompt_tokens":ap["usage"]["prompt_tokens"],"timeline":tl,"state_links_bitwise":links,"api_tokens":len(ids),"ledger_tokens":len(out),"terminal_trim":len(out)-len(ids),"fidelity_max_abs":max(x["oracle_out_vs_production_max_abs"] for x in fid),"fidelity_max_ulp16_sig":max(x["served_out_vs_oracle_ulp16_max_sig"] for x in fid),"replay_max_abs":max(x["oracle_replay_vs_published_after_max_abs"] for x in fid),"bf16_fraction_min":min(x["oracle_out_bf16_vs_production_bitwise_frac"] for x in fid)})
 print("CPU fidelity 53 rows complete",file=sys.stderr,flush=True)
 capf=[(i,f) for i,f in enumerate(fwd) if fe[f["seq"]]["num_tokens"]==40]
 require("capture_count",len(capf)==len(list((run/"logs").glob("final_logits.call*.pt")))==len(list((run/"logs").glob("layer_hidden.call*.pt")))==8)
 for c,(i,f) in enumerate(capf):
  lp=run/"logs"/f"final_logits.call{c}.pt";hp=run/"logs"/f"layer_hidden.call{c}.pt";l=torch.load(lp,weights_only=False,map_location="cpu");h=torch.load(hp,weights_only=False,map_location="cpu")
  for p in (lp,hp):R["artifacts"][str(p.relative_to(run))]=sha(p)
  require("capture_ids_"+str(c),l["capture_call_index"]==l["capture_saved_index"]==h["capture_call_index"]==h["capture_saved_index"]==c)
  for s,row in enumerate(f["rows"]):
   rid=row["request_id"];stp=next(k for j,ss,k in timeline[rid] if j==i and ss==s);st=stcache[(s,stp)];op=opcache[(s,stp)];parents=op["parents"];depth=[0]*10
   for nd in range(1,10):depth[nd]=depth[parents[nd]]+1
   before=sum(len(rr["emitted_ids"]) for ff in fwd[:i] for rr in ff["rows"] if rr["request_id"]==rid);want=api[rid]["usage"]["prompt_tokens"]+before-1
   require(f"full_positions_{c}_{s}",torch.equal(h["positions"][:,s*10:(s+1)*10],torch.tensor([want+d for d in depth]).expand(3,-1)))
   pr=next(j for j,rr in enumerate(fwd[i-1]["rows"]) if rr["request_id"]==rid);dr=draft[i-1]["draft"][pr]
   am=l["logits"][s*10:(s+1)*10].argmax(dim=-1).tolist();P=[0]+st["path"];E=row["emitted_ids"];terminal=i==max(j for j,ff in enumerate(fwd) if any(rr["request_id"]==rid for rr in ff["rows"]))
   require(f"published_edges_{c}_{s}",all(parents[b]==a and dr[b-1]==am[a] for a,b in zip(P,P[1:])))
   require(f"accepted_bonus_{c}_{s}",E==[am[node] for node in P][:len(E)] and (len(E)==len(P) or terminal and 1<=len(E)<len(P)))
   kids=[j for j in range(1,10) if parents[j]==P[-1] and dr[j-1]==am[P[-1]]];require(f"no_local_early_stop_{c}_{s}",not kids)
   R["capture_blocks"].append({"capture":c,"forward":i,"row":s,"row_step":stp,"request":api[rid]["prefix_id"],"path":st["path"],"emitted":E,"terminal":terminal,"root_position":want})
  del l,h
 overlay=t/"derived_api_run";overlay.mkdir();(overlay/"logs").symlink_to(run/"logs",target_is_directory=True);(overlay/"capture_expected.json").symlink_to(run/"capture_expected.json")
 for rid,j in api.items():
  p=overlay/"cohort"/("req_"+j["prefix_id"]);p.mkdir(parents=True);z=dict(j);z["response_logprobs_tokens"]=["token_id:"+str(x) for x in recovered[j["response_id"]]];(p/"capture_request.json").write_text(json.dumps(z))
 gates={}
 for name,args in [("operands",[str(t/"e2/e7b_operands_gate.py"),str(run),"--rows","0,1,2,3"]),("capture_derived_exact_ID_overlay",[str(t/"e2/e7b_b4_capture_gate.py"),str(overlay),"--cohort","4"])]:
  out=t/(name+".json");p=subprocess.run([sys.executable]+args+["--json",str(out)],capture_output=True,text=True);gates[name]={"rc":p.returncode,"result":json.loads(out.read_text()) if out.exists() else p.stderr[-1000:]}
 R["gates"]=gates;require("gate_reexecution",all(v["rc"]==0 and v["result"]["all_pass"] for v in gates.values()))
 m={rid:recovered[j["response_id"]] for rid,j in api.items()};(t/"api.json").write_text(json.dumps(m));(t/"phase.json").write_text(json.dumps({"phases":{"warmup":[],"timed":sorted(m)}}))
 p=subprocess.run([sys.executable,str(t/"e1/e1_join.py"),str(run/"logs/e1_events.jsonl"),"--api-tokens",str(t/"api.json"),"--manifest",str(t/"phase.json"),"--expect-reqs","4","--json",str(t/"join.json")],capture_output=True,text=True)
 R["join"]={"rc":p.returncode,"result":json.loads((t/"join.json").read_text())};require("join_reexecution",p.returncode==0);R["thresholds"]=C.FIDELITY;R["source_stable"]=all((ex/p).read_bytes()==v for p,v in raw.items())
for p in [run/"logs/e1_events.jsonl",run/"logs/fr10_mtp_draft_trace.jsonl",run/"logs/per_req_spec_trace.jsonl",run/"capture_expected.json",run/"operands_gate.json",run/"capture_gate.v3.json"]:
 if p.exists():R["artifacts"][str(p.relative_to(run))]=sha(p)
R["artifact_manifest_sha256"]=hashlib.sha256(json.dumps(R["artifacts"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
R["n_checks"]=len(R["checks"]);R["all_checks_pass"]=all(R["checks"].values());R["checks"]={k:v for k,v in R["checks"].items() if not v}
print(json.dumps(R))

```

</details>

<details>
<summary>Reproduction: independent gate controls and isolated loop evidence tail</summary>

```python
from pathlib import Path
import ast,sys,json,hashlib,tempfile,subprocess,os,shutil
import torch
torch.set_num_threads(1)
base=Path("papers/gdn-tree-scan-mlsys/v2/experiments/e2")
names=["e7b_b4_capture_gate.py","e7b_ledger_align.py","e7b_operands_gate.py","test_e7b_b4_capture_gate_cpu.py"]
raw={n:(base/n).read_bytes() for n in names}
result={"hashes":{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},"cases":{}}
with tempfile.TemporaryDirectory(prefix="b4-final-redteam-") as td:
 t=Path(td)
 for n,b in raw.items():(t/n).write_bytes(b)
 p=subprocess.run([sys.executable,str(t/names[3]),str(t/"owner.json")],capture_output=True,text=True)
 result["owner_controls"]={"rc":p.returncode,"result":json.loads((t/"owner.json").read_text()) if (t/"owner.json").exists() else p.stderr[-600:]}
 test=t/names[3];ns={"__file__":str(test),"__name__":"review"}
 nodes=[x for x in ast.parse(test.read_text()).body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign))]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(test),"exec"),ns)
 def run_case(name,kwargs=None,mutate=None,operands=False):
  d=t/name;ns["make"](str(d),**(kwargs or {}))
  if mutate:mutate(d)
  if operands:
   out=d/"og.json";p=subprocess.run([sys.executable,str(t/"e7b_operands_gate.py"),str(d),"--rows","0,1,2,3","--json",str(out)],capture_output=True,text=True)
   j=json.loads(out.read_text()) if out.exists() else {"stderr":p.stderr[-400:]}
   result["cases"][name]={"rc":p.returncode,"all_pass":j.get("all_pass"),"per_request":{k:{"n_states":v.get("n_states"),"timeline":v.get("timeline"),"failed":[c for c in v.get("checks",[]) if not c["pass"]]} for k,v in j.get("per_row",{}).items()},"other":j.get("stderr")}
  else:
   rc,c=ns["run"](str(d));j=json.loads((d/"g.json").read_text()); result["cases"][name]={"rc":rc,"checks":c,"failures":[x for x in j["checks"] if not x["pass"]],"nonmax":[x["detail"].get("diagnostic_non_maximal_accepts") for x in j["checks"] if x["check"].startswith("C4")]}
 run_case("independent_positive")
 for name,kw in [("capture_identity",{"call_idx_999":True}),("duplicate_hidden",{"copy_hidden":True}),("wrong_operand_identity",{"bad_operand":True}),("wrong_state_layer",{"bad_state_layer":True}),("wrong_draft_tokens",{"zero_drafts":True}),("nonterminal_partial",{"nonterminal_partial":True}),("wrong_request_order",{"swap_order":True}),("unmappable_api",{"unmappable":True})]:run_case(name,kw)
 run_case("correct_B4_per_request_trace",operands=True)
 def wrong_chain(d):
  p=d/"logs/e7b_operands"/(ns["LF"]+".row0.step1.pt");z=torch.load(p,weights_only=False);z["h0"]=torch.ones_like(z["h0"]);torch.save(z,p)
 run_case("wrong_request_state_chain",mutate=wrong_chain,operands=True)
 def premature_stop(d):
  p=d/"logs/fr10_mtp_draft_trace.jsonl";ds=[json.loads(x) for x in p.read_text().splitlines()]
  logits=torch.load(d/"logs/final_logits.call2.pt",weights_only=False)["logits"];ds[3]["draft"][3][0]=int(logits[30].argmax())
  p.write_text("".join(json.dumps(x)+"\n" for x in ds))
 run_case("premature_stop_despite_matching_child",mutate=premature_stop)
 result["source_stable"]=all((base/n).read_bytes()==b for n,b in raw.items())
print(json.dumps(result))

```

```python
from pathlib import Path
import tempfile,subprocess,os,sys,json,hashlib
src=Path("papers/gdn-tree-scan-mlsys/v2/experiments/e2/e7b_loop.v4.sh").read_text()
start=src.index('    if [[ "\x24{E1_RECORD_ON:-}" == "1" ]]')
end=src.index('    [[ -f "\x24RUN/logs/e7b_substitute.jsonl"',start)
tail=src[start:end];out={"loop_sha256":hashlib.sha256(src.encode()).hexdigest(),"cases":{}}
stub="""#!/usr/bin/env python3
import os,sys,json
case=os.environ['TEST_CASE'];a=sys.argv[1:]
if a[0].endswith('e1_api_tokens_from_capture.v2.py'):
 if case=='mapper_refused':sys.exit(1)
 json.dump({str(i):[i] for i in range(3 if case=='wrong_count' else 4)},open(a[3],'w'));sys.exit(0)
if a[0].endswith('e1_join.py'):
 j={'invalid': 'bad' if case=='invalid_rc0' else None,'refused':'bad' if case=='refused_rc0' else None,'n_usable':7}
 json.dump(j,open(a[a.index('--json')+1],'w'));sys.exit(2 if case=='join_rc2' else 0)
os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON']]+a)
"""
with tempfile.TemporaryDirectory(prefix="b4-loop-tail-") as td:
 t=Path(td);(t/"python3").write_text(stub.replace("#!/usr/bin/env python3","#!"+sys.executable));(t/"python3").chmod(0o755)
 for case in ("positive","missing_ledger","mapper_refused","join_rc2","invalid_rc0","refused_rc0","wrong_count"):
  run=t/case;(run/"logs").mkdir(parents=True);(run/"script_snapshot").mkdir()
  if case!="missing_ledger":(run/"logs/e1_events.jsonl").write_text("test")
  env=dict(os.environ,PATH=str(t)+":"+os.environ["PATH"],TEST_CASE=case,REAL_PYTHON=sys.executable,RUN=str(run),E1_RECORD_ON="1",E7B_COHORT="4",pid="test",arm="test")
  p=subprocess.run(["bash","-c","set -euo pipefail\n"+tail],env=env,capture_output=True,text=True)
  out["cases"][case]={"rc":p.returncode,"pass":p.returncode==(0 if case=="positive" else 16),"output":p.stdout.strip().splitlines()[-1:]}
out["all_pass"]=all(x["pass"] for x in out["cases"].values());print(json.dumps(out))

```

</details>

