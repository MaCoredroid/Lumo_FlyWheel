# Paper red-team round 1

Audit snapshot: 2026-09-22T06:33:13.823750+00:00. Current manuscript was read directly from `mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2`, HEAD `984f613d7885d9bf7898866827552b68a3199d0c`, main.tex SHA-256 `2dda58d8fc31c68407e8f8324249b3f36bd87662c7eed4729356644247d86097`. All manuscript line references below refer to that snapshot. The v2 directory was untracked; the worktree patcher was modified. No applicable ancestor AGENTS.md was present on either the remote worktree or local report path. No inference, tmux interaction, shared-source edit, or remote write was performed. Historical files unavailable in the remote worktree were read locally only where their hashes exactly matched the remote evidence manifest.

## Verdict

**The scoped systems-study claims remain defensible, but the manuscript and frozen-criteria audit need repair before closure.** No available result supports current-stack acceleration or completed model-level continuation qualification. The new blocking discovery is incomplete evaluation of the original frozen criteria, not evidence that compact reconstruction is algebraically wrong. Existing retained results suffice to repair that discovery; do not spend GPU time rerunning it.

Reviews 12/13 correctly preserve 8 fresh pilot +31/32 provenance-qualified confirmation prefixes, zero qualified E2/E7b combinations, and E1 0/18. The three old E7b boots are diagnostics; their unaligned/confounded comparisons are not science completion. The current E7B_DESIGN repair notes are staged plans/source claims, not independently qualified live evidence.

## Claim–evidence matrix

| Claim and manuscript location | Evidence examined | Verdict / boundary |
| --- | --- | --- |
| Accepted-prefix state contract; KV remap witness (main:86–149) | H1 `FR13_REMAP_SHIP_RESULTS.md:6–28`, hash matched | Keep as an archived repair witness: 15/15 to 0/15; 0/84 and 0/61 script flags. Documentary fixture evidence is not a new runtime qualification or distribution proof. |
| Physical placement affects the shared attention spine (main:153–165) | Archived H2 verdict table and described numerical witness; table hash matched | Existing scoped wording is appropriate. This audit checked the archived acceptance rows, not rerun the original attention witness; no universal bitwise theorem. |
| Tree loses to chain controls in H3 (main:247–285) | All three original `deploy_speed_kvr1.json` files independently read and arithmetic recomputed | Keep. Native5/tree rates 42.7439324 /32.8545038; summed component costs 103.4484686 /160.8792931 ms/event. Historical nominal B4, realized occupancy ~2.74–2.80; source-reconstructed double constraints and model/binary identity gaps remain. Not full service throughput. |
| Separate 27.03 ms H4 optimization (main:287–306) | Hashed closeout README:9–13 | Keep as archived paired summary, separate from H3. This audit did not rederive four individual passes; preserve “reported” interval language and never add H3/H4 deltas. |
| Same-input A/B/C arithmetic, historical/synthetic +fresh (main:314–352) | All 8 pilot and31 confirmation result JSONs, frozen criteria, original checker/summary, explicit source/manifests | Numerical characterization supported. fp32-store A/native identity is distinct from deployed bf16 identity. Confirmation failed and has more frozen misses than reported. B/C remain local published-mechanism reimplementations. |
| Model propagation and accepted-prefix continuation (main:371,381,385) | Reviews12/13, state.json, coverage map, E7B design | Not established. Missing aligned fixed inputs, proper published-state evidence, conv/KV/next-forward and selected deployment-route qualification. Staged repairs do not close the gap. |
| Current performance, service benefit, broad task quality, external-system superiority | E1 0/18; E4/E5/E7c deferred | No claim permitted. Existing text largely observes this boundary; no reason to launch E4/E5/E7c for a scoped systems study. |

## Blocking findings

### P1 — Original frozen criteria are not fully evaluated

`experiments/e7a/check_confirmation_vs_freeze.py:63–73` omits padding invariance, compact-state B/C ordering, and verify-kernel timing half-width, although `PILOT_FREEZE.json.criteria.unacceptable_decision_changes` and `.timing_precision_target` explicitly include them. The freeze's candidate rationale says pilot padding is bitwise invariant; actual pilot-v2 results disagree. Its timing basis calls 0.0623 “met” for a 0.05 ceiling, and records timing statistics from an earlier run while `pilot_harness_summary` points to pilot-v2.

Read-only recomputation from all result files:

| Frozen condition | Pilot of record (8) | Confirmation (31) |
| --- | --- | --- |
| Padding N_PAD16→32 output/factor byte identity | B and C fail on8/8 prefixes | B and C fail on31/31 prefixes |
| Verify timing spread `(p90-p10)/(2*median) <=0.05` | 7 kernel rows across6 prefixes fail; maximum20.2251% (C,p085) | 14 kernel rows across12 prefixes fail; maximum32.0861% (C,p076) |
| B/C equality of compact-state worst absolute error | Holds on8/8 | Fails p031; absolute error differs by about6.89e-8 |
| Existing checker failures | — | Already reports bf16 A/native, fp32 output bound, coverage, output ordering, probe drift; retain every failure |

**Repair:** preserve the original freeze, checker result, and raw files. Add a versioned corrected checker and immutable erratum/side verdict evaluating every original rule; distinguish “frozen criterion already contradicted by pilot” from “new confirmation miss.” Update main:348–350, the frozen-criteria narrative, and source-of-record links. Never retrospectively describe the campaign as passing. No fresh inference is necessary. Timing half-width here measures within-block central spread, not a boot-level confidence interval.

The exact audit script and per-prefix ledger below reproduce these results. The parent independently reproduced the same findings during this review; Claude owns the checker repair.

### P1 — Manuscript contains two inaccurate current evidence statements and one numeric overstatement

- `abstract.tex:2` says E7a “does not test … fresh prefixes”; `main.tex:195` says “no fresh prefixes.” Both conflict with main:318,348,350 and39 executed eligible fresh payloads. Update abstract/status table to name the eight-prefix pilot and failed 32-prefix confirmation, retaining layer62/B1/eager/single-tree-step scope.
- `main.tex:350` says B/C bf16 agreement is “at least99.980%.” The minimum is C on p005, `result_29_layers_62_linear_attn.json → stage1_verify → C_nm[ieee]_out16_vs_native_sg16_bitwise_frac =0.9997721354166667`: **99.9772135417%, 14 of61,440 elements differ**. B's minimum is99.98046875%. Use approximately99.9772% across B/C, or report both separately. Pilot B minimum at main:348 is99.97884115%; prefer “approximately99.9788%” over an upward-rounded hard lower bound.
- `main.tex:358` and conclusion:392 are stale about execution/P0 reconciliation; `review-experiments.md:3–5`, STATUS:87–92,119,298–302 contain superseded planned/count/next-step language. Preserve original plans as dated plans and make one clearly authoritative current ledger. Reconcile the manuscript's34-boots wording with failed/preflight/retry categories rather than leaving “one boot each” ambiguous.
- `main.tex:320` says finite Neumann “d terms,” while main:64 and the actual identity use powers0…d. Say “d propagation steps plus the identity term” or “d+1 terms.”

### P1 for any promotion — E7b cannot substitute fixed-input prefill for state continuation

Reviews12/13 identify concrete reducer/state/sham defects. E7B_DESIGN:95–99 now describes fixes, but only CPU/source validation can be counted until the bounded three-boot validation and artifact review complete. Require identity-bound rows/positions/token ancestry, per-layer/request/step state keys, explicitly cloned pre-state, replay-before/published-after captures, and stage-/realization-matched shams before interpreting substitution differences.

The proposed alternative at `E7B_DESIGN.md:100` re-prefills prefix+continuation as a prompt. It can characterize outputs under equal input tokens, but **does not carry the candidate's post-commit recurrent/conv/KV state through continuation**. It cannot establish error amplification from that committed state at1/8/32 tokens. A supported route must preserve/clone the actual post-commit state and feed identical future inputs, or directly exercise an identity-bound accepted-path/native-next-forward fixture. If unavailable, keep full continuation unqualified; do not erase the guard or rename unforced matching as teacher forcing.

### P2 before artifact release — Make the new evidence retrievable

`ref.bib:242–247` points to commit984f613d, which does not contain the untracked v2 experiment tree. H2/H4 and H3 raw JSON were also absent from this remote worktree; the local copies matched the remote manifest. A paper artifact needs a versioned manifest/bundle that resolves these files and the new result/checker snapshots, with rebuild instructions and the final PDF rebuilt from the reviewed source. This is packaging work, not a new experiment.

## Minimal remaining work and stop criteria

1. **Close the read-only corrections now.** Complete the frozen-checker erratum, regenerate numeric tables from existing JSON, synchronize abstract/body/status, and rebuild PDF. Stop this stage once all frozen rules map to an explicit pass/fail/unavailable outcome and all manuscript extrema agree with source data. A failure honestly reported is a closed paper finding; it need not be converted into a pass.
2. **Validate the repaired E7b instrument only in the authorized three-boot p072 batch:** instrumented none, B-verifier matched sham, B verifier-only. Do not automatically resume the obsolete28-boot sweep. Stop on missing identities, pre-substitution differences that cannot be separated from sham variation, missing after-state, changed topology/configuration, or any guard failure. A validation result is not model qualification.
3. **For a claim about continuation, run the smallest supported state-boundary experiment first.** Baseline A plus at most one selected compact candidate; verifier-only, commit-only, combined with matched shams only as needed to isolate an observed effect. Include zero/full acceptance and one sibling/deep accepted path, recurrent/conv/KV at the actual pending-token boundary, then the same next token. Extend to8/32 only after horizon1 is identity-bound and useful. One repeatable discrepancy can close a scoped failure claim; a pass covers only the exercised configuration. If no supported fixed-state route is feasible, report that limit and omit continuation/promotion claims rather than launch unrelated prefill or task sweeps.
4. **E1 is needed only to fulfill the selected current-stack performance study.** Once the exact sequential baseline/deployment route passes qualification, use the existing18-cell three-arm × B1/B4 × three paired-boot plan, balanced order and common prefixes. Do not wait indefinitely for B/C: the plan permits the sequential tree independently. Add one reconstruction arm only after its own qualification under a newly declared protocol; original E7a remains failed. A reproducible loss is a valid result. Stop rather than expand if route correctness cannot be established; narrow to the historical cost claim. Variance-based replication needs a frozen precision rule and budget, not repetition until a win.
5. **No additional E3/E4/E5/E6/E7c experiments are necessary for the present bounded claims.** E3 only if promoting those optimizations; service-latency, broad quality, quantization and authors-system superiority claims remain omitted. Do not widen the experiment menu merely to make the paper stronger.

Bole and TreeWY already cover parallel GDN tree verification and compact accepted-state reconstruction. Primary-source checks support the paper's non-superiority framing: [Bole §IV/V/VI](https://arxiv.org/html/2608.01651v1) defines the finite series0…d and a distinct SGLang/unquantized system; [TreeWY §4–5](https://arxiv.org/html/2608.20961v1) explicitly separates algebraic correctness, bf16 tolerance and nonidentical streams, using greedy verification and disabled prefix caching. Our fp32/tf32 measurements cannot identify their deployed precision or performance. Novelty should remain the concrete failure witnesses, state-boundary analysis and scoped measurements.

## Reproducible read-only audit

Run this Python snippet on the remote host; it prints the per-prefix data and SHA-256 values without changing any artifact.

```python
from pathlib import Path
import json, hashlib
base = Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2")
runs = ["out-20260922T013313Z-e7a-fresh-pilot-v2",
        "out-20260922T051611Z-e7a-fresh-confirmation"]
timing_keys = ["A_prod_scan_verify_out16", "B_fs[ieee]_verify_out16",
               "C_nm[ieee]_verify_out16", "native_sg_chain_depth5_verify(context)"]
for run in runs:
    rows = []
    for path in sorted((base / "experiments" / run).glob("result_*.json")):
        r = json.loads(path.read_text())
        pid = r["provenance"]["provenance_verdict"]["prefix_id"]
        padding = {k: r["controls"][k] for k in
                   ["B_fs_npad32_vs_npad16", "C_nm_npad32_vs_npad16"]}
        halfwidth = {}
        for key in timing_keys:
            v = r["timing_us"][key]
            halfwidth[key] = (v["sync_p90_us"] - v["sync_p10_us"]) / (2*v["sync_median_us"])
        s = r["stage3_own_factors_own_commit"]
        b = s["B_fs[ieee]_compact_own_vs_oracle"]["max_abs"]
        c = s["C_nm[ieee]_compact_own_vs_oracle"]["max_abs"]
        rows.append({
            "file": path.name, "prefix": pid,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "padding_failure": any(v["out_bitwise_frac"] != 1 or
                                   v["U_bitwise_frac"] != 1 for v in padding.values()),
            "padding_details": padding,
            "timing_halfwidth_failures": {k:v for k,v in halfwidth.items() if v > .05},
            "state_ordering_failure": b != c, "B_state": b, "C_state": c,
            "B_bf16_agreement": r["stage1_verify"]["B_fs[ieee]_out16_vs_native_sg16_bitwise_frac"],
            "C_bf16_agreement": r["stage1_verify"]["C_nm[ieee]_out16_vs_native_sg16_bitwise_frac"]
        })
    print(json.dumps({"run":run, "rows":rows}, indent=2))
# Read-only: emits JSON to stdout; writes nothing and imports no model/GPU code.
```

### Per-prefix omitted-criteria ledger

Result numbers map to `result_NN_layers_62_linear_attn.json` under the named run. Padding entries are B/C output bitwise fractions (factor fractions are also checked by the script). Every row fails padding identity. “State equality” compares B/C compact-own-state worst absolute error versus oracle; it does not imply elementwise identity. Timing lists only original verify-class values above5%.

Pilot: `experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/`

| Result | Prefix | Padding B / C | Failed timing half-width | State-error equality |
| --- | --- | --- | --- | --- |
| 01 | p072 | 0.923682 / 0.944320 | native 9.0518% | equal |
| 02 | p017 | 0.907161 / 0.944710 | — | equal |
| 03 | p015 | 0.912484 / 0.930843 | native 5.1320% | equal |
| 04 | p021 | 0.927604 / 0.947152 | — | equal |
| 05 | p085 | 0.927962 / 0.943522 | C 20.2251% | equal |
| 06 | p095 | 0.916585 / 0.942122 | native 9.4654% | equal |
| 07 | p058 | 0.872754 / 0.886898 | native 6.5731% | equal |
| 08 | p083 | 0.916960 / 0.931071 | A 6.9504%; native 5.0074% | equal |

Confirmation: `experiments/out-20260922T051611Z-e7a-fresh-confirmation/`

| Result | Prefix | Padding B / C | Failed timing half-width | State-error equality |
| --- | --- | --- | --- | --- |
| 01 | p063 | 0.914616 / 0.933073 | — | equal |
| 02 | p059 | 0.908740 / 0.922070 | native 5.8617% | equal |
| 03 | p049 | 0.905729 / 0.942074 | — | equal |
| 04 | p052 | 0.931185 / 0.946924 | native 7.5141% | equal |
| 05 | p046 | 0.912305 / 0.922461 | A 6.9631% | equal |
| 06 | p065 | 0.899772 / 0.930046 | — | equal |
| 07 | p025 | 0.907780 / 0.931331 | — | equal |
| 08 | p031 | 0.909912 / 0.946989 | — | B=2.809754127e-7, C=3.499135786e-7 |
| 09 | p013 | 0.909294 / 0.931250 | native 7.0963% | equal |
| 10 | p040 | 0.923112 / 0.940495 | native 5.3788% | equal |
| 11 | p033 | 0.933838 / 0.962272 | — | equal |
| 12 | p038 | 0.905143 / 0.934570 | — | equal |
| 13 | p060 | 0.942448 / 0.963656 | — | equal |
| 14 | p067 | 0.913802 / 0.938298 | — | equal |
| 15 | p019 | 0.932731 / 0.948486 | — | equal |
| 16 | p003 | 0.913737 / 0.939274 | native 7.1955% | equal |
| 17 | p077 | 0.930355 / 0.949723 | A 7.4442% | equal |
| 18 | p089 | 0.925195 / 0.944759 | — | equal |
| 19 | p030 | 0.914925 / 0.934277 | C 29.4439% | equal |
| 20 | p081 | 0.928271 / 0.943408 | — | equal |
| 21 | p012 | 0.919385 / 0.935107 | — | equal |
| 22 | p092 | 0.934180 / 0.959066 | — | equal |
| 23 | p086 | 0.940544 / 0.958480 | — | equal |
| 24 | p079 | 0.916081 / 0.935173 | B 19.4636%; C 5.3530% | equal |
| 25 | p027 | 0.913298 / 0.922266 | native 6.2148% | equal |
| 26 | p091 | 0.936263 / 0.965853 | — | equal |
| 27 | p082 | 0.921387 / 0.946663 | native 5.1875% | equal |
| 28 | p088 | 0.943783 / 0.966667 | — | equal |
| 29 | p005 | 0.939307 / 0.963281 | — | equal |
| 30 | p076 | 0.933138 / 0.952018 | C 32.0861%; native 6.9456% | equal |
| 31 | p078 | 0.934342 / 0.953678 | — | equal |

## Hashes of reviewed artifacts

Remote paths are relative to the v2 root above; the hash snapshot was collected at 2026-09-22T06:33:13.823750+00:00. Raw result provenance fields and summary records were inspected; raw result numbers were independently recomputed. The start/final manifests are included as hash-pinned companion records, not a new validation of every recorded compiled object. This audit did not rerun GPU kernels.

```text
2dda58d8fc31c68407e8f8324249b3f36bd87662c7eed4729356644247d86097  main.tex
ede112b273147fe736e8e7a736cb3d9157b1e2983cd8c04e1b4fe9d16668f001  abstract.tex
51c788f1ffcaef1cc1b1a7846be1e862cbbc56e58d041c37bfc267d703e22905  ref.bib
dbbd542b45baa1cf4e32d93f84a9aa84887fe1f33a4780fbc2548de8028e5171  review-experiments.md
a5116307c4d89ae9c6a18fa7d1a8da2dab28dc353bd0df74d23915c58ba9ca28  experiments/STATUS.md
9999aecafc19d1a3de981229d8f281e15d730b924713437a9abef4aaed7d2c44  p0/P0-REPORT.md
f155ad9281803a8a73cf95161e4eb2ed2e2d093d35ce7180376a3eb61fc7a91c  p0/monitor/2026-09-22-review-12.md
8b388dc0b5436fb4ea30e70387c64034d23b55802ca40e40d1b7157cc235955d  p0/monitor/2026-09-22-review-13.md
5af44fdf258fd93a307fff3aa3327647a0cb5695457061f5e6203af78ca8a92b  p0/monitor/state.json
bf9296090062bc1db3abbcedb04ddd771a3121a63a6dc39b47c7003b9a606e9c  experiments/e2/COVERAGE_MAP.md
a422149cf40a0064421fab3ee75a4bf7827ed925dbd6fd1e3e54bdfa1762c78f  experiments/e2/E7B_DESIGN.md
21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce  experiments/e7a/PILOT_FREEZE.json
7acdd1ed37af1d0a4ef3452e12ba2424e75beaf4aa89dcf7993fa12658a4272a  experiments/e7a/check_confirmation_vs_freeze.py
6250324f20c9cee93e50013d86673caa9a954696c7b32b3bf4cad10756b69d8d  notes/evidence-sources.json
4cb645cc14d3eb32763e4ea0a0dfe3c7da97f6705c266847748a3618959780e5  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_01_layers_62_linear_attn.json
9d167c3f59fae6e87330e7f6d0247bbd0da5d8f7f7730b860853484e5d9cab4f  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_02_layers_62_linear_attn.json
456ae619c8481fe6bca7c547e5388a50cee1c8cb141a4e30926fc80226ff4f93  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_03_layers_62_linear_attn.json
999e6dcbc3d381c6bc5a2ff48e3cde423c3cb1743f3bc21d58e98b88bb333252  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_04_layers_62_linear_attn.json
c5166c4f4d5d0289a4e19b261a21a77c6b2f36311dfd2567dea71ce2d5950e9d  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_05_layers_62_linear_attn.json
0d0ea8369a22bb94d71b12bdb3b9fa087c13659611bee50a79b79d7cb958cc67  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_06_layers_62_linear_attn.json
bce11c7d1b66fef06548ed538b1d1c0b549d93cec34d8648aba9a31bb7908444  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_07_layers_62_linear_attn.json
f57497b6ee804e4be9eaad0b3e505f1f300d14a1282fa1cbb0636a86cfe9d565  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/result_08_layers_62_linear_attn.json
7744f133df77bf07c0ff9dff31609a462033c5d6e0c14fb86d191bd07d81c4a1  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/summary.json
bbb070813ca0149edee836d89cd0e47fdbc03641ebe10862cedd1221a28327fc  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/manifest.json
1e58abb7c20474ca2818e381c402da43045280214ca588b65cb27b4c5fe85d1f  experiments/out-20260922T013313Z-e7a-fresh-pilot-v2/manifest_start.json
71deaaf2b745e6d145619108815f8b92fda9b1ca961fd1eaf5380678b4965b4a  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_01_layers_62_linear_attn.json
bdcfac7d61923f436528ce013e3ed7d93628f0929fa99b55aaa62b1944791697  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_02_layers_62_linear_attn.json
9ff508647c7a959d4621a548e5db3af7da10a98eedabff1109cbfee99788fc01  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_03_layers_62_linear_attn.json
6e0b4985e8c7121c2944cbec48d0be48ea993fa7b64dd09dc748c391115a8733  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_04_layers_62_linear_attn.json
1b72a4d9983f40273dadc1df277aa1048acdd08b2182aa4d222dadbb4f4cb37c  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_05_layers_62_linear_attn.json
c80ad6a602ba43d5c4e2939063e4bcc5e8c1c56929de475a70c5a7bb3e7f34c4  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_06_layers_62_linear_attn.json
52380838c678c3ca4420585982436a958bb54482eed6f18b0d29f9205e06b572  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_07_layers_62_linear_attn.json
28cf2d0d92e5f87d8c8a1bb830960b60fcd99bd84f9b6ecc611e1cdb4011da24  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_08_layers_62_linear_attn.json
f6f8160e48c55610f9cecb8a7a05c574f5a3a5fe1ff546aea41c4b7b271af0cb  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_09_layers_62_linear_attn.json
60331a814056bb7cfe2fea221857c477d5543454fda8902737656d896998b7cc  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_10_layers_62_linear_attn.json
77207af5dd8533bb530119435034ca43b16fbc99fd2d0d93f96840fa5fee3687  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_11_layers_62_linear_attn.json
bb5bc1f38692ab75bad080fbf12a8d066d41b8a42c813bfdcffca73c8b7b422d  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_12_layers_62_linear_attn.json
0a3730ec89fe9c83338a6df1a782c0fa4cf26a2ee2b8db19ce58de9beb1c487d  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_13_layers_62_linear_attn.json
81aafb05cbea28767b1ba22ff101631a5838a2b652549d7ea9762b84cd1eda4f  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_14_layers_62_linear_attn.json
a74480edf6af87e51bd64dd0e4503b06a391f40e5421ac7075ade10093d6d6f6  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_15_layers_62_linear_attn.json
9499181dd412d548b1cb400bc305c30aa6e32c921cb0cf86184d6b9ee97f0862  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_16_layers_62_linear_attn.json
41ec45209534429720c941494eece04fbdd7519a0e21d9dc3a48479d9214a3d3  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_17_layers_62_linear_attn.json
f3d1af5bce2ecd0a3f066f2fd58a7c3cea5e02a9e716fa7af9bde18a2a0aab88  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_18_layers_62_linear_attn.json
36984121c85b53b5a52c9447e944e37c3147a1dd2298e324594fd3c73ffbf63c  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_19_layers_62_linear_attn.json
d9de498913dd8825746974f869d2a9df2c29d86dcbac880306bb5abf84d983df  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_20_layers_62_linear_attn.json
4ba6fb3cfbd1afddff1ffd829b5f53a5c695a49fcb7e8b45c6f5d77708cf1e02  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_21_layers_62_linear_attn.json
4d88947eb435568c8a6c95d1db6f5dd9420a1b112c77bf964e0f8031d0e82353  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_22_layers_62_linear_attn.json
e8f3e51e54a91d5efe446b8726dc8a9a5fbdd72f1b5fc3a15f6a65bd7df6ecc3  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_23_layers_62_linear_attn.json
1297fba96870cbbac699e51aac15ad24bfe89c8c7910107e77ad71d1136a392f  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_24_layers_62_linear_attn.json
595ae338072ee1b433dca8a508762f8f77d80f36495a3c4b437592d74479b023  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_25_layers_62_linear_attn.json
5f38d4ed1564964924b6c3ca0ba6ab8c2ae5bd97dbbf6d0caf3e64806ac154d2  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_26_layers_62_linear_attn.json
fca828c47da8ffd1e99935f312e885b095b0bf46b263bb961b7b040f45f4ceb3  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_27_layers_62_linear_attn.json
36597bf34b3060d1f5abe2234f5153fa7e43aa73d4d013523995371b97be7dec  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_28_layers_62_linear_attn.json
433278d1ee85fd213b974f4d826a882286bc6ce3c016bd9df06901ed57a63613  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_29_layers_62_linear_attn.json
3aaf6fdf42a3c0acf1b00c6e9054fb130212a01de8c9dd6ac05986c2e08ddc77  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_30_layers_62_linear_attn.json
9b6c306c35c90c4d4c531ae03d751e1439aac02210c1ad55f09e2b860909aa3e  experiments/out-20260922T051611Z-e7a-fresh-confirmation/result_31_layers_62_linear_attn.json
b866b64f08724b08c6410070a1c16265850f5758de0e59ef68f332cf95839afa  experiments/out-20260922T051611Z-e7a-fresh-confirmation/summary.json
b2f4e3bc1f273a0b822d0c5e1c201a8a6a8f5584e1434af6954fab9478c68efd  experiments/out-20260922T051611Z-e7a-fresh-confirmation/manifest.json
f793c5e2ea6546342b7263a514b5c3841572892935792830476d82198e02a10d  experiments/out-20260922T051611Z-e7a-fresh-confirmation/manifest_start.json
f2b8579cf2a84c1c51ce92e895ef1c5e92fccfa89b3671660f1f7d7dbea8780c  experiments/out-20260922T051611Z-e7a-fresh-confirmation/freeze_check.json
```

Local historical paths below are relative to `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel`. Each hash exactly matches `notes/evidence-sources.json` read from the current remote source.

```text
b1b5d28bdb0a0bebbd4c38780a873f237f06a35f716c0734a9726ef2d6f4e068  FR13_REMAP_SHIP_RESULTS.md
fbea3e8dae6a3d3adf99d463ba7031cf8986709a124234164229b975d79a0b84  FR13_SLOT_REORDER_ARTIFACTS/VERDICT_TABLE.md
2ddc816ae913f1dc914a2af0da6d391da0bab4717d07218a6a8ac3ddcb817998  results/fr13_series_closeout_20260815/README.md
88deadb9f1ffac9985f263b985dae0dae16a8b2be09da30b2b076efc22cdee87  output/fr13_kvremap_tail6/native5_control_kvr1/deploy_speed_kvr1.json
acd116d76f6f58ccd6d8f72b668b90f2fabd75cb2da1292a04ed6fa6e2caeafc  output/fr13_kvremap_tail6/native11_control_kvr1/deploy_speed_kvr1.json
bd8b1d65054c852053bf4492754af879769339c639f6d7a43f4e99e7bb41f2d8  output/fr13_kvremap_tail6/kvremap_tail6_kvr1/deploy_speed_kvr1.json
```

Memory was used only to locate the earlier historical claim boundary, then verified against current source and raw H3 artifacts; no current experiment verdict relies on memory.

## Follow-up: staged manuscript candidate

At the parent's request, reviewed the local staged candidate `p0/monitor/paper-build-review1/main.tex` (SHA-256 `712df769d0b7db1a66fe06750fb2aaa7b8db854dc196fbc6e8edd66fd3574a52`) and `abstract.tex` (`e2aa41e1a353592aa0bb8a1d5dc2821bc312c9f24295ba204dbcfe2e0e6523af`) against the remote source of record. The staged changes correctly fix the major evidence-status, omitted-check, numerical-minimum, Neumann-term and qualification claims. This is a source review, not a PDF or corrected-checker closure.

Two remaining material wording fixes were sent to the parent:

- Fresh configuration paragraph (staged main:318): explicitly name **B1, eager, stock TREE_ATTN**, with layer62's first tree step. Those are in the immutable freeze notes; the fresh work must not appear to revalidate the historical forked FlashAttention-2 path.
- Kernel-cost paragraph (original main:352; shifted by added paragraphs in staging): separate **storage/allocation from bytes written**. Raw `transient_memory_bytes` records `BC_compact_factors_U+cumg=396288`, `A_prod_replay_hbm_written_rows(depth+1)=18874368`, `BC_compact_commit_hbm_written_rows=3145728`, and padded16-row legacy state export `50331648`. These are not a measured peak-live-memory comparison. Replace “50MB for any route that materializes every node state” with the tested padded16-node export scope, and label replay/compact write volumes separately.

Minor: staged main:348 retains the upward-rounded hard pilot lower bound99.979%; use approximately99.9788%. No additional experiment is needed for these wording fixes.
