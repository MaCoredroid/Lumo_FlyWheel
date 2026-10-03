# M1 qualification paper/source audit

**Disposition: numerical counts, source bindings and claim scope PASS. Two compact definition/precision wording repairs are needed before closing the prose review; no experiment, threshold or comparator change is requested.** This audit is bounded to the new M1 generator/output, added subsection, revised Discussion/appendix and configuration. It does not re-audit the rest of the manuscript or establish PDF layout quality.

## Reviewed snapshot

| File | SHA-256 |
|---|---|
| `scripts/build_m1_qualification.py` | `5f7dc50934cc5f4c6411594166f02408673a708ced56faaa5c3ac0f3b6720564` |
| `results/review-response-20260927/m1-qualification.json` | `ba93cfae27ac922c4635f696d6a720d2c6f1c3ca05160ecd65a40542c00d1afd` |
| `results/review-response-20260927/m1-qualification.tex` | `dea1d5ee45c508ebe2973e7039949fc1e57c222e60fd71ce08fe189a5814ac97` |
| `main.tex` | `a99e869d8af56353ec4b74d9f8feacb20053affb5ac3da7104abe5c8a459af82` |
| `paper.config.yaml` | `f1c805212b49a43f0b997c9e1f5ae832e5b69f317461029a89da2b256b37221f` |
| `p0/monitor/review-response-20260927/M1-Q-C0-PARENT-ACCEPTANCE-v1.json` | `e416dec59907b1c56896ce861c31a183ef4f68221e49cc3c3b57751598526ac6` |

Authority and evidence are the accepted repaired/original raw reviews, numerical receipt `22b5a831…`, and unchanged contract v4.2 `b98c3af2…`. The parent acceptance binds repaired review `e2d9f8d6…` and original review `c53acab4…` exactly. This note reviews the hashes above; a subsequent prose repair should regenerate both outputs and preserve the same evidence bindings.

## Verified data and source behavior

I executed the generator in normal CPU Python with `Path.write_text` intercepted into memory and `Path.mkdir` disabled. Its seven acceptance input hashes and all sixteen surface NPZ hashes passed; generated JSON and TEX reproduced the current output files byte for byte. No manuscript, generator or result file was written. Independent multiplication confirms 48 instances × 28 nodes × 48 value heads = 64,512 output cells; 48 × 3 publications × 48 heads = 6,912 state cells. Four policies give 285,696 unique cells and two repeats give 571,392 observations.

The table reproduces the independently recomputed failure counts per repeat: Lumo 0/0; Weaver author-default 53,745/1,154; Weaver aligned-local 57,707/388; TreeWY author-default 37,528/6,912. Both repeats and all surfaces are retained. The combined result remains FAIL; only Lumo satisfies its own specified comparator envelope. The text correctly states that repeated observations do not increase input coverage.

The generated discussion preserves the important asymmetry: Lumo C1 is the actual native GPU recurrence; author-policy C1s are specified sequential software operators, not emulators of each compact kernel's instruction sequence. TreeWY's BF16 compact commit versus the C1 FP32 outer-product updates is a source-supported example, not a causal attribution proved by this run. It correctly declines algorithm-correctness and task-quality rankings. No comparative timing is introduced.

The scope is accurate: one frozen synthetic seed, 48 recurrent instances, 28 first-verification nodes, uninterrupted publications after 1/6/11 updates, two repeats in one process, fresh S0 each cycle, and TreeWY's final deferred commit. Finite outputs/states, bitwise repeat equality and preservation of every previously completed cycle match the accepted raw audit. The initialization failure and process-owned Backend lifetime-only repair are disclosed without relabeling the failed original run. Full-model convolution/KV/drafter continuation and next-forward logits remain pending. The added Discussion, appendix and configuration consistently narrow the four-policy timing comparison to unreported; they do not promote this result into workload performance or whole-model qualification.

## Minimal prose repairs

1. **Define the cell and both quantities in the bound.** Generator line 77 / generated TEX paragraph 2 currently introduces `E_C0`, `E_C1` and especially `M_C2` without definitions. State that each cell fixes an input instance, node or publication, and value head. For each of the two metrics independently, `E_Ci` is that metric of `Ci-C2`, and `M_C2` is the corresponding RMS or maximum absolute magnitude of `C2` over the remaining output-vector or state-matrix entries. Both inequalities must pass; there is no pooling across heads or instances. This makes the concise formula reproduce `M1-NUMERICAL-CONTRACT.v4.2.json:rule_form` and `m1_c_baseline_v1_1.py:cell_metrics/ref_metrics/rule_bounds` rather than leaving a reader to infer a common scalar floor. Label the author C1s **our predeclared sequential software comparators**, avoiding an implication that the authors supplied or endorsed these comparators/envelopes.

2. **Make Weaver's TF32 statement surface-specific and normalization distinction explicit.** The same paragraph says “Weaver uses TF32”; this is true for its verification configuration (`m1_adapters_v2.py:288–294`), while accepted replay uses FP32 elementwise products/sums and no TF32 (`weaver/chunk_tree_verify.py:894–906`; contract `accepted_replay_state`). A compact replacement should say verification uses TF32 with `bf16_mode=none` and replay uses FP32 elementwise updates. TreeWY's norm floor of 1e-12 is correctly stated. Add that Weaver normalization uses the additive 1e-6 form; its default stores normalized q/k in BF16 and rounds replay beta, while the clearly labeled local variant retains FP32 preparation (`m1_adapters_v2.py:264–282`; `m1_c_baseline_v1_1.py:346–395`). These are different declared arithmetic conventions, not a shared normalization procedure.

No missing result, changed numerical rule, new experiment or GPU action is required by this review. No source/manuscript/gate edits were performed.
