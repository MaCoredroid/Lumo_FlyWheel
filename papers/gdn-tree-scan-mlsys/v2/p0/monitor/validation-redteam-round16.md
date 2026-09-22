# Round 16: continuation and targeted E2 fixture review — PROVISIONAL

**Latest status: the original CPU-fixture findings are closed by the final bounded recheck below. Earlier provisional sections are retained as the audit history. Whole-model E2 remains separate.**

Reviewed remotely on 2026-09-22 under `/home/mark/lumo-paper-v2-20260921`. This report is provisional because Claude changed the source/test pair during the audit. Parent requested a pause in source execution pending a settled pair. No GPU, inference, container launch, tmux action or shared-source edit was performed. CPU executions used hidden CUDA devices, disabled Python bytecode/cache writes and disposable fixtures. The sole persistent output owned by this review is this local report.

## Disposition

The corrected immutable three-boot commit capture remains authorized as diagnostic data collection. The first continuation revision reviewed here is **not qualified** to interpret real-arm continuation. Reproducible CPU mutations bypass its claimed path/state fidelity checks. A later source revision addresses several findings, but its tests were not yet synchronized, so no final verdict on that revision is warranted. The layer fixture remains a layer recurrence on fixed recorded operands, never full-model E2 or convolution/KV qualification.

The targeted E2 test changes mostly repair stale fixtures while retaining the current implementation's controls. The last completed stable CPU suite reported **65 passed, 1 failed**: an obsolete `forced_spine_commit` string assertion remained in the pristine-patcher compile test. Independent graph-registration and unified-committer negative controls passed. No assertion was removed or altered by this reviewer.

## Continuation: reproducible findings on the stable first revision

Source `experiments/e2/e7b_state_continuation.py`, SHA-256 `a9d9d95a7ea58619d9fd928a015c3fd2cee8c37849872303a37642315063a559`; tests `6fc988b42a5d911a821b95d34ce4db52c071023eb771fe3a9bd44e4c08395626`. The source hash was unchanged across the initial reads and mutation runs. Line references in this section refer to that exact source revision.

| Required property | Exact implementation and independent CPU result | Minimum repair |
|---|---|---|
| Enforced reference fidelity | `run_fidelity:52–68` only reports measurements; `main:122–132` proceeds. Adding `1.0` to reference step-1 `h0` produces output bf16 agreement **0**, previous-publication agreement **0**, replay error **1.3796221017837524**, yet exits 0 and emits horizons at steps `[1,2,3]`. | Make failed identity/state-chain fidelity refuse interpreted continuation; retain measurements as diagnostic evidence. Freeze the declared numerical comparison class. |
| Valid accepted lengths | `path_nodes:42` slices before validation. Setting accepted length to **−1**, or to **5 with only 2 path entries**, in both pre-injection state records still emits all horizons and exits 0. | Validate nonnegative length, available path entries and real-tree bounds before slicing, for all state records used at or before injection. |
| Consecutive recurrence | `run_continuation:96` iterates sorted available operands; missing states simply break at 98. Deleting reference step-2 operands yields steps **`[1,3]`** and exits 0. | Reject a gap, or terminate with an explicit incomplete-boundary result before it. Never count a skipped transition as a continuation horizon. |
| Actual capture identity | `comparable:70–88` checks path/length, physical row fields and API prefix, but not actual draft-token/position/operand-tree identity. The in-memory reducer accepted altered arm layer/row metadata and accepted both records with `authoritative_row=999` despite their running row being 5. | Bind each capture to layer/request row, actual tree tokens/positions/topology and its own scan/publication row contract. Equality of two inconsistent metadata records is insufficient. |
| Honest fp64 carry | `payload_of:34–36` still casts to fp32 at every lift, although `run_continuation:90–109` and report:119–120 introduce a supposedly exact float64 carry mode. Earlier exact probe maps fp64 `[1.0,1.0000000009313226]` to `[1.0,1.0]`. | Pass through h0 precision or remove the float64-carry option. Keep arithmetic precision separate from the stored fp32 state boundary. |

The identity mutation probes operated on loaded dictionaries, then invoked the real `main` with in-memory loader/API providers. They directly test the reduction contract; they do not claim an arbitrary wrong `step` value would survive the real loader, which keys records by the payload's step. The observed layer/row and internally inconsistent physical-row cases are the relevant omissions. Inputs and reports remained in memory or disposable synthetic directories.

Useful repairs already present in this first revision: missing API IDs refuse comparison; API length must cover the complete cumulative boundary; state row fields must exist and match; `rows_consistent_before` must be true; visited path nodes must follow the recorded parent chain. These checks should remain intact.

Reproduction recipe: execute the definitions before the top-level fixture block in `test_e7b_state_continuation_cpu.py`; use its `make_run` with paths `[[1,2],[1],[1,2,4,6],[1,2]]`, IDs `100..139`, and an arm perturbation at step 0 of `2e-6`. Load the four operand/state dictionaries. Change one field per row above, deepcopying the baseline for each case; call the real fixture `main` with `ref --arm arm --inject-step 1`, redirecting JSON output to memory. For the fidelity mutation use `ops[1]['h0'] += 1.0`; for the sequence mutation remove `ops[2]`. A valid control emits steps `[1,2,3]`, output bf16 fidelity 1.0, h0-chain fidelity 1.0, and replay error 0. The malformed controls above also emitted continuation, which is the failure.

## Later revision seen during review — not yet qualified

The next source hash was `f6df85816c7792930089ef98edf6587f763c7453cd4cd7f5a703a9447beccc90`, while the test file still had hash `6fc988b4…`. Read-only inspection showed:

- `payload_of` now preserves h0 dtype.
- `path_nodes` now validates accepted length before slicing.
- `FIDELITY` thresholds and a default refusal gate were added, with an explicitly diagnostic `--allow-fidelity-miss` option.
- Draft-token and full-position comparisons now call existing reducer helpers.

The attempted supplied-test run encountered this new source with the old test: its initial synthetic fidelity case failed the new declared class, then the test crashed with `KeyError: 'horizons'` because it expected unconditional continuation. This is a **moving source/test pair**, not evidence that a settled implementation regressed. Do not lower a real-data tolerance just to make an arbitrary synthetic scale pass; align the synthetic fixture with the intended declared class and retain a deliberate outside-class negative control.

After the parent signaled the pause, no further source execution was performed. The settled recheck should target the concrete original mutations above, actual draft/position binding, internally consistent state-row metadata, and carry precision. It need not add a GPU campaign. Any retained diagnostic override must remain visibly unqualified in its output. Preserve the immutable capture snapshots and previous checker outputs.

## Targeted E2 test repair review

Three changed files were reviewed against `git diff` and current implementation source:

1. **`tests/test_fr13_b4_gdn_bv64_production.py`** changes only the expected launcher refusal wording. The same protected environment overrides still require exit code 2. This retains the test's meaning; no credential/source/batch gate is weakened.
2. **`tests/test_fr13_conv_committed_path.py`** supplies missing globals to extracted fragments, updates wiring assertions for the unconditional gather and column-zero running-state convention, and moves old accepted-leaf semantic tests to the explicit `FR13_TREE_RUNROW_INIT=0` route. A new default-route test checks column-zero selection across zero, branch and longer accepted paths. These changes correctly distinguish current served behavior from the retained escape-hatch unit semantics. The old forced-spine behavioral tests targeted a function already deleted from HEAD, so removal is justified only as retirement of that implementation, not proof of equivalent modern behavior. The unified committer's fail-loud check remains. The pristine compile test still fails on `assert 'forced_spine_commit' in sampler_text` (line 515 at the reviewed hash); replace the obsolete positive-field expectation with the actual emitted refusal/unified-committer contract. This is an assertion repair, not a reason to resurrect the deleted forced-spine mode.
3. **`tests/test_fr13_fixed32_gdn_batch_graph_gate.py:151–167`** registers records inside an actual CUDA capture context when CUDA exists, matching the production registration precondition. It retains the 48 unique layer/signature assertions. The fixture's record callbacks still operate on CPU tensors, so this is a registration-contract test, not evidence of real GDN graph correctness or performance. With CUDA hidden, the CPU branch passed.

Completed suite command (run in a Python `TemporaryDirectory`, passing its child as `--basetemp`):

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  -p no:cacheprovider --tb=short --basetemp <isolated-temp-child> \
  tests/test_fr13_b4_gdn_bv64_production.py \
  tests/test_fr13_conv_committed_path.py \
  tests/test_fr13_fixed32_gdn_batch_graph_gate.py
```

Result: **65 passed, 1 failed** at the stale string assertion. An earlier invocation using pytest's default temp root hit the pre-existing `/tmp/pytest-of-mark` symlink guard (27 passed, 39 setup errors); the isolated temp base resolved that environmental problem without modifying the symlink.

Independent CPU negative controls executed the actual kernel registration functions and extracted actual unified committer, with CUDA predicates mocked only as booleans (no CUDA call or GPU execution):

| Control | Result |
|---|---|
| CUDA-visible registration outside a capturing stream | Rejected with “record was not captured by CUDA” |
| Finalization with 47 records | Rejected |
| Finalization with 48 records containing a duplicate layer key | Rejected |
| Finalization with 48 unique records | Accepted |
| `FR13_FORCE_SPINE_COMMIT=1`, `all_greedy=False` | Rejected with the named refusal |
| Same flag, `all_greedy=True` | Rejected with the named refusal |

The final two controls demonstrate the current fail-loud semantics; the repository test presently exercises only the default sampled argument. Adding the greedy parameter is a small durable negative control, not a new research experiment. Existing byte-gate source compares the point-mass route to an independent greedy walk; that gate was read, not rerun or promoted to fresh evidence here.

## Bounded next step

Once the owner declares the continuation source/test pair settled, rerun the provided CPU controls and the concrete negative mutations, then rerun the three E2 fixture files after the obsolete assertion is repaired. A pass qualifies these fixtures for the already authorized diagnostic interpretation; it does not itself qualify whole-model E2. Keep the minimal corrected commit capture, its horizon-1 publication/next-read identity check, and sequential E2→E1 as separate evidence steps. No mandatory combined-method arm, wider prefix sweep, B4 graph campaign or longer continuation horizon follows from this audit.

## Reviewed hashes

Paths below are relative to the remote repository root unless stated otherwise.

```text
a9d9d95a7ea58619d9fd928a015c3fd2cee8c37849872303a37642315063a559  papers/gdn-tree-scan-mlsys/v2/experiments/e2/e7b_state_continuation.py [first revision, mutation-tested]
f6df85816c7792930089ef98edf6587f763c7453cd4cd7f5a703a9447beccc90  papers/gdn-tree-scan-mlsys/v2/experiments/e2/e7b_state_continuation.py [later revision, provisional]
6fc988b42a5d911a821b95d34ce4db52c071023eb771fe3a9bd44e4c08395626  papers/gdn-tree-scan-mlsys/v2/experiments/e2/test_e7b_state_continuation_cpu.py
0c16cb61653f50cd4142fcbc72a7a1cd6fe3b8d8a8dc4a9972e890a604403101  tests/test_fr13_conv_committed_path.py
8bcfeecba139d0a12548d285fae5e51d0648ab5d91b788c1c7a2da0c37923269  tests/test_fr13_fixed32_gdn_batch_graph_gate.py
58e5a6a604a968234077524ff774bc2c91feaa2e5a3d7c17fa25eabd42aec8dc  tests/test_fr13_b4_gdn_bv64_production.py
```

## Settled-pair recheck: most findings closed; two identity checks remain

At the parent's request, performed **one** recheck after `p0/monitor/2026-09-22-claude-reply-16.md` declared the pair ready. Source `bb3f2408875886866c44dd41acd4481740e3ed157c7ba118e2ee7bab892b3299` and tests `0aeb7a0b86000d09c8e8feef0d278bec62694e49567ab9f1e8ab94f671e27c4a` were stable before and after execution. This section supersedes the provisional test outcomes above; the earlier findings remain preserved as history. No live capture/model was inspected or exercised.

**Verified closures.** All **27/27 supplied continuation controls pass**, including valid nonzero/zero perturbation controls. I independently repeated the earlier mutations using real temporary files and the actual `main`, without substituting loaders: wrong next-h0 now refuses with exit 5; negative/excess accepted length, missing intermediate operands and invalid authoritative publication row refuse with exit 4; altered arm file layer/row/step identity, mismatched arm authoritative row and equally short API prefixes produce `not_comparable` without horizons. The untouched reference passes fidelity and emits all three synthetic horizons. `payload_of` preserves the exact fp64 pair `[1.0,1.0000000009313226]`, closing the silent rounding defect. Carry rounding now occurs at its declared boundary.

All **66/66 targeted E2 CPU tests pass** (55 conv/bv64 tests plus 11 graph tests), using the same isolated temporary-base command above with CUDA hidden. The updated conv-test hash is `bf7ce331752a499810ee8f7f9299de685c1fb1c3a7c0ee3a5580435d145fb514`. Its pristine-patcher check now rejects the obsolete function/log field while requiring the unified refusal; its negative behavior check covers both `all_greedy=False` and `True`. This repairs the obsolete assertion without hiding a live implementation failure. The graph and bv64 test hashes remain those listed above.

**Two remaining binding defects, both within the original identity requirement.** Line references below are for the stable `bb3f2408…` source. Both were reproduced after making one small mutation to the supplied synthetic fixture's real files; both return exit 0, `fidelity_pass=true`, and interpreted continuation horizons.

| Remaining requirement | Actual reproduction and source | Minimal fix before interpreting a real-arm result |
|---|---|---|
| Actual tree topology must match | In `arm/logs/e7b_operands/<layer>.row0.step0.pt`, change `parents[3]` from 1 to 2, preserving the accepted `[1,2]` path, all API IDs, draft tokens and positions. Horizons still emit. `main:191–198` loads only arm state; `_bind_drafts_positions:110–119` compares drafts/positions but no parent array. The comparator therefore cannot establish that those indexed tokens form the same tree. | Load/bind the arm operands for the compared prefix and require actual `n`/parent topology to match the reference and the recorded capture topology. Validate the accepted paths against those actual trees. This is a CPU reduction change; existing captures suffice. |
| Position identity must be complete | In **both** runs' `logs/layer_hidden.call0.pt`, replace `positions` with `positions[:,:1]`, changing shape **3×10 → 3×1** while retaining ten row IDs. Horizons still emit. At 117–118 only truthiness/equality is required, so identical incomplete records pass. | Require the expected position-axis count and complete per-axis coverage of every recorded row/tree node before equality comparison. Equal truncated captures must refuse. Existing complete captures need no rerun. |

The actual mutation harness used the supplied `make_run` to create four-step synthetic reference/arm directories, with the arm's step-0 state perturbed by `2e-6`. It deep-copied these directories per case, edited only the named tensor payloads with `torch.load/save`, and called the unmodified fixture `main` with `--arm ... --inject-step 1`, collecting output in memory. Positive, negative, and carry controls used the same stable source. This is a direct reproduction of the file-based interface, not an arbitrary malformed in-memory dictionary.

**Bounded disposition.** Path-length handling, sequential-step handling, the declared fidelity gate, capture-file metadata, authoritative-row comparisons and carry typing are now supported by passing positive and negative controls. The fixture still needs the two small identity checks above before its “comparable” output can stand alone as a real-arm continuation result. Alternatively, a particular captured pair can be interpreted only after a separate explicit audit establishes those exact topology and complete-position invariants; do not call that a general checker pass. The already authorized immutable three-boot collection can continue. No new GPU run, broader prefix set, combined arm or extended horizon follows from either remaining defect. Whole-model E2 and sequential E2→E1 remain separate gates. No further source re-execution was performed after this one recheck.

## Final bounded recheck: original CPU-fixture findings closed

The owner subsequently supplied a new settled pair and authorized one recheck limited to the two remaining identity mutations plus a positive check on the newly captured none arm. Reviewed source SHA-256 `9aaa30503c0d7c114c1e9492aab7ac253a8b5a840b21cda9426f81b62226301d`; tests `b9a1f2babdf79aa242208fc19a410f05267b9e8c8c9c0dba4e1e495a5c1cc5b9`. Both hashes were unchanged before and after execution. No source edit, GPU execution or new capture was performed.

**Both exact remaining reproductions now refuse before emitting horizons.** The same temporary-file mutation `arm operands.parents[3]: 1→2` returns `not_comparable: arm tree topology differs at step 0`. The same paired mutation `positions: 3×10→3×1`, with ten row IDs unchanged, returns `not_comparable: ref positions do not cover every recorded row id at step 0 (rows 10, position rows [1,1,1])`. These are expected structured refusals with exit 0 and no interpreted continuation. Source `_bind_arm_operands:122–133` now loads the arm's actual operands and compares tree/scan identity; `_bind_drafts_positions:116–120` now requires position coverage for all recorded rows. This closes the two findings identified by their original reproductions.

**Positive actual-capture control also passes.** Used the completed none arm as both reference and comparison arm, injection step 1, default fp64 arithmetic/fp32 carry, through the actual fixture `main`:

`experiments/out-20260922T071945Z-e7b-r16-p072-commit-batch/e7b_001_p072_arm1_none-B_fs_ieee-all`

| Independent CPU observation | Result |
|---|---|
| Operand/state pairs loaded | 13 / 13 |
| Steps passing declared numerical fidelity | 13 / 13 |
| Recorded next-h0 versus previous publication | 12 / 12 bitwise equal |
| Maximum served-output versus oracle absolute error | 0.00023373506989901593 |
| Maximum significant bf16 ULP distance | 1 |
| Maximum replay-state versus publication absolute error | 0.0000007152557373046875 |
| Offline self-comparison horizons emitted | 12 |
| Self-comparison input-state, output and final-state deltas | Exactly zero at every emitted horizon |
| Declared carry boundary | float32 |

This self-comparison is a positive plumbing/numerical control on a real served capture; it is **not** an independent perturbation result or evidence that compact commit is harmless. Its twelve offline recurrence horizons do not establish twelve fully emitted API continuation boundaries. The final partial capture and whole-model E2 limitations remain as described earlier. No live-model result was manufactured from this control.

Hashed 44 actual none-arm input artifacts: `e7b_spec.json`, `capture_expected.json`, `capture_request.json`, `capture_provenance.json`, `logs/fr10_mtp_draft_trace.jsonl`, all 13 `logs/e7b_operands/*.pt`, all 13 `logs/e7b_state/*.pt`, and all 13 `logs/layer_hidden.call*.pt`. SHA-256 of their sorted index (`sha256 + two spaces + path-relative-to-arm + newline`) is `0c6e8fb672caf05660e7f11d89ffb4301fa7365505d806df4f07b35d729c128a`.

**Closure:** no actionable finding remains within this bounded review of the original CPU-fixture defects. Earlier supplied controls and targeted E2 tests passed at their recorded revisions; this final check deliberately reran only the two remaining mutations and the actual-none positive control, rather than claiming a fresh execution of the entire advertised 32-control suite. The fixture may support the scoped layer-level diagnostics once each real pair passes its capture/provenance and fidelity gates. Compact-arm causal conclusions, full-model state/conv/KV qualification, and sequential E2→E1 require their separate evidence; this closure does not promote any of them.
