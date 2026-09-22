# E1 measurement preflight — bounded read-only review

Reviewed 2026-09-22. No source changes, inference, GPU jobs, or tmux interaction. Remote root: `/home/mark/lumo-paper-v2-20260921`; paper root below means `papers/gdn-tree-scan-mlsys/v2`.

**Disposition: design only, implementation awaiting worker.** `experiments/e1/` contains only `E1_DESIGN.md`; its lines 12–15 explicitly say the per-event recorder/joiner is not built. `experiments/STATUS.md:99` records E1 0/18, and lines 209–213 distinguish passing historical measurement fixtures from this new recorder. This is a preflight checklist for the already planned 18 cells, not a claim that an existing E1 implementation or result has failed.

## Two concrete plan corrections before implementation

1. **Use a physical-step denominator for aggregate output.** `experiments/e1/E1_DESIGN.md:7–10` keys walls by `(request_id, step_idx)` and defines tokens/s as summed tokens divided by summed such walls. Four co-resident requests share wall time: if each emits five tokens during one 0.24 s physical step, the written request-wall sum gives `20/0.96 = 20.83`, whereas aggregate output is `20/0.24 = 83.33` tokens/s. The former can be labeled a request-time rate, but is not the aggregate metric required by `review-experiments.md:72`. Record `(boot_id, physical_step_id)` once for wall/component spans, then one child row per actual request step. Aggregate output sums emitted tokens across those child rows and counts each retained physical interval once. Keep per-request TPOT separate. The existing distinction is illustrated in `tests/test_fr13_b4_timing_math.py:17–52`; its passing unit arithmetic does not establish the new recorder's event support.

2. **Make the E1 gate match the frozen E1 configuration.** `experiments/e2/E2_QUALIFICATION_PLAN.md:3` qualifies a specific B1/eager/temperature-0/10-node tree route, while lines 22–23 permit all 18 E1 cells after Q1 + J1/J3. That wording alone does not establish actual B4, another tree descriptor, cache path, or stochastic sampling. Bind each cell to the applicable qualified route/flags and identify any still-missing coverage before timing; reuse existing E2 fixtures and the planned boots. If the small tree remains selected, native-11 is a longer-chain control and must not be called depth-matched to it. `review-experiments.md:66` calls native-11 depth-matched specifically **for tail6**. This requires a configuration decision and accurate labeling, not an additional broad experiment matrix.

## Minimal recorder/reducer acceptance checklist

- **Exact token and wall support.** Persist boot/block/arm, physical step, request ID and request step, prompt ID, real active request IDs, prefill/mixed/decode class, timestamps, actual emitted IDs/count, acceptance/draft counts, and inclusion/exclusion reason. The hook must cover native-5, native-11 and tree, even though their commit paths differ. Do not assume accepted + one equals emitted output under EOS/max-token truncation; reconcile the complete per-request sequence with the API token stream and record withheld/pending tokens separately. A “consistent path” alone (`E1_DESIGN.md:7`) is not an exact count reconciliation. Check that every retained request row belongs to one retained physical interval; missing rows must reject/report the affected observation rather than silently change its support.

- **Unambiguous interval edges.** For start-to-start intervals, bind `[start_j, start_{j+1})` to tokens produced by step **j**, not j+1. State what happens at the first/last step, mixed prefill, request-set changes, idle gaps, and B4 drain. Use an explicit final boundary or exclude terminal tokens and their unmatched interval together; report the discarded totals. A same-request next start can bridge other physical work, so it cannot substitute for the physical-step clock. Existing `scripts/fr10_phase4_patch_vllm_tree_gdn.py:35780–35849` already attributes wall deltas to the previous physical step, breaks on changed request sets and mixed steps, and exposes idle rejection. Preserve that accounting intent without importing its global-counter proxy.

- **Actual B1/B4 and matched workload.** Freeze tokenized prompts, context strata, request/cohort ordering, stop/output-budget policy and seeds across the three arms in each block. Log actual active request count on every retained physical step; padded graph shape or `max_num_seqs=4` is not actual B4. Define ramp/drain exclusion in advance and retain occupancy/discard summaries. Native defaults to `MAX_NUM_SEQS=1` (`scripts/fr13_launch_native_mtp_server.sh:75`), tree to 4 (`scripts/fr10_launch_speed_server.sh:38`), so set both explicitly. Record the exact tree descriptor and suffix proposal method; native/tree differing candidates support a stack comparison, not a pure topology claim (`review-experiments.md:66`).

- **Matched component accounting without invented host attribution.** Component spans attach to physical steps and are included only over the same support as their wall denominator. Keep the throughput support independent of optional missing component instrumentation: `E1_DESIGN.md:6,9–10,15` currently alternates between “all three streams” and optional components. Define the core token+wall set and a separately reported component-complete subset if necessary. Shared/overlapping CUDA spans are not additive per-request time. Even matching IDs do not prove `wall − sum(spans)` measures host work: overlap, uncovered device work and asynchronous launch boundaries remain. Report a residual unless direct spans justify a stronger label; no extra timing cells are needed.

- **Cold/warm and observer boundaries.** Freeze one stated warm-up/cache policy within the 18 cells. Separate fresh server boot, kernel/JIT/graph warm-up, prefix-cache warm state and retained decode timing; record resets/cache flags and realized cache hits. Both launchers default to non-enforced-eager and cache off (native lines 77,89–96; tree lines 129,139–163), whereas the current qualification is eager. Explicitly set and post-boot verify the intended choice. Keep operand capture, detailed diagnostics and synchronization-heavy profiling outside timing as `review-experiments.md:68` requires. The necessary event recorder must have the same stated policy across arms and avoid a per-step device sync or large tensor export merely for logging.

- **Paired replication and sampling gate.** Three arms × two actual batch conditions × three boot replicates remains 18 cells. Pair arms by replicate block and frozen workload; balance arm order. Bootstrap paired whole boot blocks separately for B1/B4, preserving all paired arm observations (and any declared within-block prefix clustering), not individual steps/tokens as independent replicates. Three blocks provide limited between-boot precision; freeze the estimand, interval procedure and practical precision target after pilot and before confirmation, report wide intervals and failures, and use the already authorized precision rule for any justified replication (`review-experiments.md:30,68,74`). Save requested and effective temperature/top-p/top-k and confirm the loaded native/tree/bonus constraint paths apply the intended policy exactly once. A temperature-0 correctness boot does not validate nonzero-temperature timing behavior; `p0/P0-REPORT.md:21–25` expressly requires fresh loaded-path confirmation after the historical duplicate-pass finding.

CPU fixture cases for the new joiner can be small: shared B4 wall, changing occupancy, missing/duplicate IDs, off-by-one start-to-start assignment, terminal truncation, mixed/idle rejection, and overlapping spans. These validate measurement bookkeeping; they do not add GPU experiment cells. No external-system comparison, broad task sweep, or new reconstruction arm is required by this review.

## Reviewed identities

- Remote `experiments/e1/E1_DESIGN.md`: SHA256 `a613fb830ad354b1602a535a31f5dfab198319da174e34bd2982fda7da4fe1ed`.
- Remote `experiments/e2/E2_QUALIFICATION_PLAN.md`: `c04ae0e7d455cf3663f1f3a3323d60409add516c2437016f94f1e0af1e08438f`.
- Local `review-experiments.md`: `9c2ccb46360409fdf4c9f2502e286557bcc6610c53d3ae6fa3b58533baad8e62`.
- Local `notes/design/experiment-matrix.csv`: `94309c1eddf3f3743994bd5789ef05afa7cb8b4cebd7a97e2791c4ccabd01f09`.
- Local `main.tex`: `5e579cbf64ed4211efafbbdd9e15acc8ba99a53574162fdc12e6069178255b76` (E1 scope at line 388).
- Remote native launcher: `03b891c404a7df43cb3ed2abb1b8ceab0d021c2d19c644f04c1786768d3ba15e`; tree launcher: `ac689e67a09c67212d4f87bb5584792f666d3fad9a1141f3fb073b1fef8551a8`.

The implementation and its event-support validation remain pending. This preflight does not certify E2, the future E1 instrument, or future acceleration claims.

## Initial implementation review — 2026-09-22, after STATUS 07:28:33

**Not ready for E1 timing.** Unique physical wall summation is implemented correctly, but the initial recorder/joiner does not establish emitted-token support or reliable interval boundaries. The findings below are CPU-reproduced implementation defects, distinct from the preceding design-only checklist. No GPU, tmux, canonical source edit or benchmark expansion was performed. CPU probes used temporary files outside the source tree; the reviewed functions were loaded from source into isolated Python namespaces.

Reviewed remote identities:

- `experiments/e1/e1_recorder.py`: `5855828fd432c72847a101c7d80ebb3a27cbc8e7bad954a4864ea927c3c0deb3`.
- `experiments/e1/e1_event_recorder_shim.py`: `1cd89b4a1a3cbc75fb356dcdb41bc35dd07a68ef9c587562fa74da707a033858`.
- `experiments/e1/e1_join.py`: initially `19f5814e261aafc52c274aa422bf217e235ffe72709e7ff35c7343a0b0b8a5aa`; changed during review to `bf1eedc2f7c55e0f8892311ffe7e4e22b3e5d54bd8d3233c0e49992eb875bec8` (adds optional API count checking). **All counterexample results below used the latter hash**, and joiner line references refer to it.
- `experiments/e1/test_e1_recorder_join_cpu.py`: `01148d4629cae633fe956258356cb8146be3ae40727355eed24cadaa22b7ecb8`.
- `experiments/e1/E1_DESIGN.md`: `5bdffb3e5f97113afa046056a58c6ecc614151adcdc9d2222d620ef3bc84f707`.
- Emitted source root used by the E1 design: `experiments/out-20260922T015656Z-e7b-shim-dryrun/E7B_worktree_patcher_plus_shims/files/`. Its `vllm/v1/worker/gpu_model_runner.py` SHA256 is `9487b658ab8eef802737370e6923aaeae865156625d07fdcd49d3fb2772e9cf4`; `vllm/v1/sample/rejection_sampler.py` is `e01008d857b8cccfe3cb076d432448d8f776716460986b86d63a3764f7d85d65`.

### Required fixes

**I1 — P1: structural counts remain labeled committed output; API count equality is not token identity.** Recorder lines 45–47 emit accepted length/path only. Joiner line 44 derives every token count as `accepted_len + 1`. The optional API check at lines 46–55 compares total lengths only and writes the misleading key `token_identity`; it never compares IDs or establishes which API tokens belong to retained physical steps. A CPU case supplied recorder rows containing `committed_ids=[10]`, then `[11]`, and API IDs `[999,888]`: the reducer returned exit 0 and `token_identity={"a":2}`. It also succeeds with no API evidence. EOS/output truncation is not representable in the recorder, so a structural bonus count can either overcount or make a valid truncated request fail the total-count check. **Fix:** record actual sampled/output IDs with their request/step binding, reconcile them with actual emitted IDs and termination policy, then count only the emitted IDs assigned to retained intervals. Make the evidence mandatory for a measured-output result. Keep any count-only diagnostic explicitly structural and unqualified. The sampled tree source already has `out_rows`/`output_token_ids` at sampler lines 2940–2957; reuse real output data rather than inferring IDs from acceptance counts. Native-path capture must use its corresponding output boundary.

**I2 — P1: mixed steps do not create the ID gap assumed by the joiner.** Recorder docstring lines 5–6, design line 24 and joiner lines 6–8 assume a missing timer index signals a non-pure step. In the emitted runner, `_fr13_sfwd_begin` returns on non-pure work at lines 11394–11399 **before** `timer.begin` at 11403; only `begin` increments `_fwd_next_index` (11091–11092). The shim only hooks `wall_mark`, not `wall_break` (shim lines 40–44). Executing the actual emitted `_fr13_sfwd_begin` AST with a CPU fake timer on pure → mixed → pure produced IDs `[0,1]` and one wall-break call. The joiner then accepted the resulting five-second mixed-spanning interval. Its successor test also ignores changed active request sets and idle policy (lines 41–43). **Fix:** emit an explicit break/epoch on every non-pure step, rejected/timer-missing boundary and relevant route discontinuity, clear the recorder's active step, and require matching epoch plus the declared cohort/idle policy when making an interval. Alternatively issue a true all-forward physical ID independent of the pure-only timer. Preserve the existing previous-step attribution and terminal exclusion, which are otherwise correctly oriented.

**I3 — P1: replay arrays can be assigned to the wrong request.** Recorder lines 36–46 label `_fr13_replay_*` rows with `_LUMO_FA_SAMPLER_ROW_REQ_IDS`. The emitted sampler explicitly converts these arrays into `_LUMO_FA_SPEC_ROW_REQ_IDS` order at lines 3092–3126 before the shim's commit anchors. A CPU case with sampler order `[a,b]`, spec order `[b,a]`, replay lengths `[1,2]` and paths `[[3],[4,5]]` records `a→(1,[3])`, `b→(2,[4,5])`, reversing the true request attribution. Set equality in the joiner cannot detect this. **Fix:** pass the exact matching request-ID list explicitly alongside each replay array at the shim call site; validate its length and uniqueness. Do not re-read another coordinate system's module-global list. CPU-test a permutation and a full-batch/spec-subset case.

**I4 — P1: malformed or incomplete cohorts can still produce a successful rate.** Joiner lines 38–40 compare sets, allowing duplicate request rows; it never validates `num_reqs` against IDs/rows, request `step_idx` progression, orphan commit rows, or complete measurement coverage. It excludes missing-row physical steps and returns success whenever any usable step remains (37–40,59). The counterexample with an incomplete 100-second step and a complete 0.1-second step returned success at approximately 100 tokens/s. The exclusion reason is visible, but success still permits a biased partial result to pass the timing gate. **Fix:** require exactly one row per actual request per physical step, consistent counts and request-step ordering, and reject malformed/orphan data. Distinguish prespecified ramp/drain/mixed/terminal exclusions from instrumentation loss; the latter must invalidate a timing cell rather than shrink its support. Check expected actual B1/B4 on retained intervals and the declared successor cohort policy. Keep raw exclusion counts and durations in the report.

**I5 — P2: recorder error handling can raise into the served path.** `_emit` opens/writes synchronously (recorder lines 14–18); the exception handlers invoke the same failing `_emit` again (27,49). A CPU call to `physical_step` with `E1_RECORD=/dev/null/child` raises `NotADirectoryError` despite the docstring's “Never raises” contract. **Fix:** validate the sink before timing, retain an in-memory failure flag/count when writing fails, and ensure the error path itself cannot throw into inference; incomplete logging must invalidate the run. Disabled calls return before mutation, and the shim's replay arguments in this route are already host lists, so this review found no added CUDA synchronization in those calls. Enabled JSON writes still cost host time: keep recording policy identical across arms and label timing accordingly; compilation alone is not a measurement-overhead check.

### CPU counterexample outcomes

Each fixture supplied `physical_step` + `commit_rows` JSON events with a final successor. Unless stated otherwise, each row had `accepted_len=4`. All cases below returned exit **0**, demonstrating missing rejection/validation rather than a hypothetical input concern.

| Case | Observed reduction |
|---|---|
| B4, four rows, one 0.24 s interval, no token-ID evidence | 20 structural tokens / 0.24 s = 83.3333; unique wall sum itself is correct |
| One active request, duplicated twice in commit rows | 10 tokens / 0.24 s; request counted twice |
| `num_reqs=4`, only one request ID and one row | Accepted as one-row measurement; requested B4 is not enforced |
| Four-request step followed by one-request successor | Prior B4 interval accepted despite cohort change |
| Missing one of two rows on a 100 s step; next complete step lasts 0.1 s | Slow step excluded, successful rate ≈100 tokens/s on only 0.1 s |
| Extra commit record at nonexistent physical ID 999 | Orphan silently ignored |
| Actual emitted pure/mixed/pure control gives IDs 0,1 | Mixed-spanning 5 s interval accepted |
| Recorder IDs 10,11 vs API IDs 999,888, equal count | Claimed `token_identity={"a":2}` |

The existing 10 tests intentionally accept structural counts and missing-row exclusion, and construct an artificial ID gap for mixed work (`test_e1_recorder_join_cpu.py:21–50`). They therefore cannot close I1, I2 or I4. Extend only these targeted CPU controls and repeat the emitted-source dry run after the minimal repairs. Native launcher integration, live per-cell qualification, bootstrap/reduction policy and timing are still separate pending gates, not additional defects inferred here.

## Settled recheck — owner checkpoint 2026-09-22T07:49:42Z

Bounded recheck of the original I1–I5 surface, with independent CPU negatives and the actual shimmed emitted methods. No inference, GPU, source edits or tmux interaction. **Several original defects are closed, but ordinary prefill output, phase selection and terminal sink failure still prevent end-to-end closure.** No broader experiments are requested.

Reviewed settled hashes:

- Recorder `9ffde26874a846c4a1f262cfa7a3afca4f59d42d1711b5cd09cf83e4377fce9e`.
- Shim `06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df`.
- Joiner `ef8b8e4d7aa755d71c409a609b8b66895b433307eb6aaaa5bc6c9128687782f4`.
- Emitted runner after this shim `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40`, against pre-shim runner `9487b658ab8eef802737370e6923aaeae865156625d07fdcd49d3fb2772e9cf4`, as bound in `experiments/e1/dryrun/e1_shim_dryrun_report.json`.
- `E1_FREEZE.md`: `d5c10dbff393b944526d1523bcb07fc99f2777b19151ac0d0cc4f1ed5c90f6c1`.

**Closed at these hashes.** Equal-length wrong token IDs now return rc 2; missing API evidence returns rc 3 without a rate. Duplicate request rows, missing commit rows and orphan rows independently return rc 2. A B4 case still reports 20 API-bound tokens / 0.24 s = 83.3333 tokens/s. A pure → mixed → pure event chain excludes the mixed-spanning interval using the forward sequence, retaining only the later 0.1 s interval. The new output hook passes `req_ids_output_copy` with `valid_sampled_token_ids` from the same `_bookkeeping_sync` call: the earlier SAMPLER-vs-SPEC coordinate error is removed. Its source location consumes existing host lists and adds no explicit device synchronization. Bad-sink handling no longer throws into inference. These close the specific old I2/I3/I4 counterexamples and the throwing part of I5; I1's actual-ID comparison is implemented but has the integration gap below.

### R1 — P1: ordinary prefill/mixed output still invalidates the entire run

The recorder's `commit_rows` requires a pure physical step (`e1_recorder.py:92–93`), but the shim calls it unconditionally at synchronous model-runner output (`e1_event_recorder_shim.py:62–63`). A non-pure forward increments `seq` and writes `chain_break` without `physical_step`. Nevertheless the real `_bookkeeping_sync` returns its first generated token or mixed-step outputs: original emitted runner lines 5019–5038 parse those tokens before the new hook at the 5062 anchor.

Independent reproduction executed the **actual shimmed** `_fr13_sfwd_begin` and `_bookkeeping_sync` ASTs, with CPU stand-ins for tensors/timer and one 256-token prefill producing sampled ID 7. Bookkeeping returned `[[7]]`; recorder events were `recorder_probe, forward_entry, chain_break, error`, with `RuntimeError('commit without a recorded physical step (timer off, capture, or non-pure step)')`. The joiner returned rc 2. The owner's mixed-step test supplies no output callback for its mixed forward (`test_e1_recorder_join_cpu.py:35–37`), so it misses this ordinary path.

**Minimal fix:** maintain an every-forward output ledger keyed by forward sequence and request, including untimed prefill/mixed/discarded rows. Pure physical-step records should reference that ledger for retained-rate accounting. Reconcile the complete API sequence against all output rows, then select only the declared pure intervals for the numerator/denominator. Merely suppressing the non-pure callback drops first-token/mixed outputs from the concatenation and therefore cannot satisfy the full API sequence check at joiner lines 78–91. Add one ordinary prefill → decode and one mixed-output CPU callback control using the emitted methods.

### R2 — P1 before confirmation: the reviewed reducer cannot select the frozen timed phase

`E1_FREEZE.md:23` requires two untimed warm-up requests before the timed cohorts. The recorder runs wherever `E1_RECORD` is enabled, with no request/phase classification; the joiner has no timed-request/phase manifest input (`e1_join.py:21`) and reduces every eligible physical interval (`63–74,92`). Supplying only the timed requests in the API map does not exclude warm-up: lines 82–91 reject the missing warm-up evidence.

CPU reproduction: a warm-up request emitted `[1,2,3,4,5]` on a retained 0.1 s interval and `[6]` on its terminal/cohort-change boundary; the timed request emitted `[7]` on a retained 0.1 s interval and `[8]` terminally. With both full API sequences the reducer returned rc 0, **6 tokens / 0.2 s = 30 tokens/s**, instead of the timed phase's **1 / 0.1 = 10 tokens/s**. With only the timed API sequence it returned rc 2 (`request warm has no API token evidence`).

**Minimal fix/integration requirement:** bind the orchestrator's frozen workload IDs and warm-up/timing phases to a run manifest, and let the reducer retain only fully timed cohorts/intervals while still reconciling all recorded output. An independently validated phase-scoped artifact is another acceptable implementation; none is present in the reviewed components. Preserve full physical-step cohorts and require the declared N1/N4 support once set from pilots. No additional timing cells are needed.

### R3 — P2: final sink failure can leave an apparently valid successful prefix

The recorder now handles a write failure without raising and writes a failure sidecar (`e1_recorder.py:23–48`), but the joiner reads only the JSONL and counters inside surviving events (`e1_join.py:22,28–30`). It neither checks the failure sidecar nor requires a successfully written end/seal. If the final write fails, no later JSONL event carries the failure counter.

CPU reproduction began with a valid two-step log and complete API `[10,11]`, then forced `open(events_path)` to fail on the recorder's next `forward_entry`, while allowing its per-run failure sidecar to be written. No exception escaped; `_CUR['sink_failed']` was true and the sidecar existed. Reducing the surviving log nevertheless returned **rc 0, 1 token / 0.1 s = 10 tokens/s**. This violates the freeze's rule that instrumentation loss invalidates the cell.

**Minimal fix:** require a run-bound successful close/status record after the requests finish, carrying final sequence/count/error totals and completion status, and consume the run's failure sidecar (or a driver verdict that verifiably includes it). A missing or failed close must invalidate the cell. This also supplies an explicit phase/end boundary instead of treating any well-formed log prefix as completed. Keep the now-correct non-throwing failure path. Add a final-write-failure control with a surviving valid prefix.

Remaining route qualification/N1/N4 pilot work stays pending as declared. This review stops at these reproduced token/event-lifecycle gaps and does not reopen closed arithmetic or add graph/cache/stochastic requirements.

At handoff, a hash-only freshness check showed new in-progress recorder `fcf56f9ad605096423d4f64f5d363fc579d661bcdb69b432023db3f5deafea85`, joiner `5bdaf226bcc8a4d5c6761ad9caa04e3d706b33ccbd19491fd5eb4c617a3e082d` and test `059dcca99d1869ebf797638b16e520ab946925a04732c8747fb39eed69683ef3`. Those new implementations were **not** re-reviewed; R1–R3 above apply to the explicitly frozen 07:49:42Z component hashes, not an assertion that later repairs remain defective.

## Settled v5 recheck — 2026-09-22

**R1–R3 are closed at the pinned component hashes below.** Independent CPU reproduction used the actual shimmed emitted `_fr13_sfwd_begin` and `_bookkeeping_sync` function bodies with fake tensor/timer inputs, not a rewritten approximation of their callback ordering. No inference, GPU, source edits or tmux interaction. The bounded assertions passed; no new material defect was found in these reviewed component surfaces.

| Reviewed file | SHA256 |
|---|---|
| `e1_recorder.py` | `1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0` |
| `e1_event_recorder_shim.py` | `06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df` |
| `e1_join.py` | `cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766` |
| Owner control file (read for identity, independent probes below) | `7a13f3a10fb16aa12b0b2bfb157135a3788dcb33813f1cdb82b23e1f26ba7475` |
| Actual shimmed emitted runner | `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40` |
| `E1_FREEZE.md` at this read | `00af8bf7876d36102871c1721828e4f85337fd77dc16b61762242814df5de2eb` |

### Independent results

- **Actual prefill/mixed callbacks + phase selection:** two request streams (one warm-up, one timed), each containing prefill → pure → pure → mixed → pure → pure. All 12 emitted token IDs reconcile with complete API sequences; four non-pure output records are retained in the ledger; only the two eligible timed pure intervals contribute to the result. Exit 0, numerator **2 tokens**, denominator **0.2 s**, primary rate **10 tokens/s**. Warm-up contributes six ledger/API tokens and zero retained rate support. This closes R1 and R2 (`e1_recorder.py:106–124`; `e1_join.py:64–94,101–107,118–141`).
- **Negative token/phase controls:** changing the timed request's prefill token to a wrong equal-count ID returns exit 2; leaving the recorded warm-up request out of the manifest returns exit 2. Thus untimed output remains reconciled rather than silently omitted.
- **Final-write failure:** an otherwise valid unsealed log returns exit 2. Forcing the recorder's final `run_close` write to fail does not throw into inference; the sidecar makes reduction fail with exit 2. Removing that sidecar still returns exit 2 because the successful close seal is absent. This closes R3 (`e1_recorder.py:55–75`; `e1_join.py:38–47,63`).
- **Slow valid interval retained:** extending one valid timed same-cohort interval to **3.0 s**, while shifting later timestamps consistently, leaves both timed intervals in the primary support: **2 / 3.1 = 0.64516129 tokens/s**. The over-cap diagnostic reports one interval/3.0 s; only its separately labeled trimmed diagnostic is 10 tokens/s. No cutoff-based improvement is substituted for the primary rate (`e1_join.py:111–114,132–136`).

### Serving integration remains a separate open gate

At this read, `experiments/e7a/serve_drivers.v9.sh` SHA256 `c35c13252e8aedb28de3d0f6407626bda7fe106d65b8541bc8f84a0c3ab219d6` forwards the E1 shim/record/runtime flags (204–207); `e7a_capture_launch.v5.sh` SHA256 `39dd69d5c11af8784e09fb6fc20a0f01fa57e29307d30d10ebd3d94013e597f0` forwards them into the container (349–351) and applies the shim (434–435). The driver cleanup uses `docker stop -t 30` then owned-container removal (96–97). These source facts do not show that worker shutdown actually runs the recorder's `atexit` seal, nor do they establish the final E1 phase-manifest/reducer orchestration. The recorder's explicit `close()` and reducer seal validation work in the CPU reproduction.

The parent reported a live B1 pilot with an async-scheduling recorder error at 08:17:44Z. That observation was not independently re-read here; it is consistent with the deliberate refusal at recorder line 112 and is **not** a valid E1 pilot. Before timing, the final launch must explicitly select and verify synchronous scheduling, supply the run-bound warm-up/timed manifest and complete API map to the reducer, preserve/check its failure flag, and produce a matching final close seal. A successful CPU close is not an executed serving qualification. Native/B4 pilots, minimum support and route qualification remain pending as separately tracked. No new experimental cells are requested.

<details>
<summary>Exact independent CPU reproduction command</summary>

The initial submission of this command contained a local Python typo (`else256`) and stopped before execution; the corrected command below ran successfully. A second run made the slow-interval fixture shift later timestamps consistently. Its terminal output was `RESULT all independent assertions passed`.

```sh
ssh mark@100.103.10.122 'python3 -' <<'PY'
from pathlib import Path
import ast,builtins,contextlib,copy,hashlib,io,json,os,sys,tempfile,types,time,atexit
b=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e1")
rs=(b/"e1_recorder.py").read_text();js=(b/"e1_join.py").read_text();d=json.loads((b/"dryrun/e1_shim_dryrun_report.json").read_text());src=Path(d["edits"][-1]["file"]).read_text()
assert hashlib.sha256(rs.encode()).hexdigest()=="1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0"
assert hashlib.sha256(js.encode()).hexdigest()=="cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766"
assert hashlib.sha256(src.encode()).hexdigest()=="b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40"
tr=ast.parse(src);fw=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=="_fr13_sfwd_begin");bk=next(n for n in ast.walk(tr) if isinstance(n,ast.FunctionDef) and n.name=="_bookkeeping_sync")
class Timer:
 def __init__(self):self.i=0
 def _prof_tick(self):pass
 def begin(self,num_reqs,cg_mode):i=self.i;self.i+=1;return(None,num_reqs,cg_mode,i)
 def wall_mark(self,**k):pass
 def wall_break(self):pass
 def wall_bookkeeping_error(self):raise AssertionError("unexpected timer error")
class Array:
 def __setitem__(self,k,v):pass
with tempfile.TemporaryDirectory(prefix="e1-v5-independent-") as td:
 td=Path(td);evp=td/"events.jsonl";flag=td/"e1_recorder_FAILED.flag";os.environ.update(E1_RECORD=str(evp),E1_RECORD_FAILFLAG=str(flag),FR13_SFWD_GPU_TIMER="1")
 ns={};exec(compile(rs,"settled_e1_recorder.py","exec"),ns);clock={"t":0};ns["time"]=types.SimpleNamespace(perf_counter=lambda:clock["t"],time=time.time,gmtime=time.gmtime,strftime=time.strftime);rm=types.SimpleNamespace(**ns);t=Timer()
 en={"_e1_rec":lambda:rm,"torch":types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False)),"_fr13_sfwd_is_pure_decode":lambda mx,*a:mx==10,"_fr13_sfwd_timer":lambda:t,"envs":types.SimpleNamespace(VLLM_COMPUTE_NANS_IN_LOGITS=False),"np":types.SimpleNamespace(nonzero=lambda x:([],))}
 mod=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),copy.deepcopy(fw),copy.deepcopy(bk)],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),"actual_emitted_methods","exec"),en)
 api={};phase={"phases":{"warmup":["warm"],"timed":["timed"]}}
 def step(rid,pure,token,stamp):
  clock["t"]=stamp;api.setdefault(rid,[]).append(token)
  batch=types.SimpleNamespace(num_reqs=1,req_ids=[rid],req_id_to_index={rid:0},generators={},num_tokens_no_spec=[0],token_ids_cpu=Array(),is_token_ids=Array())
  me=types.SimpleNamespace(input_batch=batch,discard_request_mask=types.SimpleNamespace(np=[False]),use_async_scheduling=False,max_model_len=4096,requests={rid:types.SimpleNamespace(output_token_ids=[])},_to_list=lambda x:[[token]],_get_prompt_logprobs_dict=lambda *a:{})
  n=10 if pure else 256
  en["_fr13_sfwd_begin"](n,n,1,10,[rid])
  out=en["_bookkeeping_sync"](me,types.SimpleNamespace(num_scheduled_tokens={rid:n}),types.SimpleNamespace(sampled_token_ids=types.SimpleNamespace(shape=(1,1)),logprobs_tensors=None),None,[0]*n,n,None)
  assert out[2]==[[token]]
 for rid,off in [("warm",0),("timed",10)]:
  for pure,token,dt in [(False,500,0),(True,1,.1),(True,2,.2),(False,3,.3),(True,4,.4),(True,5,.5)]:step(rid,pure,token,off+dt)
 def reduce(name,ep=evp,manifest=phase,apis=api):
  ap=td/(name+".api");mp=td/(name+".manifest");op=td/(name+".out");ap.write_text(json.dumps(apis));mp.write_text(json.dumps(manifest));old=sys.argv;sys.argv=["e1_join.py",str(ep),"--api-tokens",str(ap),"--manifest",str(mp),"--expect-reqs","1","--json",str(op)]
  try:
   with contextlib.redirect_stdout(io.StringIO()):
    try:exec(compile(js,"settled_e1_join.py","exec"),{"__name__":"__main__"});rc=0
    except SystemExit as e:rc=e.code
  finally:sys.argv=old
  z=json.loads(op.read_text());print(name,json.dumps({"rc":rc,**{k:z.get(k) for k in ["invalid","n_nonpure_output_records","n_usable","ledger_tokens_all_forwards","sum_emitted_tokens_api_bound_pure_support","sum_wall_s_unique_physical_steps","tokens_per_wall_second","phase_summary","over_cap_diagnostic"]}}));return rc,z
 rc,z=reduce("unsealed_valid_prefix");assert rc==2 and "run_close" in z["invalid"]
 rm.close("CPU independent complete")
 rc,z=reduce("actual_prefill_mixed_and_phase");assert rc==0 and z["n_nonpure_output_records"]==4 and z["ledger_tokens_all_forwards"]==12 and z["sum_emitted_tokens_api_bound_pure_support"]==2 and abs(z["sum_wall_s_unique_physical_steps"]-.2)<1e-8
 wrong=copy.deepcopy(api);wrong["timed"][0]=999
 rc,z=reduce("wrong_prefill_ID",apis=wrong);assert rc==2
 missing={"phases":{"warmup":[],"timed":["timed"]}}
 rc,z=reduce("missing_warmup_manifest_membership",manifest=missing);assert rc==2
 rows=[json.loads(l) for l in evp.read_text().splitlines()];pures=[x for x in rows if x.get("event")=="physical_step" and x["request_ids"]==["timed"]]
 cut_seq=pures[1]["seq"];delta=3.0-(pures[1]["t_start"]-pures[0]["t_start"])
 for e in rows:
  if e.get("seq",0)>=cut_seq or e.get("event")=="run_close":
   for key in ("t","t_start"):
    if key in e:e[key]+=delta
 slow=td/"slow.jsonl";slow.write_text("".join(json.dumps(x)+"\n" for x in rows));rc,z=reduce("slow_primary_retained",ep=slow);assert rc==0 and abs(z["sum_wall_s_unique_physical_steps"]-3.1)<1e-8 and z["over_cap_diagnostic"]["n_intervals_over_cap"]==1 and abs(z["tokens_per_wall_second"]-2/3.1)<1e-8
 badp=td/"failed_tail.jsonl";before=[x for x in rows if x.get("event")!="run_close"];badp.write_text("".join(json.dumps(x)+"\n" for x in before));os.environ["E1_RECORD"]=str(badp);ns["_CUR"].update(closed=False,n=len(before),sink_failed=False,sink_failures=0)
 def badopen(path,*a,**kw):
  if str(path)==str(badp):raise OSError("injected final close failure")
  return builtins.open(path,*a,**kw)
 ns["open"]=badopen;rm.close("injected failing final close");assert ns["_CUR"]["sink_failed"] and flag.exists();rc,z=reduce("final_sink_failure",ep=badp);assert rc==2
 flag.unlink();rc,z=reduce("final_sink_failure_sidecar_unavailable",ep=badp);assert rc==2 and "run_close" in z["invalid"]
 print("RESULT","all independent assertions passed")
 atexit.unregister(ns["close"])
 for k in ["E1_RECORD","E1_RECORD_FAILFLAG","FR13_SFWD_GPU_TIMER"]:os.environ.pop(k,None)
PY
```

</details>

## Pre-freeze document consistency review — 2026-09-22

Bounded read-only review of `E1_FREEZE.md` SHA256 `00af8bf7876d36102871c1721828e4f85337fd77dc16b61762242814df5de2eb` and `E1_DESIGN.md` SHA256 `5bdffb3e5f97113afa046056a58c6ecc614151adcdc9d2222d620ef3bc84f707`. No new tests or experiment axes requested. The selected arms, balanced order, explicit B1/B4, shape-matched warm-up and slow-interval primary policy are internally workable. The three-block bootstrap remains the explicitly limited design already accepted; this review does not demand more blocks.

**Resolve these source contradictions before declaring the freeze:**

1. **Pilot reuse/threshold chronology (`E1_FREEZE.md:30–32,37`).** The current text requires separate native pilot boots, conflicting with the parent's authorized first-planned-native-boot preflight. Use this replacement policy: “Before the first retained confirmation interval, freeze the source/configuration, workload/phase assignment, N1/N4, 0.10 practical precision target and reducer. The already required standalone tree qualification/instrument pilots remain pilot-only. A native route's first planned confirmation boot may begin with a prespecified untimed qualification segment. Its outputs are recorded and reconciled but are excluded from the confirmation result; they may determine only pass/fail against the frozen criteria, not tune thresholds or choose a favorable segment. After successful preflight, apply the same declared warm-up and phase boundary as later boots, then collect that boot's confirmation cell. A failed preflight makes the cell ineligible; retain its artifacts.” This saves the four extra native boots without reclassifying pilot timing as confirmation. Remove the claim that the fixed 0.10 target is calibrated by native pilots' **between-boot spread** unless multiple independent matched pilot boots actually provide that evidence; a single boot per arm/batch cannot estimate it. The target can simply be the practical target chosen in advance. N1/N4 must come from eligible pre-confirmation evidence or an explicit advance decision, never later native preflights after confirmation has started.

2. **First-step support and TPOT label (`E1_FREEZE.md:27`).** Remove “first” from “first/terminal steps”: validated v5 retains the first pure step whenever it has a valid successor. Prefill output still stays in the complete token ledger without entering pure-wall throughput. Replace the undefined B1 “TPOT” promise with its exact intended quantity: if only retained step walls are available, report per-request `sum(retained wall)/sum(API-bound emitted tokens on those intervals)` as **retained decode seconds/token**; reserve full-request TPOT for a separately defined complete request interval. This is a label/definition fix, not a new measurement requirement.

3. **Source/route freeze (`E1_FREEZE.md:18,37`).** Explicitly require `--no-async-scheduling` and post-boot confirmation of synchronous output, plus the finalized tree attention-KV remap setting and actual loaded engagement. Replace “exactly as in the qualified E7b configuration” with “the finalized E2-qualified configuration recorded in the qualification manifest.” The current E7b description is still conditional and cannot silently serve as the release gate. No graph/cache/stochastic routes need adding.

4. **Retire superseded active instructions (`E1_DESIGN.md:6–10,18,24–27`).** These still call for a tree-committer hook, SAMPLER-global row labels, accepted-count fields, and a recorder “not built,” and require all three streams even though components are optional. Mark the old anchor-discovery section historical/superseded and state the current binding contract: every-forward `seq` + actual model-runner output IDs for complete API reconciliation; pure `physical_step_id` for unique retained wall; request/phase manifest and close seal; optional matched component support. This prevents the document from reintroducing the defects closed by v5. At line 13, label the existing layer-62 evidence as conditional surface evidence, not full qualification of the tree/B1 route.

5. **Incomplete cells/extra blocks (`E1_FREEZE.md:30,33`).** “Insufficient support, not timed” is inaccurate if the support count was obtained from an executed confirmation cell; say “executed, insufficient support, ineligible for the planned comparison,” preserving raw data. State that a three-paired-block contrast requires all three valid matched block observations; do not quietly substitute a missing cell, use unpaired data or treat two blocks as the planned three. Keep the initial 18-cell result as the frozen primary report. Any later replication needs its own declared whole-block count/stopping rule before collection and a clearly labeled follow-up or sequential analysis; it must not replace an unfavorable/imprecise initial result. This clarifies reporting and does not authorize or require more cells now.

No additional material design issue was found within the fixed scope. The optional native preflight reuse is compatible with the design only under the frozen-policy/no-retrospective-tuning chronology above.
