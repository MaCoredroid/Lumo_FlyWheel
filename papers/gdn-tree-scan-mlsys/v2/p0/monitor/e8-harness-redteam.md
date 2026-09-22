# E8 single-logits harness — independent qualification review

Date: 2026-09-22. **PASS to run the two untimed qualification boots at the exact manifest below. Timing is not enabled or qualified by this disposition.** No GPU, model inference, serving boot, production source modification, or E1/E7 evidence change was performed by this reviewer.

Stage: `papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits` in `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel`; execution host repository `/home/mark/lumo-paper-v2-20260921`.

## Frozen reviewed identity

| File | SHA-256 |
|---|---|
| `manifest.json` | `45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf` |
| `e8_shim.py` | `b6c69fd47170d072ee1649a2be5ade7ff03214fa44a9caf18a85715c4ef362af` |
| `e8_head_gate.py` | `ab66067f0148a138b9fddea3494e4b76c1525259504d2445b38093c18166cd67` |
| `e8_verify.py` | `d8f4ee02021d568a3d88eaa025c1ad43c7b8dc66b1c976f1fcc890e8a38c4dbf` |
| `e8_qualify.py` | `f0cbbd6097eb77bd2233a42dcac2bf86caef9cff02971ae4cc71cf9c674bb5c0` |
| `e8_launch.v1.sh` | `96376508fa6dad0efe243bf194601949f0eae512c74b74857e7b79d67b9a52a6` |
| `e8_prepare.py` | `9b0d1e7c0ae9210ea2c90654bc9e99a676b0b8e35fc4fd7e001facade3e41842` |
| `test_e8_cpu.py` | `7264e5d25762fc3c79d02626683c1f1f4716caffe02cc727e17648887a20862a` |
| `README.md` | `c7fd080023b04ceba5d9c14ab928804390ced6d0e7c8aacdc42156a4ec219ef8` |

The manifest binds 21 staged files, including six frozen E1 repository dependencies, copied E1 recorder/client/reduction files and prefixes. The shim accepts only actual E1 `eagle.py` hash `aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7`. Independent rendering reproduced all four manifest outputs:

| Arm / instrumentation | Emitted `eagle.py` SHA-256 |
|---|---|
| ON qualification | `3cdf64ba92bd24bdd3d6e8255277186f2fd211f3de54c5cfd4394a558e115358` |
| OFF qualification | `364b767f6003cac1cd021e82249438edef4e0f87ba49a6aa6887506b8c9dc9e5` |
| ON clean | `7a1f20b0897ec488f0a1e168105ce392cac566e79dbd206a1edeee527c0060c8` |
| OFF clean | `959a7e9f46dc239403c8c5ed42e78e3208d80d2f3e762235c570b6ed5e837101` |

The existing frozen native/attention/runner/sampler/GDN module hashes are retained; the new arm switch is confined to the emitted drafter module and its counted head helper. The source-controlled boolean reaches the real legacy double-head branch; it does not pretend that the old baked-on environment flag was already an ablation.

## Material review findings and resolutions

1. **Generated Docker arguments contained literal `+` markers.** Draft `e8_prepare.py` inserted `\n+` in mount/environment continuations. Fixed before freeze. Independently regenerated the replacement text in memory, checked no leading-plus lines and passed `bash -n` on the final launcher.
2. **Import-only processes could overwrite the head receipt.** The draft helper registered an unconditional shared-file atexit writer. Final helper:18–29 claims an exclusive owner file at the first actual head; lines39–52 permit writes/sealing only from that PID. The verifier binds the owner sidecar to the head receipt. The import-without-head control preserves a sentinel receipt. This avoids trusting an unrelated process's empty exit report.
3. **Result receipts did not fully bind the declared qualification workload.** Final verifier:90–103 requires exactly eight distinct frozen-prefix records, both recorded prompt hashes, preflight phase, max32 in the request and record, temperature zero, direct API IDs and explicit request seed. The copied workload itself verifies frozen prompt text before sending requests.
4. **Qualification could spend hours in sequential client timeouts.** Final driver bounds the complete eight-request subprocess to600s; launch to300s, health to1500s and ordinary subprocess calls to180s. The copied E1 client is preserved. Deadline failure stops qualification and invokes owned-container cleanup; no second arm or timing follows.
5. **Independent receipt needed actual configuration checks.** Final verifier:49–75 checks source/recorder env, exact nine-draft Cat10 topology, policy B, runrow initialization, absence of force/argmax substitution, device committer default, absence of fixed32/TAW/depthsync/graph/multistream arms and marker files, image/model/seed/B1/eager/sync/cache-off arguments. The driver additionally constructs a sanitized explicit environment and checks unchanged loaded modules before workload. Recorder/staged bytes are verified before launch and again by final `stage()` validation.

All five findings are closed at the manifest above. No unresolved blocker to the bounded qualification remains.

## Independent CPU checks executed

No serving module was imported. Local source checks used Python `-B`; Torch checks ran on the DGX host with `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`, Torch2.4.1, tiny CPU tensors only. Temporary copies/sinks were removed; staged source and the author's test output were not changed.

- Independently rendered/AST-compiled all four variants; each has four primary call sites, one legacy wrapper and one proposal-finalization site. Wrong source bytes, invalid arm and nonboolean qualification are refused.
- Independently reran **27/27 supplied CPU controls** in a temporary stage copy. These include report-census corruption, missing/failed/unsealed/ownerless records, source dependency refusal, import-only receipt protection and launcher/source checks.
- On final head helper `ab66067f…`, independently completed ON and OFF five-head proposals: both qualification modes have primary5 + legacy5, root1 + loop4, qualified proposal1, and a valid owner-sealed receipt.
- Independently refused **nine semantic negatives**: hidden-input write; parameter write under inference mode; RNG mutation; changed same-input full logits; corrupted packed candidate; unpaired legacy call; changed hidden bytes between heads; incomplete depth-five proposal; unsupported local-argmax route.
- Exercised clean helper call sequences: ON records primary5/legacy0/proposal1; OFF records primary5/legacy5/proposal1, both sealed. This verifies the counter machinery and agrees with the emitted branch structure; it is **not** observation of a clean GPU timing cell.
- The default qualification command returned only `PLAN`, `timing_enabled:false`, and created no run directory. Manifest hashes remained unchanged throughout the final pass.

A positive CPU gate does not establish FP8 GPU head equality. That is precisely the purpose of the next untimed qualification. A false/missing/failed GPU result must remain a failure; no tolerance adjustment follows from it.

## What actual qualification must prove

Both arms evaluate both heads on the same hidden input. Full raw logit bytes, argmax/top-two order and the complete packed nine-draft array must match. The hidden tensor, CPU/CUDA RNG state, model tensor identity/version census and dispatcher-checked read-only storage contract must remain unchanged around each head. These are local same-input checks, not equality between independent boots or all model states.

The copied E1 preflight path uses all eight frozen prefixes sequentially at B1, max32, with EOS unchanged. `e8_qualify.py` maps all engine request IDs into the joiner's untimed exclusion, never into timed support. Its join must return3 only because support is untimed: full API/recorder reconciliation still has to pass, including the final closing seal. `e8_verify.py:104–110` additionally requires all eight API bindings, zero usable timed intervals and pure-decode exposure for every request. An incomplete or failed ledger cannot supply a successful prefix of evidence.

Exact continuation code identity plus equal same-input ordered candidates supports this head-local ablation's state-preservation obligation. It does not newly establish live all-layer convolution/KV byte equality, universal numerical equivalence or full-model sequential equivalence. Existing E2's finite route qualification and E1's adverse/divergent observations remain unchanged.

## Command, resources and stopping rule

After the parent finishes its model-weight preflight and verifies the synced stage, execute on the idle DGX. This command runs **qualification only**:

```bash
E8_REVIEW_RUN="/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-$(date -u +%Y%m%dT%H%M%SZ)-e8-single-logits"
python3 -B /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/e8_qualify.py \
  --repo /home/mark/lumo-paper-v2-20260921 \
  --run "$E8_REVIEW_RUN" \
  --port 9968 --execute
```

Omit `--execute` for the verified plan-only command. Do not rerun `e8_prepare.py` on an attempted run snapshot. Existing output roots/container names are refused rather than overwritten.

Resources: one exclusive GB10/DGX Spark serving container at a time, pinned Qwen3.6-27B-FP8 checkpoint/image, GPU utilization0.6, max model length16384, actual B1; two serial boots and16 total qualification requests, at most512 requested output tokens. The read-only model mount and host-memory recovery check are retained. Parent is separately rehashing the original P0 weight inventory into `p0/monitor/e8-weight-preflight-20260922.json`; this review does not assert that unfinished operational check has passed.

**Realistic expected time:**15–25 minutes for both qualification boots, allowing diagnostic overhead; not guaranteed. The prior E1 B1 tree boot logged health only after320s, so boot/JIT time dominates this small token workload. The fixed deadlines are failure bounds, not expected durations. The first invariant, source, API, seal or deadline failure must produce `FAILED.json`, preserve evidence and prevent the OFF boot if ON failed. No timing, correctness retry, arm replacement or relaxed threshold is authorized by this driver.

## Separate timing gate

Only a successful two-arm qualification receipt at the same source manifest can be considered for the later six-cell B1 timing implementation. The current launcher rejects `E8_QUALIFY=0`; no timing driver was supplied or reviewed in this stage.

That later driver must bind the reviewed clean emitted hashes, disable all dual-head checks/captures, and enforce the actual terminal census: ON `primary=5P, legacy=0`; OFF `primary=5P, legacy=5P`, for completed proposals `P`, with no failures or incomplete head work. It must also retain the E1 API/physical-step seal and exact per-prefix support. A counter-only CPU simulation is not sufficient to credit head removal in an actual timing cell.

The planned finite design is six new B1 boots, OFF/ON, ON/OFF, OFF/ON; two32-token warmups then eight max128-token requests per cell; no repeats/replacements. The qualification manifest’s disabled, provisional timing plan names ratio of arm-mean cell rates minus1. After this qualification freeze, the parent selected mean of the three paired relative differences as the later timing primary, with ratio of means diagnostic. That prospective change must be dated and frozen in the new sibling timing stage before any timing data; it does not rewrite or invalidate the qualification manifest. The timing analysis/driver remains outside this disposition, and three blocks must remain explicitly coarse. Expected additional elapsed time is roughly40–55 minutes, conditional on successful qualification and the later reviewed driver. No rerun of native baselines, B4 extension, fixed32/TAW port or FA2 campaign is necessary for this bounded component question.

## Post-review infrastructure failure — preserved chronology

The first attempted qualification root, `experiments/out-20260922T2119Z-e8-single-logits`, stopped before any container/model boot: generated launcher line11 executes the detached comment suffix `, NOT HARDCODED. This read`. Original `FAILED.json` SHA is `60a01a22ed29afa94e0100f43bdd417746df1fa80905145eafeac10bab235096`; launcher stdout SHA is `9cdc27b64fd8053d3cfa4d14e9c73ab96da6ac6e414cdd9e973a51d6f35a12fa`. The frozen original manifest/source and failed attempt remain untouched.

The approval above missed this infrastructure defect: syntax checking and source/AST checks did not exercise shell execution. A new sibling qualification-v2 is being reviewed separately with actual launcher execution against strict non-GPU stubs, including its constructed inner command. No scientific qualification or timing data were obtained; no head/API/source threshold is being weakened. The original PASS must not be interpreted as a successful GPU qualification, and the original launcher must not be relaunched. The parent authorized one replacement infrastructure attempt only after the sibling review.

The sibling infrastructure correction subsequently passed independent Linux outer/inner execution regression and27/27 CPU controls at manifest `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`. See `e8-qualification-v2-redteam.md` for the qualification-only replacement-attempt disposition; no GPU qualification result is implied.
