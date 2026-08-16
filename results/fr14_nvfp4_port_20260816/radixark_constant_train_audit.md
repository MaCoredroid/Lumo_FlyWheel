# FR14 arm B constant-train audit — 2026-08-16

Mechanical record for the model-bound constant train that re-points the fixed32
serving stack from `/models/qwen3.8-27b-nvfp4` (`qwen3.8-27b-nvfp4`, **arm A**,
unsloth, conservative bytes) to `/models/qwen3.8-27b-nvfp4-radixark`
(`qwen3.8-27b-nvfp4-radixark`, **arm B**, RadixArk, aggressive bytes).

This is the FR14 **bytes ablation**: same stack, same workload, aggressive vs
conservative bytes. Arm A's train is recorded in `constant_train_audit.md` in
this directory; this file is its sibling and follows the same structure so the
two are diffable.

Commits: `fa2705196` (constant train) · `e5e2a6137` (lm_head patch → production
boot flow) · `80ba13b9e` (DVK Phase-1 dequant + fp8-lever guard evidence).
98 files, +2418 / −544 against `61e6b98a1`.

---

## Decisions

**Served-model-name: `qwen3.8-27b-nvfp4-radixark`** (suffix discipline
inherited from arm A, and now load-bearing in both directions). Arm A is
`qwen3.8-27b-nvfp4`, the FP8-3.8 baseline is `qwen3.8-27b-fp8`, and all three
are structurally identical serves of the same base model. Without the suffix a
mis-pointed serve would satisfy every Prometheus label check, trace `model`
field and QC comparison meant for the other arm — exactly the decomposition the
ablation exists to protect. With it, a mis-pointed serve 404s on request one.
Applied consistently across the serve line, the contract argv, every Prometheus
label builder, the chat `model` field and the agent tag
(`qwen3.8-27b-nvfp4-radixark::qwen-code-0.19.4::q38-a`).

**Hard swap, not a switchable arm — evaluated and rejected.** The directive
asked for `FR14_MODEL_ARM=conservative|radixark` *if it could be done
fail-closed and simply*. It cannot:

* ~50 timing instruments (`fr13_run_b1_*.sh`, `fr13_run_b4_*.sh`, the floor
  timer sequence, the live gates) carry the floor as a **shell literal** —
  `export FR13_MANDATORY_WEIGHT_BYTES=25210209416`, and several gate scripts
  *assert* on it (`[[ "$FR13_WEIGHT_FLOOR_MS" == "93.152286652" ]]`).
* The contract's model block is a single byte-pinned 26-row table with no
  partial-swap mode; a switchable contract needs two complete manifests plus a
  selector that cannot disagree with the on-disk config.
* A PARTIAL switch — launcher + contract + ledger only — is the worst outcome
  available: it would export one arm's floor against the other arm's weights,
  silently, and every ratio downstream would be wrong by ~10 ms.

So: **hard swap**, exactly as the FR13→FR14 train did. **Re-serving arm A is a
revert of `fa2705196`**, not an environment change:

```
git revert --no-commit fa2705196   # then re-run the offline verification below
```

Arm A's checkpoint, its floor ledger (`floor_ledger.json`) and its audit are
all still on disk and in the tree, so the revert is a pure constant restore.

**Floor**: re-derived by SUMMING real safetensors tensor spans across the three
shards. See `floor_ledger_radixark.json` (live) beside `floor_ledger.json`
(arm A, retained), and `scripts/fr13_hardware_floor_ledger.py
--derive-from-checkpoint`, which reproduces the arithmetic on demand and exits
2 on drift. The re-derivation was byte-cross-checked against the bring-up
agent's banked `radixark_floor_derivation.md`: identical.

| term | arm A (3.8-nvfp4) | arm B (3.8-nvfp4-radixark) | why |
|---|---:|---:|---|
| target_model_bytes | 17,831,788,928 | **16,892,610,688** | NVFP4 across the whole MLP stack, not MLPs-minus-the-last-8 |
| mtp_forward_bytes_per_pass | 849,398,784 | 849,398,784 | **identical** — both repacks ship the 15 MTP tensors BF16 |
| FULL_HEAD_BYTES (verifier head) | 2,542,796,800 | **715,161,608** | NVFP4 head as shipped vs a BF16 head arm A was forced into |
| SUBSET_HEAD_BYTES (K64 draft) | 671,088,640 | 671,088,640 | **identical** — Phase 1 dequants the slice to BF16 at boot |
| FIXED32 root_64k bytes | 27,977,022,848 | **25,210,209,416** | |
| FIXED32 floor_ms | 102.479937172 | **92.345089436** | −10.135 ms |
| ONE_SIDED_U95_CAP_MS | 117.8519277478 | **106.1968528514** | 1.15x floor, PROVISIONAL |
| K64/root0 arm | 29,848,731,008 / 109.336011018 | 25,254,282,384 / 92.506528879 | |
| full-vocab arm | 37,335,563,648 / 136.7603064029304 | 25,430,574,256 / 93.15228665201465 | cap 107.12512964981684 |
| FR13_DRAFT_HEAD_FP8 arm | RETIRED (exit 2) | **RETIRED (exit 2), new reason** | the head IS quantised now, but Phase 1 dequants the slice |
| FR13_COMPUTE_MS_PER_ROW | 0.54 | 0.54 | fp8-era MEASURED value, still conservative |

**Read the table, not the headline.** The aggressive *backbone* is worth
0.94 GB; the *head* is worth 1.83 GB. This same checkpoint served with a BF16
head lands at 27,037,844,608 B / 99.040 ms — 3.44 ms under arm A, i.e. nothing.
**The NVFP4 lm_head is the entire arm**, which is why the boot-time loader
patch is a hard, fail-closed requirement of this pin rather than an
optimisation.

Derivation rule (stated so it can be re-run): every tensor in every
`*.safetensors` shard is classified **by name**; `target_model_bytes` = all
`model.language_model.*` except `embed_tokens` (64 decoder layers, 1,840
tensors = 16,892,600,448 B, plus the final norm = 10,240 B); `lm_head*` is the
4-tensor NVFP4 set counted separately; `mtp.*` is the 15-tensor BF16 drafter
counted separately; `model.visual.*` (921,460,192 B) and `embed_tokens`
(2,542,796,800 B) are off the text decode path. Whole-ledger cross-check: the
2,194 spans sum to 21,921,428,072 B, which equals RadixArk's own
`qualification.json:checkpoint.output_indexed_payload_bytes` **exactly**.

---

## The MTP_SHARD assumption (directive item 1) — fixed, and it mattered

Arm A's checkpoint carried the drafter in its own `model_mtp.safetensors`, so
`fr13_hardware_floor_ledger.py` located it by FILENAME:

```python
TARGET_SHARD = "model.safetensors"
MTP_SHARD    = "model_mtp.safetensors"
```

RadixArk has neither file. It ships three shards and puts the 15 `mtp.*`
tensors **inside `model-00003-of-00003.safetensors`**, next to ordinary body
tensors. Left alone, the old rule would not merely have failed — it would have
charged the drafter, a **five-times-per-step** term, into `target_model`, a
**once-per-step** term, understating the floor by 4 × 849,398,784 B = 3.11 ms.

Replaced by a shard scan plus `_classify_tensor()`, which buckets every tensor
in every shard by name. `SHARD_SUFFIX` is the only file-level assumption left.
`verify_pinned_constants` additionally gates:

* `mtp_tensor_count == 15` and `lm_head_tensor_count == 4` — so a head that got
  silently dequantised, or a drafter that got requantised, fails loudly;
* `checkpoint_total_tensor_bytes == 21,921,428,072` and
  `checkpoint_total_tensor_count == 2,194` — a whole-ledger check against the
  publisher's own audit, which catches a truncated or partially-redownloaded
  shard that happens to leave one bucket intact.

`scripts/fr13_fixed32_contract.py:315,407` turned out to be `MODEL_FILES` /
`MODEL_FILE_RECORDS` entries, i.e. regenerated wholesale by the generator, not
logic to fix.

---

## Tokenizer normalisation (directive item 2) — done FIRST, before the regen

RadixArk ships a 1,121-byte `tokenizer_config.json` produced by their conversion
tool. Against official 3.8's 17,928-byte file (the same bytes arm A serves) it
is missing `added_tokens_decoder` (33 special-token records), `chat_template`,
`additional_special_tokens`, `extra_special_tokens` and `add_bos_token` — and,
worse than missing, it sets `pad_token = "<|im_end|>"` where official 3.8 sets
`"<|endoftext|>"`. `<|im_end|>` is the chat format's **stop** token; conflating
pad with stop surfaces as a truncation artefact under batching, not as an error.

Normalised by `radixark_tokenizer_normalize.py` (record:
`radixark_tokenizer_normalization.json`):

* `tokenizer_config.json` replaced with official 3.8's
  (`e5d078b0…` → `b11349aa…`).
* `tokenizer.json`, `vocab.json`, `merges.txt` and `chat_template.jinja` were
  **verified already byte-identical** to official 3.8 and left untouched. No
  token id moves, so `MODEL_VOCAB_JSON_SHA256` (`ce99b4cb…` — the pin that
  carries the K64 DVK block map across a model swap) is unchanged **by
  construction**, not by luck.
* The original is archived **outside** the model dir at
  `/home/mark/shared/models/_fr14_orig_nvfp4_fp8head/tokenizer_config.json.radixark.bak`,
  alongside the arm-A tokenizer provenance. Deliberately not an in-dir sidecar:
  the pinned name set stays at the 26 names the checkpoint actually ships.
* The script **unlinks before writing**, so the hardlinked as-shipped view
  (`qwen3.8-27b-nvfp4-radixark-asshipped/`, used for the SGLang native
  calibration) keeps its own inode — verified still 1,121 B afterwards.

Run before the manifest regen, because the manifest pins every file's sha256.
`--check` re-verifies it.

---

## Contract regen

`scripts/fr13_fixed32_contract.py`'s model block regenerated wholesale by
`scripts/fr14_gen_model_manifest.py` (`--check` PASSED). Result:

- `MODEL_ROOT = /models/qwen3.8-27b-nvfp4-radixark`
- `MODEL_SERVED_NAME = qwen3.8-27b-nvfp4-radixark`
- `MODEL_FILES`: **26 names** — 22 upstream + `.lumo_pinned_revision` +
  `.lumo_radixark_kv_surgery.json` + `config.json.pre_kv_surgery.bak` +
  `hf_quant_config.json.pre_kv_surgery.bak`. Three shards, no `layers-N`, no
  `model_mtp.safetensors`.
- `MODEL_CANONICAL_SHA256 = 7e89afacd7351493508a358b7d83e43f141111736d19142bf89c5698033fe84f`
- `MODEL_TEXT_CONFIG_VOCAB_SIZE = 248_320` (unchanged)
- `MODEL_VOCAB_JSON_SHA256 = ce99b4cb…` (unchanged — same digest FR13 pinned
  for the 3.6 dir)

**No lm_head surgery sidecar, on purpose.** Arm A pinned
`.lumo_lmhead_surgery.json` + its `.bak` as the provenance of a head it had to
dequantise to boot at all. Arm B has no such file, and its **absence is the
evidence** that the 4-bit head is served as shipped.

---

## lm_head patch → production boot flow (directive item 4)

`results/…/fr14_patch_nvfp4_lmhead.py` → `scripts/fr14_patch_nvfp4_lmhead.py`
(`git mv`, so the boot smoke and the launcher exercise ONE file;
`boot_smoke_radixark.sh` now mounts `scripts/` as `/ovl_scripts`).

* Invoked in the launcher's container shell immediately after
  `fr10_phase4_patch_vllm_tree_gdn.py` and before `fr13_patch_fa2_tree_bias.py`,
  so every downstream verifier still sees a fully patched tree. Ordering is
  asserted in `tests/test_fr14_lmhead_patch.py`.
* Pinned in the runtime-manifest closure
  (`fr13_runtime_manifest.FIXED32_HOST_SCRIPT_SOURCE`); manifest regenerated
  and confirmed to contain it.
* `export FR14_REQUIRE_NVFP4_LMHEAD=${FR14_REQUIRE_NVFP4_LMHEAD:-1}` sits next
  to `SERVED_MODEL_PATH` and refuses any value but `0`/`1`. The launcher's env
  sweeper forwards `^(FR[0-9]+_|LUMO_|VLLM_)` into the container, which is how
  the patcher sees it.

**A second script rather than four more anchors in the fr10 patcher**: that
file is 42k lines, its sha is consumed by the M32 / M1-U8 / M4-U8 production
credentials, and its anchors are all tree/GDN concerns — these four are
loader/quantization concerns in four *different* upstream files.

### The fail-closed guard was fail-OPEN, and is now baked

The banked patch emitted, *inside the model file*:

```python
if os.environ.get("FR14_REQUIRE_NVFP4_LMHEAD") == "1" and _fr14_qm != "ModelOptNvFp4LinearMethod":
```

i.e. it re-read the requirement in the process that builds the model. vLLM v1
builds the model in a separate EngineCore/worker whose environment is
**curated** — this launcher's own notes record that only 14 of 66 `FR13_*` vars
reach it, and the repo carries a family of "worker-env-drop-proof" sidecars for
exactly this class. If the var is dropped there, the banner still prints and
the assertion **silently does not fire**: the arm serves a BF16 head against a
floor 6.695 ms of which is then fiction. (The boot smoke's PASS came from the
banner, not from the assertion.)

Now the requirement is read at **patch time**, in the container shell, and baked
into the emitted source as an unconditional `raise`. The environment is off the
runtime path entirely. The sentinel encodes the mode
(`FR14_LMHEAD_QUANT_ROUTE_{REQUIRED,PERMISSIVE}`), so re-running under the other
mode is **refused** rather than silently no-opping and leaving the first mode
live. Routing (G1–G4) is byte-identical, so the banked GPU evidence for "the
NVFP4 head loads and generates" carries unchanged.

---

## DVK Phase-1 dequant-at-slice (directive item 5)

`_fr13_dvk_prepare` walks a quantised head for the first time.

**The slice itself needed no change.** On disk the block scales are
`[out, in/16]`, so a row gather picks consistent rows of weight and scale — but
that is not the tensor the shim walks:
`FlashInferCutlassNvFp4LinearKernel.process_weights_after_loading` replaces
`weight_scale` with `swizzle_blockscale(...)`, which keeps the logical shape
`[248320, 320]` and interleaves rows. That permutation **never crosses a
128-row tile boundary** and is identical in every tile, so the FR13 block map's
128-id granularity — chosen years earlier for fp8 block-scale alignment — is
exactly what makes the swizzled scale sliceable.

**What was built** is the dequant that keeps the five K64 reads at the
671,088,640 B of BF16 the floor pins:

* vLLM's **own** `nvfp4_emulation_utils.dequantize_to_dtype(..., swizzle=True)`
  — the exact call `EmulationNvFp4LinearKernel.apply_weights` makes — so the
  BF16 rows are by construction the numbers the NVFP4 GEMM computes against.
* **Chunked at 8192 rows**: `break_fp4_bytes` expands every packed byte into
  two int64 indices before the lookup, so a whole-head pass peaks at several GB
  of transient int64/float32 on a pool that already holds the 46 GiB KV
  reservation. 8192 is a whole number of 128-row tiles, so each chunk is
  independently de-swizzlable.
* Every quantisation-only attribute **deleted** from the shim (a swizzled scale
  or an `alpha` next to a BF16 weight is how a scale gets applied twice);
  `output_size_per_partition` → K64 and `logical_widths` → `[K64]`, each
  cross-checked against the weight's own row count; `quant_method` replaced
  with a real `UnquantizedEmbeddingMethod`, because the sealed FR13 sub-arms
  assert on the method's class **name**.
* **Fail-closed, no `_fr13_dvk_dead` fallback.** A silent fallback would read
  the full head five times — a different byte profile from the pinned floor.
  Six named `RuntimeError`s. A head with no `weight_scale` (arm A, or the
  FP8-3.8 baseline) skips the block entirely, so it is backward compatible.

Boot-log banner (checkable against the floor without inference):

```
[FR14_DVK_DEQUANT] phase1 nvfp4->bf16 at slice K=65536 packed_in=…
  swizzled_scale_in=… bf16_out=(65536, 5120) bytes=671088640
  logical_widths=[65536] output_size_per_partition=65536
  quant_method=UnquantizedEmbeddingMethod
```

Phase 2 — reading those slices **as** NVFP4 (188,743,680 B each,
22,798,484,616 B / 83.511 ms, −8.834 ms) — is emitted in the ledger's
`projected_scenarios` **labelled `PROJECTION, NOT PINNED`** and is deliberately
unreachable as a live scenario. It needs an FP4 GEMV unit and its own byte gate.

---

## fp8-lever guard (directive item 6) — CONFIRMED by evidence

Arm B declares `quantization_config.quant_method == "modelopt"` — a *different*
non-fp8 value from arm A's `"compressed-tensors"`. `tests/test_fr14_fp8_lever_refusal.py`
(16 tests) extracts the guard text **from the launcher** (so a copy cannot
drift) and runs it against real and synthetic configs:

* all five levers refused under `modelopt`, with the value and the specific
  lever named;
* no levers armed → boots;
* an **fp8** checkpoint still ARMS them (the guard has not become "refuse
  everything");
* absent / empty / non-mapping / null-method `quantization_config` → refused
  rather than assumed fp8 — worth pinning, because the probe swallows every
  exception;
* run against all three real checkpoints on disk (`radixark` → refused,
  `unsloth` → refused, `fp8` → armed).

---

## Offline verification performed

- `python3 -m py_compile` / `bash -n` / `node --check` / `yaml.safe_load` /
  `json.load` over **all 98 files** touched by the train: clean.
- **The injected fragment compiles.** `_fr13_dvk_prepare` lives inside a
  ~5,000-line replacement STRING, so `py_compile` on the patcher proves nothing
  about it; `tests/test_fr14_dvk_dequant_shim.py` extracts the literal and
  compiles it. This is the only thing between a typo and an ~8-minute boot that
  dies at first forward.
- `fr14_gen_model_manifest.py --check`: **PASSED** (26 files, digest match).
- `fr13_hardware_floor_ledger.py --derive-from-checkpoint`: **PASSED** against
  the real three-shard checkpoint; every pinned byte term re-derived.
- `fr13_fixed32_contract.py external-manifest --repo $PWD`: **succeeded end to
  end** against the real model dir, the pinned docker image and the pinned
  baseline FA2 `.so`.
  `overall_canonical_sha256 = 6ce9e47f1bb4c2db98e21b0e355f13937611c0e9a68bfcfb87084fe19e58d0be`.
  (The FA2 `.so` at `output/auto_research/…/_vllm_fa2_C.abi3.so` in **this**
  worktree is 299,183,936 B / `f51e23c5…` — it matches the contract pin, so the
  arm-A-era reconciliation note does not apply here.)
- `fr13_runtime_manifest.py --profile fixed32`: regenerated, contains
  `scripts/fr14_patch_nvfp4_lmhead.py`.
- `radixark_dvk_swizzle_check.py`: **PASS** (unchanged).
- `radixark_dvk_dequant_check.py`: **PASS** at the real `[248320, 320]` /
  `[248320, 2560]` shapes with the real 512-block gather —
  `deswizzle(slice) == slice(deswizzle)`; `chunked(8192) == whole-slice`;
  chunked dequant == whole-slice dequant **bitwise**; `dequant(slice(head)) ==
  slice(dequant(head))` **bitwise**. Both controls behave: a 100-row chunk
  cannot even be reshaped into the tile layout, and a non-128-aligned gather
  does not commute.
- `radixark_tokenizer_normalize.py --check`: PASSED.
- registry loads (`load_registry('model_registry.yaml')`, 8 entries).

### Targeted test runs

`TMPDIR=/home/mark/shared/tmp-scratch PYTHONPATH=$PWD/src` over 34 modules
covering every floor constant, the contract, the launcher wiring, the metric
label parsers, the retired arm, the new lm_head patch, the new DVK shim and the
fp8-lever guard: **861 passed, 2 skipped, 25 failed**.

All 25 failures reproduce at HEAD (verified by stashing the entire change set
and re-running), i.e. **zero regressions**:

| test | count | pre-existing cause |
|---|---:|---|
| `test_fr13_treeconv_zero_tail_credential.py` | 12 | committer boundary-snapshot key set moved ahead of the fixture |
| `test_model_server.py` | 10 | `kv_cache_dtype` allowlist / VRAM-grace fixtures moved ahead of the tests |
| `test_fr13_b1_composed_stack.py::test_combined_sfwd_gate_…` | 1 | fixture `SimpleNamespace` lacks `direct_nodegroup8` |
| `test_fr13_dfwd_k64_fp8_selector.py::test_selector_accepts_exact_b1_b4` | 2 | stale FR13 credential — see below |

Never run (house rule): `tests/test_codex_long_assets.py`, and no full pytest.

### Test fixtures re-placed with the floor (not silently retargeted)

Two synthetic fixtures encode an operating point *relative* to the floor and
cap. Leaving them fixed turns an eligibility assertion into a tautology (or, in
the first case, into an assertion that the cap works), so they moved to the same
RELATIVE position under arm B's numbers, with the reason recorded in-line:

- `tests/test_fr13_b1_composed_stack.py`: wall/u95 111.0/115.0 → **100.0/103.5**
  (115 now exceeds the 106.197 cap), phase breakdown 68/18/9/16 → 61/16/8/15,
  re-summed to the new wall. Ratios held: wall/floor 1.083, u95/floor 1.121.
- `tests/test_fr13_qrow32_split2_timing.py`: 103–106 ms → **93–96 ms** (106.0
  was within 0.2 ms of the cap).

Third generation of the same move: FR13 130/135 → arm A 111/115 → arm B 100/103.5.

---

## Deliberately NOT changed

- **`scripts/fr13_dfwd_k64_fp8_selector.py` (`SOURCE_SHA256 = 0696bfc5…`) and
  `scripts/fr13_run_b1_dfwd_k64_fp8_real_task.sh`** — the fr10 patcher's sha
  moved (`a61b1d73…` → `227c0c2c…`) when the DVK shim landed, but these pins
  were **already stale at HEAD** and belong to the RETIRED `FR13_DRAFT_HEAD_FP8`
  arm, which the launcher and the floor-timer sequence refuse outright. A fresh
  pin would falsely claim the arm had been requalified against a patcher it
  never ran with. Every other consumer recomputes the sha with `sha256sum`, and
  the runtime manifest hashes it dynamically.
- **`scripts/fr13_b4_honest_floor.py` + `tests/test_fr13_b4_honest_floor_artifact.py`**
  — FR13-frozen. Only the docstring's pointer to the live floor was updated to
  name both arms; every measured anchor stays a Qwen3.6-FP8 measurement.
- **`results/fr14_nvfp4_port_20260816/floor_ledger.json`** — arm A's ledger,
  retained as the other half of the ablation. The live binding moved to
  `floor_ledger_radixark.json`, and `test_fr13_hardware_floor_ledger.py` now
  asserts BOTH, plus the 2,766,813,432 B delta between them and the fact that
  1,827,635,192 B of it is the head.
- **`scripts/fr14_leg3_launch_nomiddleware.sh`** — untracked, another agent's
  in-flight work on this worktree. Not swept, not committed.
- **`model_registry.yaml`'s `qwen3.8-27b-nvfp4` entry** — arm A stays bootable
  for A/B, exactly as the 3.6 entry was retained through the FR13→FR14 train.
- **`FIXED32_B4_KV_CACHE_MEMORY_BYTES` (46 GiB)** — KV geometry is identical.
  Re-confirm against a real boot log's reported KV pool.

---

## Must be verified on the next GPU boot

1. `[FR14_LMHEAD_NVFP4]` + `FR14_LMHEAD_QUANT_ROUTE lm_head
   quant_method=ModelOptNvFp4LinearMethod` banners, under the **baked**
   enforcement (the mode this train changed).
2. `[FR14_DVK_DEQUANT] phase1 …` banner with `bytes=671088640`,
   `logical_widths=[65536]`, `output_size_per_partition=65536`,
   `quant_method=UnquantizedEmbeddingMethod` — the new risk, and the first time
   the dequant path runs on a device.
3. PID1 argv equality: `vllm serve /models/qwen3.8-27b-nvfp4-radixark
   --served-model-name qwen3.8-27b-nvfp4-radixark …` vs `expected_pid1_argv`.
4. `kv_cache_dtype=auto` in the engine log and no `fp8_e4m3` anywhere (the KV
   surgery holding under the fixed32 serve line, not just the smoke's).
5. Peak host memory across the chunked dequant at `GPU_UTIL=0.70` with the
   46 GiB KV reservation live.
6. `FR13_COMPUTE_MS_PER_ROW` — re-measure on the first arm-B B1 profile; 0.54 is
   the fp8-era value, retained as a conservative (high) bound.
7. The provisional `ONE_SIDED_U95_CAP_MS = 1.15 x floor` — Mark's open ruling.
8. Prometheus label plumbing end to end: every counter bracket now reads through
   `model_name="qwen3.8-27b-nvfp4-radixark"`.

## Reduction reminder

Same deploy-speed env as arm A, but:

```
FR13_MANDATORY_WEIGHT_BYTES=25210209416
FR13_WEIGHT_FLOOR_MS=92.345089436
```
