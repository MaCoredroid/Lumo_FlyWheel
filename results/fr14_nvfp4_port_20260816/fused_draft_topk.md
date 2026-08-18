# FR14 lever 1 — fused draft top-k (K0 full-vocabulary drafter head)

**Greenlit** by Mark as "fused draft top-k — build now", against the banked surface
*"K0 materializes 248320-row logits ×5 passes ≈ +4.5 ms measured vs +0.8 byte floor"*
(`REDTEAM_20260816.md` pass 31; derivation `host_dfwd_characterization.md` §5.1).

**Status: BUILT, GATED, MEASURED. Default OFF. Not serve-promoted.**

`analysis_only=true`, `acceptance_valid=false`, `step_envelope_claim=false`.
No TPS, floor or acceptance claim is made or moved by anything below. The lever is
acceptance-invariant *by construction* — see §3 — so per instrument doctrine its
promotion is judged on `step_wall_ms` and `s_per_fwd_gpu` (plus the `dfwd` span via
`FR13_DFWD_SPLIT`), **never on TPS**.

---

## 0. Headline

Three things, in the order they matter.

1. **The lever is real and it is byte-exact.** One fused CUDA launch replaces the
   `argmax` + `torch.topk` pair on every one of the five drafter head reads, produces
   the identical token ids in the identical order across **6 840 gated configurations
   including 320 planted exact-tie plateaus**, and is **0.3078 ms/step**
   faster.

2. **The briefed mechanism is wrong, and the brief's own number does not survive.**
   The surface was flagged as a *materialisation* cost. Materialisation is
   ~2.5 MB/step of logits writes ≈ 0.01 ms at 273 GB/s, and the measured cost of
   copying the row is 7.552 µs. The whole selection stage — write plus
   every re-read plus both reductions — is **0.3483 ms/step**. The
   ≈ +4.5 ms that flagged this surface therefore **cannot** be the selection, and the
   +4.5 was always a confounded (checkpoint ⊗ flag) one-pair observation that its own
   author banked as *"not a claim"*. §6.2 bounds where it actually is.

3. **A live defect fell out of the gate.** ATen's `argmax` and ATen's `topk` do **not
   agree with each other** on ties, and the deployed drafter uses one for the spine
   slot and the other for the sibling slots of the *same* node. On
   **10/48 (21 %) of realistic draft-logit rows** the rank-1 sibling is
   therefore a **duplicate of the spine token** — a wasted verify node, every time.
   This lever reproduces the defect exactly (that is what byte-exact means); fixing it
   is a separate, acceptance-affecting, Tier-B decision. §9.

---

## 1. What the deployed path actually does

Served arm: `hydra27_fixed32`, B1, radixark NVFP4, **K0** — i.e.
`FR13_DRAFT_VOCAB_ROOT=0`, `FR13_DRAFT_VOCAB_K=0`, the 64K subset head retired
(`host_dfwd_characterization.md` §5). The drafter's five head reads are 1 root +
4 graph-captured loop passes (`drafter.mtp_forward_calls = 4`, single-valued over
20 579 consecutive decode steps).

`_FR13_FIXED32_CHOICES` gives per-depth widths **(3, 3, 3, 3, 3)** at head depths 0..4
(= `fr13_fixed32_topology.SAMPLER_MAX_FANOUT`), so at each of the five reads the
drafter runs, on the `[B, 248320]` bf16 row that `compute_logits` just wrote:

```python
draft_token_ids       = logits.argmax(dim=-1)                 # rank-0 slot (spine)
_fr10_wide_topk[pos]  = torch.topk(logits, 3, dim=-1).indices  # rank-1/2 slots (leaves)
```

`_fr10_leaf_steps` is empty under `_fr10_is_wide` and `_fr13_dedup_slack == 0` for
fixed32, so those two calls are the *entire* consumption of the logits row. Nothing
else reads it.

Both are latency-bound multi-kernel affairs at this width. `torch.topk` on a
248 320-element slice takes ATen's **multi-block** radix-select path (`mbtopk`) —
fill, radix histogram passes, blockwise within-k counts, blockwise kth counts, a cub
scan, `gatherTopK`, then a sort of the k survivors. `gatherTopK` is the kernel the
FR14 attribution table already lists at 5×/step.

lm_head is `ModelOptNvFp4LinearMethod` via `FlashInferCutlassNvFp4LinearKernel`,
shared target↔draft (`radixark_boot_probe_20260817T012523Z.log:62,77,86`), so the
logits row is **bf16**, contiguous, `[B, 248320]`.

## 2. What was built

| artifact | path |
|---|---|
| kernel | `csrc/fr14_dfwd_full_topk.cu` |
| pinned builder | `scripts/fr14_build_dfwd_full_topk.py` |
| gate + microbench | `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe.py` |
| gate + bench evidence | `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe_result.json` |
| build attestation | `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_build_attestation.json` |
| build reproducibility | `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_build_reproducibility.json` |
| integration | `scripts/fr10_phase4_patch_vllm_tree_gdn.py`, env `FR14_FUSED_DRAFT_TOPK` (default `0`) |
| torch-free pins | `tests/test_fr14_fused_draft_topk.py` |

One launch of `torch.ops.fr14_fused_draft_topk.select_out` emits **both** the spine
int64 `[B]` and the width-3 int64 `[B,3]`, reading the row exactly once. Under the
drafter graph it writes **straight into** `_dg["spine"][t]` and `_dg["wide"][t,:,:3]`,
so the two static-buffer `copy_` calls the stock path needs are gone as well.

Selection is defined by a **total order on (value, index) packed into one uint64 key**:

```
key = (order_preserving_u32(value) << 32) | (0xFFFFFFFF - index)
```

Indices are unique, so keys are unique, so "the three largest keys" is a function of
the row alone — independent of how many CTAs ran, in what order they finished, or in
what order two candidate lists merged. **Determinism here is structural, not
empirical**: there is no reduction-order freedom left to be non-deterministic about.
That is why the gate sweeps `blocks_per_row ∈ {1, 8, 32, 64, 121}` and all five must
agree with torch *and* with each other.

## 3. Why the lever is acceptance-invariant

Every token the drafter proposes is bit-identical, so the tree handed to the verifier
is bit-identical, so acceptance is unchanged **by construction, not by measurement**.
That is what makes it Tier-A, and it is also why TPS is the wrong instrument: the
acceptance channel is provably closed, so any TPS delta observed would be trajectory
noise. `step_wall_ms` / `s_per_fwd_gpu` / the `dfwd` span are the instruments.

## 4. The tie-break: the part that decided the lever

This is where the build nearly stopped, and it is the transferable finding.

**bf16 ties at the top of a 248 320-wide row are the common case, not an edge case.**
bf16 carries 8 mantissa bits; at logit magnitude ~32 the spacing is 0.25, and a
248 320-sample draw puts several values inside one ulp of the maximum.

A first gate run against the naive assumption ("descending value, ascending index",
the rule the campaign's existing K64 kernel documents) failed on **295 of 460
configurations**, including plain random rows. Rather than weaken the gate, the
question was answered by measurement: a 411-row adversarial sweep at the real geometry
(random at three scales, coarse-quantised at four levels, and planted plateaus over
n ∈ {2,3,4,5,8,17,64,1000} × stride ∈ {1,2,7,256,257,1024,4096,30011} × five offsets)
tested three candidate rules. The result was unambiguous:

| rule | matches `torch.topk` |
|---|---:|
| **SET = k largest by (value desc, index ASC); ORDER = (value desc, index DESC)** | **411 / 411** |
| SET as above; ORDER = (value desc, index ASC) | 62 / 411 |
| SET = k largest by (value desc, index DESC); ORDER = (value desc, index DESC) | 99 / 411 |

`torch.topk` repeated on the same row disagreed with itself **0 times in 411 cases**,
so the deployed selection *is* a function of the logits and exact parity is definable.
(Had it not been, the pre-registered rule was STOP — the selection would not have been
reproducible even by torch.)

Meanwhile ATen's `argmax` is **(value desc, index ASC)** — i.e. `argmax` is rank 0 of
the *set*, which is **not** in general `topk(...).indices[:, 0]`.

The kernel therefore emits both semantics from one ladder: `spine_output` is taken
before `fr14_emit_order` (argmax semantics), and `fr14_emit_order` then reverses each
equal-value run to recover ATen's emission order. **Reproducing the disagreement is
the point.** Byte-exactness means matching what ships, including where what ships is
internally inconsistent.

## 5. The gate — PASS, 0 mismatches

`fr14_fused_draft_topk_probe.py`, run in the pinned image
(`vllm/vllm-openai@sha256:3dbe092e…c38e776`, torch 2.11.0+cu130) on GB10 / SM 12.1.

Pre-registered verdict rule, written into the probe's docstring before the data
existed:

* 0 raw-byte mismatches over every case, seed, row count and CTA count → Tier-A parity.
* any mismatch on a **non-tie** case → the kernel is wrong.
* any mismatch confined to **tie** cases → tie-break parity NOT established; report
  **STOP**, not "approximately equal".
* torch disagreeing with torch across repeats → exact parity undefinable → **STOP**.

Result:

| | |
|---|---:|
| cases evaluated (rows 1..4 × case × seed) | **1 368** |
| configurations (× 5 CTA counts) | **6 840** |
| raw-byte mismatches, total | **0** |
| … on tie cases | **0** |
| … on plain cases | **0** |
| torch self-disagreements | **0** |
| powered negative control fired in every case | **yes** |

Comparison is on **bytes** (`.view(torch.uint8)`), not values, so NaN or signed zero
cannot launder a mismatch past it. Case coverage: 320 planted tie plateaus (the full
n × stride × offset cross, strides chosen to straddle the 8-wide vector, the 256-thread
CTA and the CTA grid stride; offsets at index 0, mid-row and V−1), plus all-equal,
all −inf, coarse-quantised, signed zeros, narrow-scale random, and realistic random.
NaN is carried as a **reported diagnostic, never gated** (a NaN draft logit means the
model is broken, not the kernel).

Anti-vacuity: every case also compares a deliberately corrupted reference that MUST
mismatch. It did, in every case; the comparison is looking at something.

## 6. The measurement — and the honest re-pricing

Pinned image `vllm/vllm-openai@sha256:3dbe092e...c38e776`, torch 2.11.0+cu130,
GB10 / SM 12.1 / 48 SMs / 24 MiB L2. Candidate `.so`
`8f7a99e78c0898a4…` (the byte-reproducible build, §7.1). CUDA events,
25 warmup + 200 reps, p50. Synthetic logits at the real geometry, GPU idle.
Two earlier runs taken under a concurrent lane-5A serve agree within 2 %, so the
numbers are not contention artefacts.

| per head read, B=1 | p50 µs | ×5 head reads = ms/step |
|---|---:|---:|
| `logits.argmax(dim=-1)` | 10.96 | 0.0548 |
| `torch.topk(logits, 3)` | 63.52 | 0.3176 |
| **deployed pair (OLD)** | **69.664** | **0.3483** |
| fused, 1 CTA | 39.008 | 0.1950 |
| fused, 8 CTAs | 12.48 | 0.0624 |
| **fused, 32 CTAs (best, NEW)** | **8.096** | **0.0405** |
| fused, 64 CTAs | 8.224 | 0.0411 |
| fused, 121 CTAs | 8.256 | 0.0413 |
| *reference: bare 496 640 B row copy* | *7.552* | *0.0378* |

**MEASURED SAVING, B=1: 61.568 µs per head read = 0.3078 ms/step (8.6× on the stage).**
At B=4 it is 69.344 µs per head read = **0.3467 ms/step**.

The fused kernel lands at **8.096 µs against 7.552 µs for a bare copy of the same
496 640 bytes** — within 7 % of the small-kernel floor on this box, so there is
essentially nothing left in this stage to take. The single-CTA variant
(39.008 µs) stays in the sweep as the proof that the multi-CTA ticket path is
what buys the win, not the fusion alone.

Honest scale check: **0.3078 ms is 0.15 % of the 210.7 ms step and 0.6 % of the
49–53 ms `dfwd` span.** It is real, it is free of acceptance risk, and it is
roughly one fifteenth of the briefed number.

### CUDA-graph replay gate

24 replays × 4 captured selects, fresh logits every replay, compared against the
eager ATen reference: **0 mismatching replays**, and the scratch ticket was back at
zero afterwards (`ticket_self_cleaned: true`). The graph path is the one the
serve will actually run.

### 6.1 Materialisation is not the cost

The surface was named "draft-logits **materialization**". Priced directly:

* the logits row is 248 320 × 2 B = **496 640 B**; five per step = **2.48 MB/step**;
* at the campaign's pinned 273 GB/s that is **0.009 ms/step** of write traffic;
* measured, copying the row costs **7.552 µs** — i.e. even *perfectly*
  eliminating materialisation, which would require fusing the selection into the
  CUTLASS NVFP4 GEMM epilogue itself, buys at most 0.038 ms/step.

The cost is **kernel latency, not bytes**: `torch.topk`'s multi-block radix select is a
chain of small kernels on a 0.5 MB slice, and it is ~8.4× the cost
of reading the slice once. That is exactly what the fused kernel removes, and it is why
this lever does **not** need to avoid materialising the row to collect nearly all of
the available win. A chunked-GEMM / running-merge variant that never materialises the
full row would additionally have to prove that N-chunking the FlashInfer-CUTLASS FP4
GEMM is bitwise-invariant (it is autotuned per problem shape, so that is a real risk),
would add C−1 extra GEMM launches, and would buy at most the copy line above.
**Not built. Priced and parked, with the number that prices it.**

### 6.2 So where is the +4.5 ms?

**Not here — and this measurement bounds it.** The *entire* K0 selection stage
costs **0.3483 ms/step**. Even if the K64 arm's selection over a 65 536-wide row
were free, at most **0.3483 ms** of the ≈ 4.5 ms K0-vs-K64 `dfwd` difference can be
selection; **≥ 4.1 ms/step lies elsewhere**. Combined with the 0.807 ms byte-floor
difference the two arms' ledgers already declare, the residual is ≈ 3.4 ms/step
that neither bytes nor selection explains.

The remaining candidate is the projection itself: under K64 the drafter head was
a dedicated BF16 GEMV over a 65 536-row slice (`_fr13_dvk_logits`), while under
K0 it is the shared `lm_head` through vLLM's `ModelOptNvFp4LinearMethod` →
`FlashInferCutlassNvFp4LinearKernel` at M = 1, N = 248 320, K = 5120. Those are
different kernels, not different widths of one kernel.

**That measurement belongs to lane 5A**, which took it on the REAL RadixArk
4-tensor head through vLLM's own dispatch with a batch sweep
(`fr14_lane5a_head_gemm_microbench.py`, `nvfp4_lmhead_characterization.py`). A
synthetic-weight duplicate was written here and then deleted rather than shipped
as a second, worse answer to the same question.

Reading their artifact as it stood when this note was written
(`lane5a_head_gemm_microbench.json`, schema `fr14.lane5a.head_gemm_microbench.v1`):
the NVFP4 head at **M = 1 costs 3.500 ms** — 204 GB/s achieved, 74.9 % of the
pinned 273 GB/s, against a 2.620 ms byte floor. Five head reads per step is
therefore **≈ 17.5 ms/step of drafter head against a 13.098 ms floor: ≈ +4.4 ms
above floor, in the projection.** That is the same size as the ≈ 4.5 ms that
flagged this whole surface, and it is 14× the entire selection stage.

**The number and its interpretation are lane 5A's, not this lane's** — cited
here only because it closes the question this lever was greenlit against. What
this note establishes on its own evidence is the **bound**: the selection stage
is 0.348 ms/step, so it is not where the milliseconds are.

## 7. Integration

Env flag `FR14_FUSED_DRAFT_TOPK`, strict `"0"`/`"1"`, **default `0`**, validated
fail-loud (a typo is never read as OFF). Armed only under the exact served profile;
anything else is a hard `RuntimeError` at the first `propose()`:

* `_fr13_is_fixed32` and `_fr13_single_logits`
* **K0 only** — `FR13_DRAFT_VOCAB_ROOT=0` and `FR13_DRAFT_VOCAB_K=0` (under K64 the
  row is 65 536 wide and the kernel, compiled for 248 320, would read past it)
* mutually exclusive with `FR13_DFWD_K64_TOP3`
* `_fr10_is_wide` with widths exactly `(3,3,3,3,3)` at the five head depths
* `batch_size ∈ {1,2,3,4}`
* binary identity: `FR14_FUSED_DRAFT_TOPK_SO` + `FR14_FUSED_DRAFT_TOPK_SHA256`,
  verified **before** `torch.ops.load_library`
* per-call: logits `[B,248320]` bf16 stride `(248320,1)`; outputs int64 with the exact
  strides; root call must be **outside** capture, loop calls **inside** it
* `FR14_FUSED_DRAFT_TOPK_BLOCKS` (default 64, range 1..121)

Static per-batch homes are allocated once and never reallocated, because the drafter
graph bakes their addresses.

The build attestation claims nothing it has not measured
(`status: BUILT_UNQUALIFIED`, `byte_equality_claim: false`,
`production_default_enabled: false`). Its reproducibility credential is the
**extracted `sm_121a` cubin sha256** (`cuobjdump --extract-elf`), not the `.so`
sha256 and not a SASS-text digest — see §7.1.

### 7.1 A build-reproducibility finding, offered to the FA2 credential problem

Pass 32 banked **FINDING 2, still open for Mark**: the FA2 `.so` sha256 is not
rebuild-reproducible, nvcc stamps `tmpxft_<pid>_<counter>` into ~87 kB of
host-side name-table bytes, and six hard-fail sites pin that sha — so a
byte-identical-source rebuild has a coin-flip rejection chance. Pass 37 answered
it by splitting the credential (SASS digest attests the kernel, `.so` sha attests
the artifact).

Building this kernel reproduced the problem **one level deeper, and then fixed
it.**

* A SASS-text digest is **not available** in the pinned image at all: it ships
  `cuobjdump` but no `nvdisasm`, so `--dump-sass` fails outright. The credential
  pass 37 chose cannot be computed here.
* `cuobjdump --extract-elf` needs no `nvdisasm`, and the extracted cubin **is**
  the device code — strictly stronger than disassembled text. Verified identical
  whether extracted from the linked `.so` or from the relocatable `.cuda.o`.
* But the first two builds of byte-identical source produced **different cubin
  shas**. Cause, read straight out of `--dump-resource-usage`: the kernel lived
  in an **anonymous namespace**, so nvcc mangled it as
  `_ZN58_GLOBAL__N__0d861e1e_22_fr14_dfwd_full_topk_cu_66fa687e_412...` — and
  that middle token is a per-build hash sitting inside the cubin's symbol table.
* Moving the kernel into a **named** namespace fixes the symbol
  (`_ZN26fr14_fused_draft_topk_impl28fr14_fused_draft_topk_kernelE...`). Three
  independent builds then produced **one cubin sha256
  `194f1ef28781c641…` and one `.so` sha256 `8f7a99e78c0898a4…`, 181 328 bytes** —
  the shared object is byte-reproducible too.

That is a **lead, not a claim**, for the FA2 case: it is a different translation
unit with a different symbol population, and nothing here was tested against it.
But it is cheap to test, and if FA2's non-reproducible bytes are also
anonymous-namespace mangling, the credential split pass 37 was forced into may
not be needed — a named namespace would restore the plain `.so` sha as a
first-class credential. Handing it to whoever owns that item.


## 8. What remains before serve-promotion

1. **Launcher forwarding — DONE 2026-08-18, and now PROMOTED-ON. See §10.**
   (original text preserved below)

1. ~~**Launcher forwarding — NOT DONE, deliberately.**~~ The three env vars
   (`FR14_FUSED_DRAFT_TOPK`, `_SO`, `_SHA256`, optionally `_BLOCKS`) are not yet in the
   `-e` block of `scripts/fr13_launch_forked_fa2_tree_server.sh`,
   `scripts/fr14_leg3_launch_nomiddleware.sh`,
   `scripts/fr14_armb_leg3_launch_nomiddleware.sh` or
   `scripts/fr14_run_b1_max_stack_serve.sh`. Launcher territory is owned by the
   concurrent lever-2 (suffix-aware MTP pass gating) lane this evening; this lever kept
   its diff to the in-pass logits/top-k path so it could land first. Until those lines
   exist the flag **cannot be armed in a serve**, which is also why it cannot be armed
   by accident.
2. **A live B1 serve A/B on the instruments** — `step_wall_ms`, `s_per_fwd_gpu`, and
   the `dfwd` span. `FR13_DFWD_SPLIT` should be armed for it: its `lmhead` span is
   exactly the bracket this lever moves, and the flag now reaches the container
   (fixed in the host-instrument pass). **A stack-level `dfwd` delta of
   0.3078 ms is small against a 49–53 ms span**, so the serve A/B needs
   the span timer, not the step total, to resolve it.
3. **B4 coverage.** The gate sweeps rows 1..4 and the guard admits them, but only the
   B1 arm is served today; a B4 promotion needs its own serve evidence.

## 9. Open item for Mark (Tier-B, not taken)

`argmax` ≠ `topk(...).indices[:, 0]` on **10/48 (21 %) of realistic
draft-logit rows** at this geometry. Because the packer fills the rank-0 slot from
`argmax` and the rank-1 slot from `topk`, that node's first sibling is then a
**duplicate of the spine token**: one of 27 active draft nodes spent on a token the
tree already contains, and at temperature 0 an argmax tie between two identical
candidates.

`FR13_DEDUP_SIBLINGS` exists for exactly this and is **explicitly disabled for
fixed32** (`... and not _fr13_is_fixed32`). Fixing it — e.g. sourcing the rank-0 slot
and the sibling ranks from one consistent selection, which this kernel already
computes for free — would change proposals, hence acceptance, hence it is **Tier-B and
Mark's call**. It is recorded here rather than taken.


---

# 10. PROMOTED (2026-08-18, Mark's ruling — pass 57)

`FR14_FUSED_DRAFT_TOPK` now defaults to **1** in both launcher families
(`fr13_launch_forked_fa2_tree_server.sh`, `fr14_armb_leg3_launch_nomiddleware.sh`).

**Evidence the ruling rests on (pass 57):** live byte-proof **268 paths / 0 diffs /
96 215 steps**, bracket **−0.071 ms**, accept **flat**, eyeball **clean**. §8's blocking
item — launcher forwarding — was closed first (strict `0|1`, default OFF); this promotion
flips that default.

## 10.1 What "promoted default" means here

The flag alone is not enough: a default that still required the caller to supply a credential
would not be a default. So the **artifact and its credential are launcher literals** now:

| variable | promoted default |
|---|---|
| `FR14_FUSED_DRAFT_TOPK` | `1` |
| `FR14_FUSED_DRAFT_TOPK_SO` | `/workspace/output/fr14_fused_draft_topk_build/fr14_dfwd_full_topk_sm121a.abi3.so` |
| `FR14_FUSED_DRAFT_TOPK_SHA256` | `8f7a99e78c0898a4221f045aa8e15a8085883dbc41b08f609da0da71e66a449e` |
| `FR14_FUSED_DRAFT_TOPK_BLOCKS` | `64` (unchanged) |

That sha is the **`.so`** sha256 — which is what the patcher actually hashes before
`torch.ops.load_library` — and §7.1's named-namespace fix is what makes it a legitimate
credential: three independent builds produced one `.so` sha at 181 328 bytes. Verified
against the artifact on disk, and pinned by a test.

## 10.2 Nothing relaxed

**A promoted default that silently fell back to the unfused path on a missing binary would be
a silent no-op**, so a missing, symlinked or mismatched `.so` is a **refusal**, not a
fallback. The launcher now proves the artifact **host-side, before the container starts** —
the patcher re-hashes it inside the container as well, so the host check exists to turn an
engine death mid-boot into a readable launch refusal.

Executed, not asserted (neither launcher has a dry-run mode, so the validation block is
extracted and run):

| case | result |
|---|---|
| plain launch, nothing set | **arms the promoted kernel**, rc 0 |
| `FR14_FUSED_DRAFT_TOPK=0` | opts out, rc 0 |
| `.so` missing | refuse, rc 2 |
| `.so` sha mismatch | refuse, rc 2 |
| flag not `0`/`1` | refuse, rc 2 |
| `_BLOCKS` out of 1..121 | refuse, rc 2 |

`FR14_FUSED_DRAFT_TOPK=0` remains the opt-out the paired A/Bs need.

## 10.3 One consequence to flag

The promoted artifact lives at `output/fr14_fused_draft_topk_build/…`, and `output/` is
**gitignored** — the `.so` is untracked and exists only on the box that built it. With a
promoted default plus hard refusal, **a fresh clone cannot launch until the kernel is
rebuilt**. That is the ruling's intent (refusal over silent no-op), and the refusal message
says so explicitly and names the rebuild:

```
FR14_FUSED_DRAFT_TOPK is PROMOTED-ON but its pinned .so is missing: <path>
  the artifact lives under output/ (gitignored), so a fresh tree must rebuild it:
    python3 scripts/fr14_build_dfwd_full_topk.py
  or set FR14_FUSED_DRAFT_TOPK=0 to opt out. Refusing rather than silently
  serving the unfused path.
```

If the intent is instead that any box can serve without a rebuild, the artifact needs a
tracked home or a build step in the launch path — flagged, not decided here.
