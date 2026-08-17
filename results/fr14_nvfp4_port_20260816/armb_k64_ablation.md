# FR14 Arm B — K64 vs full-vocab drafting on the aggressive NVFP4 head

**Lane: DIAGNOSTIC. Calibration-grade, NON-CITABLE.** Screen for the next config
train; not an acceptance instrument.

2026-08-17 · GB10, uncontended (engine only) · RadixArk NVFP4 @ `554ebba9`

## The question (Mark, via REDTEAM pass 12)

K64 draft-vocabulary subsetting exists to make the five draft-head reads cheap.
Under the fp8-era checkpoint that was worth **+34.3 ms of floor** (full-vocab
153.938 vs K64 119.658). Under the RadixArk NVFP4 head it is worth **+0.807 ms**
(93.152 vs 92.345), because five full-vocab reads of a *quantised* 0.715 GB head
cost about what five BF16-dequantised K64 slices cost.

So the byte argument for K64 is gone. Three things it might still buy:

* **DFWD compute** — 65,536 vs 248,320 rows per draft-head GEMV, five times per
  event. The floor is weight traffic only and cannot see this.
* against which, **K64 costs acceptance** whenever the drafted token falls
  outside its measured 128-id block map.
* and **K64 costs machinery** — the DVK shim, the Phase-1 boot dequant, the
  block map, and a Phase-2 FP4 draft-head GEMV that does not exist yet.

Only the measured wall settles it. Both arms, identical content.

## Result

| | **K64** (as built) | **K0** (full-vocab) | delta |
|---|---:|---:|---:|
| mandatory weight bytes | 25,210,209,416 | 25,430,574,256 | +220,364,840 |
| weight floor | 92.345089436 ms | 93.15228665201465 ms | **+0.807 ms** |
| accept / event | 2.7388 | **3.2143** | **+0.4755 (+17.4%)** |
| committed / event | 3.7388 | 4.2143 | +12.7% |
| **step wall** | **157.744 ms** | 161.587 ms | **+3.843 ms (+2.4%)** |
| decode-only TPS | 23.644 | **26.033** | **+10.1%** |
| client output TPS | 23.23 | **25.68** | **+10.5%** |
| mean TPOT | 42.36 ms | 38.30 ms | −4.06 ms |
| floor ratio | 1.7082 | 1.7347 | +0.027 |
| tok / draft | 31.00 | 31.00 | — |
| spec decode events | 2,205 | 1,955 | −11.3% |

Internally consistent: `committed_per_event / step_wall` gives 23.70 and 26.08
tok/s, within 0.25% of the measured decode-only 23.644 / 26.033.

### Decomposing the step-wall delta — this is the part that answers the question

K0's step is 3.843 ms more expensive. Only **0.807 ms** of that is bytes (it is
exactly the floor delta). The remaining **3.036 ms is compute** — the extra
5 × (248,320 − 65,536) rows of draft-head GEMV that K64 was buying and the floor
could never see.

So K64's advantage is real and it is almost entirely *compute*, not traffic:
**~3.0 ms/step.** Against that it gives up 0.476 accepted tokens per event. On
this workload the acceptance wins by roughly 4:1 in throughput terms.

## Reading

**On this workload, K0 wins by ~10%, and K64's remaining justification is
~3.0 ms/step of DFWD compute.** Everything the fp8 era built K64 for — the
34.3 ms byte saving — is gone; what is left is a compute saving smaller than the
acceptance it costs.

K0 additionally retires, at zero engineering cost: the DVK shim, the Phase-1
boot dequant, the 128-id block map, and the block-map-maintenance problem. It
drafts through the stock fp4 GEMM.

**The workload caveat that must travel with this number.** Random 1024-token
prompts are the *worst case* for K64's block map, which was measured from our
own agentic-code trace corpus. Leg 3 put a number on the gap: accept **4.526**
on real SWE agent traffic vs **2.358** here. On SWE traffic the map hits, so the
K0−K64 acceptance gap should shrink and could invert. **This result is a lower
bound on K64's cost, not a verdict on the deployed workload.** The clean
follow-up is the same pair on real SWE traffic; until then K64 stays as built
(which is why the stock B1 serve runs K64 — see below).

**Consequence for the ceiling ladder, worth stating because it is not obvious.**
Phase 2 (reading the K64 slices *as* NVFP4, floor 83.511 ms, −8.834 ms) is
predicated on there being a slice. Under K0 there is none, and the floor is
93.152. But the step wall here is ~158–162 ms of which only ~92 is bytes, so
Phase 2 could move the wall by at most ~8.8 ms ≈ **5.5%** — *less than the
10.5% K0 already delivers*, and it requires building an FP4 GEMV unit that does
not exist. If the SWE-traffic rerun confirms K0, the Phase-2 kernel work is
worth less than it looked.

## Method

Arm A's leg-3 vehicle (`ablation_a_leg3.md`), unchanged except for the
checkpoint and the variable under test:

* **engine only** — no SWE runner, proxy or agent, so nothing competes for the
  GB10's unified memory;
* **middleware dropped** — the fixed32 ASGI ingress guard authenticates a
  per-task HMAC header only the SWE harness mints.
  `scripts/fr14_armb_leg3_launch_nomiddleware.sh` is a copy of the arm-B
  launcher whose only difference is `FR13_FIXED32_MIDDLEWARE_FLAGS=""`; the
  ablation asserts that parity before booting anything;
* **deployment sampling** (temp 0.6 / top_p 0.95 / top_k 20) — the fixed32
  committer hard-refuses greedy;
* **client** `sglang.bench_serving --backend vllm`, random 1024/1024,
  `--num-prompts 8 --max-concurrency 1 --seed 1`. The seed is pinned, so both
  arms saw byte-identical prompts.

`FR13_NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0` is set for the K0 arm only — the
launcher gates full-vocab mode behind that sanctioned override so nobody drifts
off K64 by accident. Audited: every other gate reading it sits inside a
lever/qualification block this ablation does not arm, so it unlocks the
full-vocab arm and nothing else.

## Engagement and identity proof

* **PID-1 argv is BYTE-IDENTICAL between the two arms** (`diff` of
  `armb_k64ab_{k64,k0}_engine_cmdline.txt` is empty). The only differences are
  environment: `FR13_DRAFT_VOCAB_K/ROOT`, the two floor variables,
  `FR13_NEEDS_ALLOW`, and run-dir paths.
* `tok_per_draft = 31.00` exactly in both arms — the fixed32 tail6 tree ran on
  **every** decode step; the merged drafter logged `[FR13_MERGED ENGAGED]`.
* **K64**: `[FR13_DRAFT_VOCAB] shim built K=65536 (head rows 248320->65536)
  mode=gather`, `[FR14_DVK_DEQUANT] phase1 ... bytes=671088640
  logical_widths=[65536] output_size_per_partition=65536
  quant_method=UnquantizedEmbeddingMethod`, `[FR13_DRAFT_VOCAB_ROOT] engaged`.
* **K0**: **zero** shim / dequant / root-engaged lines across the whole run log
  — the Phase-1 dequant is inert, read from the log rather than assumed. The
  mechanism is `_fr13_dvk_prepare`'s own `_fr13_dvk_configured <= 0` early
  return, **not** the root flag: there are two call sites and the second is
  taken exactly when `not _fr13_dvk_root`.
* lm_head routed to `ModelOptNvFp4LinearMethod` on both heads in both arms; zero
  tracebacks in either.

## Against the rest of the triangle (same 1024/1024 shape, bs=1)

| | engine | output TPS | accept |
|---|---|---:|---:|
| leg 1 (pass 11) | sglang EAGLE 3/1/4, greedy | 26.10 | 2.90 |
| leg 3 (pass 12) | our tree, **arm A** conservative bytes | 19.27 | 2.358 |
| **this, K64** | our tree, **arm B** aggressive bytes | **23.23** | 2.739 |
| **this, K0** | our tree, arm B, full-vocab drafting | **25.68** | 3.214 |

Two independent things moved between leg 3 and arm-B K64 on the identical
shape: the step got **8.1% cheaper** (171.6 → 157.7 ms — the aggressive bytes
passing through) *and* acceptance rose **16%** (2.358 → 2.739) with the same
tree and the same drafter topology, which is a checkpoint-quality effect, not a
tree effect. Together the gap to their chain narrows from **−26%** to **−11%**,
and K0 closes it to **−1.6%** — on their workload, at our deployment sampling
against their greedy.

## Artifacts

- `armb_k64_ablation.json` — the numbers
- `armb_k64ab_{k64,k0}_bench.txt` — bench_serving summaries
- `armb_k64ab_{k64,k0}_container_env.txt`, `..._engine_cmdline.txt` — identity
- `armb_k64_ablation.sh` — the vehicle
- `armb_k64_ablation_reduce.py` — the reduction (self-tested: it reproduces arm
  A's banked leg-3 numbers from arm A's own metrics bracket)
- `scripts/fr14_armb_leg3_launch_nomiddleware.sh` — the middleware-dropped launcher
