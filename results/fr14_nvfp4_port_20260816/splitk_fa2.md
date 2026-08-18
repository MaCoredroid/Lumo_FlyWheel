# Split-K FA2 (Tier-B) — the kernel, the numbers, and the one thing still missing

Status 2026-08-18. Lane 4 (GREENLIT split-K), FR14 campaign. Domain: the FA2 fork
C++ tree (`scripts/fr13_patch_fa2_tree_bias.py`) and its build script.

**Headline.** The kernel is built, it is deterministic, and it is **1.9–2.0× faster**
than the promoted arm at every context length measured — 13.38 → 7.00 ms/step at 23k,
a **−6.4 ms/step offline** win. On numerics the result is better than "acceptable":
measured against a float64 dense reference at the operand scale of the *real* served
model, **split-K is closer to exact attention than the kernel that ships**, in 11 of
12 cases, and it disagrees with the exact argmax **less** often (1 row in 384 vs the
served kernel's 2). **Mark's degenerate-eyeball condition is NOT discharged** — §7
says exactly why and exactly what it needs. Nothing is promoted.

Artifacts: `fr14_splitk_fa2_probe.py` (harness), `fr14_splitk_fa2_probe_result.json`
and `..._process_b.json` (raw), `fr14_splitk_fa2_build_attestation.json` (build),
`tests/test_fr14_fa2_qrow32_gqa_pair_splitk_b1.py` (codegen contract).

---

## 1. What was built

`treeattn_v2_design.md` §12 measured the two-parameter fit and found today's B1 tree
attention **78% parallelism-bound**: 12 CTAs on 48 SMs, and driving staged bytes to
*zero* at that CTA count would save 2.98 ms of 13.44. The whole win lives in the CTA
count. This is that lever.

| | promoted `gqa_pair` | **`gqa_pair_splitk`** |
|---|---|---|
| traits | `<256,64,64,4>` | **identical** |
| smem / warps / heads-per-CTA | 96 KiB / 4 / 2 | **identical** |
| grid @ B1 | `(3, 1, 4)` = 12 CTAs | `(3, 4, 4)` = **48 CTAs** |
| tiles walked per CTA | all `n_blocks` | `n_blocks / 4` |
| staged GB/step @ 23k | 4.53 | **4.53** (unchanged) |
| combine | none | FA2's `flash_fwd_splitkv_combine_kernel` |
| sentinel | 1179791670 | **1179791671**, *and* `num_splits == 4` |

`blockIdx.y` partitions the context walk four ways. Each CTA still stages one K/V tile
for both of its query heads and still walks its own tiles in the same reverse order;
it simply walks a quarter of them and writes a partial attention — running max, partial
denominator, partial weighted sum — into the stock split accumulators. FA2's own
combine then rescales each split by `exp(m_split − m_global)`, sums, and blends.

**Everything is derived, not retyped.** The translation unit comes from the promoted B1
unit by counted substitution; the API gate from the promoted gate the same way, changing
only the sentinel, the split count and the two accumulator pointers; the combine launch
from the already-qualified `split2` unit. A drift in any of those fails *in the
derivation* rather than silently forking two kernels apart.

**Tier-B by construction, and the tree already knew it.** Splitting the walk changes
per-row accumulation order, so `_fr13_fa2_qrow32_b1_require_same_reduction` refuses to
byte-compare this arm against the qrow16 reference *at all*. The arm is gate-only: it is
absent from `_FR13_FA2_QROW32_B1_PRODUCTION_ARMS` and reachable only by an operand
carrying **both** the sentinel and `num_splits=4`.

**Registers.** `REG:246 STACK:0 SHARED:1024 LOCAL:0` for the attention kernel, against
the promoted arm's `REG:243` — +3 registers, still 8 below the TU's `__maxnreg__(254)`
cap. The combine is `REG:36 STACK:0 SHARED:1104 LOCAL:0`. No spill anywhere.

---

## 2. Two defects the guards caught, and what was done about them

**(a) Seven `CALL`s in the combine.** The first compile tripped the build script's
LDL/STL/CALL contract with seven `CALL.REL.NOINC` — all inside FA2's combine kernel,
which the promoted arm never instantiates because it has no combine. `STACK:0` and
`LOCAL:0` throughout, so nothing had spilled; the calls were a **64-bit integer division
helper**, from two places: the unpadded-LSE tensor is built by composing three CuTe
layouts whose flat extent is an `index_t`, so every store evaluated an int64 div/mod;
and the output epilogue decomposes the flat index with two runtime divisions.

Both are pure *addressing*. At `sequences == 1` the composed unpadded layout reduces
exactly to the flat index (offset `q + h·seqlen_q` at `b == 1`, and the flat index *is*
`h·seqlen_q + q`), and `kStaticQueryRows` is a power of two so head and row fall out of
a shift and a mask. The launcher `TORCH_CHECK`s `b == 1`, `h == 24`, `seqlen_q == 32`
before launching, so neither reduction is an assumption. Combine `REG` fell 46 → 36.

The alternative was to widen the SASS contract to admit `CALL` when `STACK` is 0. That
would have been a guard relaxed to make a build pass, and it would have left an
unmeasured division in the hot path of the kernel this lane exists to make faster.

**(b) A ULP metric that inflated its own tail.** The first probe run reported a max
output ULP of 38 592 — not a rounding difference, a bug in the *measurement*. The
monotone key used `+2^(n−1) − bits` for negatives instead of `−2^(n−1) − bits`. Both are
monotone *within* a sign, so the constant cancels for same-sign pairs and the two agree
there; the wrong one misreports only pairs straddling zero, where it inflates the
distance by `2^n`. Fixed, with a self-check that runs before the probe does. Had it
shipped, this note would have led with a five-figure ULP number that meant nothing.

---

## 3. DETERMINISM — the hard gate. **PASS**

The combine visits splits in index order with a fixed `__shfl_xor_sync` butterfly and no
atomics. Measured rather than argued:

- **16 cases** (2 operand scales × 4 context lengths × 2 seeds) × **8 repeats each**,
  bitwise identical output *and* LSE within each process. Each call allocates its own
  split accumulators, so their device addresses move between repeats; an
  address-dependent or atomics-ordered reduction would show here.
- Repeats are **interleaved with the reference arm**, so the candidate never sees the
  same allocator or launch state twice.
- **Cross-process**: two independent container runs produced **identical digests on all
  16 cases** (`probe_p0` vs `probe_p1`; verified equal key-by-key). A third run agrees.

No STOP condition fired.

---

## 4. ULP CHARACTERIZATION — vs the served `gqa_pair` kernel

**The operand scale matters, and the banked byte probe's scale was the wrong one.** It
drew q, k, v as `randn * 0.1`, which at this geometry gives pre-softmax logits with
std ≈ 0.01 — an essentially *uniform* attention distribution, the easiest possible case
for a split-K rescale because every split's running max is nearly equal. Measured from
16 banked `FR13_TREE_ATTN_OP_CAPTURE` artifacts of the served model (same 1/√256 scale):
**query std 1.234, key std 1.404, value std 1.731**, giving logit std ≈ 3.6 with max
≈ +13. That is the peaked regime where one split holds the global max and the others are
rescaled by `exp(−10)` or smaller. Both regimes are reported; the captured one is what
the verdict rests on.

Real fixed32 geometry, 4 context lengths (20 480 / 23 000 / 32 768 / 40 960) × 5 seeds,
3.93 M output elements per scale.

| | **captured scale** | legacy 0.1 scale |
|---|---|---|
| output ULP = 0 | **61.6 %** | 63.9 % |
| = 1 | 27.1 % | 26.0 % |
| = 2 | 4.6 % | 4.2 % |
| = 3 | 1.9 % | 1.7 % |
| = 4 | 1.1 % | 1.0 % |
| > 4 | 3.7 % | 3.3 % |
| **output max abs delta** | **3.906e−03** | 1.526e−05 |
| output max ULP, same sign, \|x\| ≥ max/1024 | **126** | 662 |
| **LSE max ULP** | **4** | 5 |
| LSE max abs delta | 3.815e−06 | 4.768e−06 |
| argmax flips vs the served arm | **8 / 640 (1.25 %)** | 3 / 640 (0.47 %) |

**Read the absolute number, not the ULP tail.** Output magnitudes at captured scale run
to ≈ 0.85–1.07, where one bf16 step is 1.95e−03. The worst observed disagreement,
3.906e−03, is therefore **1–2 bf16 ULP at full magnitude** — squarely the last-few-ulp
scale the STOP condition names, and **no STOP condition fired**. The long ULP tail is
what a *step count on the float grid* does to near-zero elements: values three or four
binades below the tensor maximum are hundreds of grid steps apart while being
numerically negligible. Measured rather than assumed: only 1.1 % of the >4-ULP tail is a
sign disagreement, and among elements at or above 1/1024 of the tensor's own maximum the
worst same-sign distance is **126 ULP** (≈ 0.5 binade on an element ≤ 1/1000 of full
scale). LSE, which is fp32 and never near zero, sits at **≤ 4 ULP** everywhere.

---

## 5. WHICH ARM IS CLOSER TO EXACT? — the comparison a byte gate cannot make

"Split-K differs from the served kernel" is not by itself a finding: the served kernel is
not exact either. Both approximate the same attention, so the question is which sits
closer. The probe builds a **float64 dense reference** to the kernel's own contract (tree
bias on columns `[seqlen_k−32, seqlen_k)`, same 1/√256 scale) and measures each arm
against it.

| captured scale | `gqa_pair` (**served**) | `gqa_pair_splitk` |
|---|---|---|
| output RMS error vs float64, L=20 480 | 1.172e−04 | **1.121e−04** |
| … L=40 960 | 8.384e−05 | **8.055e−05** |
| output max abs error | 1.9955e−03 | 1.9955e−03 (identical) |
| LSE RMS error | 1.025e−06 | **8.531e−07** |
| elements at or below the bf16 quantization floor | 62.3 % | **65.0 %** |
| **argmax disagreements with exact, all cases** | **2 / 384** | **1 / 384** |

**Split-K is closer to exact on output RMS in 11 of 12 cases** (the exception is a tie at
the legacy scale: 9.985e−07 vs 9.986e−07) and closer on LSE RMS in all of them. This is
the expected direction — four partial sums of 90 tiles each accumulate less rounding than
one sum of 360 — but it is measured, not derived.

**This is what makes the 1.25 % arm-vs-arm flip rate interpretable.** Those flips are
bf16 noise both arms carry, and split-K is on the *better* side of it: against the exact
argmax the served kernel flips twice where split-K flips once. The change is not a
degradation of the attention; it is a small improvement that happens to be visible
because bf16 output has only 8 mantissa bits.

---

## 6. PERFORMANCE — measured, and honestly against the prediction

Median of 3 independent runs, ms/step over the 16 full-attention layers, B1, kernel-level
(no serve). Timing at captured operand scale.

| seq_len | `qrow16` 48 CTAs | `gqa_pair` 12 CTAs | **`splitk` 48 CTAs** | Δ vs served | speedup | fit predicted | measured over |
|---|---|---|---|---|---|---|---|
| 20 480 | 13.084 | 11.920 | **6.093** | **−5.83** | 1.96× | 5.001 | +21.8 % |
| **23 000** | 14.656 | 13.384 | **6.999** | **−6.39** | 1.91× | 5.608 | +24.8 % |
| 32 768 | 20.547 | 19.315 | **9.558** | **−9.76** | 2.02× | 8.043 | +18.8 % |
| 40 960 | 26.150 | 23.800 | **11.690** | **−12.11** | 2.04× | 9.990 | +17.0 % |

**The fit under-predicted the cost by 17–25 %, consistently, and that gap is real.** The
§8 model is `T = α·staged_GB + β/CTAs`; it has no term for a second kernel launch, and
split-K adds one — a 192-block combine plus the implied global sync between the two
launches. Its cost is roughly fixed (≈ 1.0–1.7 ms/step across the range), which is why
the relative over-prediction *shrinks* as context grows. The model is still useful: it
called the direction and got the magnitude within a quarter.

**Against the design doc's number.** §12 predicted 3.61 ms at 23k for split-K. That row
assumed read-once staging (1.51 GB) *as well as* 48 CTAs, i.e. G=6 **and** 12 splits.
This arm keeps the promoted arm's staged bytes (4.53 GB), so the like-for-like prediction
is 2.98 + 2.61 = **5.60 ms** — which is exactly what the probe's own fit produced. The
remaining 1.4 ms is the combine.

**In-serve projection, and its assumption.** The served `gqa_pair` costs 29.9 ms/step
in-serve against 13.38 offline at 23k, a 2.23× overhead. *If* that overhead is
multiplicative, split-K lands at ≈ 15.6 ms/step, a **−14.3 ms/step** in-serve win against
the −17 the lever was briefed at. That is a projection, not a measurement, and it is
probably optimistic: the combine's launch overhead is fixed, so it should scale *worse*
than multiplicatively in serve. **Measure it; do not bank it.**

---

## 7. GENERATION PROBE + DEGENERATE EYEBALL — **NOT DISCHARGED**

Mark's condition on this lane is that generation traces be read for degeneration before
any promotion. **This has not been done, and I could not do it within this lane's
boundaries.** Being explicit about why, because "we ran out of time" and "it is not
reachable from here" are different claims and this is the second:

The fixed32 tree kernel is only ever called with **32 query rows**. Nothing but the
campaign's tree-speculative decode produces that shape — plain decode has one query row,
and the in-binary gate hard-refuses anything else. So a generation loop with this kernel
in it requires the forked tree-attention serve
(`scripts/fr13_launch_forked_fa2_tree_server.sh`) with its custom drafter, hydra27
topology and K64/root1 draft vocabulary. There is no offline forward loop in the tree
that reaches the kernel: the only other consumers are replay harnesses over banked
captures, and the banked `TREE_ATTN_OP_CAPTURE` artifacts are hydra10-era (10-row trees,
5 query rows, short context) — the wrong geometry for this kernel, though their operand
*statistics* were used in §4 and are the reason the characterization was run at a
realistic softmax sharpness at all.

Standing that up is the paired A/B serve itself, which is the promotion step this
condition gates — and it needs an arm branch in the launcher's `_FR13_FA2_QROW32_B1_PIN_ARM`
case, which is the committer lane's territory, not this one's. I did not go and take it.

**What the offline evidence does and does not say about degeneration.** It says the
mechanism by which a kernel delta becomes a degenerate stream — logit perturbation large
enough to flip an argmax — is *no more likely* under split-K than under the kernel that
ships today: against exact attention the served kernel flips 2 argmaxes in 384 and
split-K flips 1 (§5). It does **not** say anything about repetition loops, gibberish,
truncation or tool-call malformation, because no text was generated. Those are the
signatures Mark asked to see, and no offline number substitutes for reading them.

---

## 8. Build regime — both credentials, from birth, plus one that was not asked for

`scripts/fr14_build_fa2_qrow32_gqa_pair_splitk_b1_sm121a.sh`, derived from the sealed
gqa_pair builder by counted substitution: the ABI audit, the 55 byte-reused reference
objects, the mandatory offline torch load and the SASS contract are the *same code* that
qualified the promoted arm.

| credential | value | when asserted |
|---|---|---|
| source closure | `4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878` | before compiling |
| **SASS digest (this kernel)** | `3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e` | before the link |
| **baseline SASS digest** | `fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470` | before the link |
| **`.so` sha256** | `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857` | after the link |
| **`.so` size** | 300 123 792 B | after the link |
| ABI diffs (defined / dt_needed / runtime_path / undefined) | 0 / 0 / 0 / 0 bytes | after the link |
| forbidden SASS (LDL/STL/CALL) | 0 bytes | before the link |

The staged binary is at
`/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so`,
built from a clean sparse clone of the pushed branch at
`/home/mark/fr14_splitk_build_repo_20260818` (the shared worktree carries four other
lanes' in-flight changes, so the builder's clean-tree preflight was satisfied the way
the FR14 build-environment proof did it, not weakened). Rebuild:
`FR14_BUILD_B1_GQA_PAIR_SPLITK=1 PYTHON_BIN=<repo>/.venv/bin/python BUILD=<new dir>
bash scripts/fr14_build_fa2_qrow32_gqa_pair_splitk_b1_sm121a.sh` — expect exit 97 on the
`.so` sha unless the container lands the same nvcc PID; the SASS digests are what tell
you the kernel reproduced.

**Bootstrap without a half-credentialed artifact.** A brand-new arm has no pin to assert
against, and minting one after a link would mean a linked artifact existed before it was
credentialed. So the first build stops *before* the link (exit 97), prints the digest,
and the value is written into the script; from then on the assertion is live. The
bootstrap path can only ever refuse to link.

**The baseline credential, which nobody asked for and which turned out to matter most.**
Every ULP, argmax and timing number here is measured against the `gqa_pair` kernel *in
this binary*, so the builder compiles the promoted unit too and asserts its SASS digest
against the **sealed 2026-08-10 kernel's** pin, plus its `REG:243 STACK:0 SHARED:1024
LOCAL:0`. It passes. That does two things: it proves the baseline is the kernel that
actually serves rather than a rebuild that resembles it, and — because the split-K header
edits (the paired O/LSE tensors learning to address the stock split accumulators, and the
combine's static geometry) are supposed to be **inert at `Split=false`** — it is the
*measurement* of that invariance claim. A claim of invariance that nothing checks is a
claim, not a property.

**The `.so` sha is still not rebuild-reproducible, and the guard fired.** Four links from
the same closure produced **one** split-K SASS digest and **one** baseline SASS digest,
with **two** distinct `.so` hashes at an identical 300 123 792 bytes — the nvcc
container-PID stamp the credential note documents, now observed on a second arm. The
staged-artifact guard's first live execution refused a PID-shifted rebuild whose kernel
was provably identical (exit 97), which is exactly the intended behaviour: the evidence
in this note names one binary.

**Regression on the promoted arm.** With the split-K flag off, the gqa_pair source
closure re-derives as `172b5e71…` byte for byte, its four `static_assert(!Split)` are
intact, and neither FR14 marker appears. That is the first test in
`tests/test_fr14_fa2_qrow32_gqa_pair_splitk_b1.py`, and it is the load-bearing one.

---

## 9. What remains before a paired A/B serve

1. **The generation probe and the eyeball (§7).** Blocking, by Mark's own condition.
   Needs a `gqa_pair_splitk` branch in the launcher's `_FR13_FA2_QROW32_B1_PIN_ARM`
   case carrying this binary's pins, and a live-arm route that does not go through the
   raw-byte gate — which structurally refuses this arm, correctly. That plumbing is the
   committer lane's territory; the numbers it needs are in §8.
2. **The in-serve measurement (§6).** The −14.3 ms/step projection assumes the 2.23×
   offline→serve overhead is multiplicative. It probably is not, since the combine's
   launch cost is fixed. Measure both arms in the same boot.
3. **`num_splits` is not tuned.** 4 was chosen because 3 head pairs × 4 splits × 4 KV
   heads = 48 CTAs = one per SM on this device. 6 or 8 would trade a longer combine and
   more accumulator traffic for shorter walks; nothing here says 4 is optimal, only that
   it is the value the geometry makes obvious.
4. **B4 is untouched.** This arm is `sequences == 1` only, asserted at compile time by
   `static_assert(!kStaticQueryBatch || !Split || kStaticSequences == 1)`.
5. **F2 (cluster/DSMEM) is now the smaller prize.** The design doc's surviving Tier-A
   option was worth a predicted −1.99 ms. Split-K measures −6.4 at the same context, and
   the two are composable in principle: F2 attacks the bytes term, split-K the
   parallelism term. Not attempted here.

---

## 10. Verdict

**No STOP condition fired.** The combine is deterministic run-to-run and across
processes; the ULP deltas are 1–2 bf16 steps at full magnitude, and the arm is *closer*
to exact attention than the kernel that ships; the speedup is 1.9–2.0×, worth −6.4 ms/step
offline at 23k. There is no degenerate trace to report because **there is no trace** —
§7. The arm stays out of `_FR13_FA2_QROW32_B1_PRODUCTION_ARMS`, env-flagged and default
off, and it must not serve until someone has read its generations.


---

## 11. Launcher plumbing — landed 2026-08-18, and what it did NOT do

§9.1 asked for a `gqa_pair_splitk` branch in `_FR13_FA2_QROW32_B1_PIN_ARM` carrying this
binary's pins, plus a live-arm route that does not go through the raw-byte gate. Both are in,
in **both** launcher families (`fr13_launch_forked_fa2_tree_server.sh`,
`fr14_armb_leg3_launch_nomiddleware.sh`).

**Reachable as a LIVE arm only.** `FR13_FA2_QROW32_B1_LIVE_AB_ARM` now accepts
`gqa_pair_splitk`; `FR13_FA2_QROW32_B1_PRODUCTION_ARM` deliberately does **not**, so the arm
stays gate-only exactly as §1 requires, and the promoted `gqa_pair` production default is
untouched. A launch that never names the arm cannot reach one line of the new code.

**What the branch asserts**, beyond the four fields the `gqa_pair` arm checks:

- the two **SASS digests** — this kernel's `3f24d70d…` and the **sealed baseline's**
  `fa01f988…`. §8 is explicit that the `.so` sha is not rebuild-reproducible and the SASS
  digest is what attests the kernel, so pinning only the artifact would have pinned the
  weaker of the two credentials. Pinning the baseline digest is also what keeps
  "the split-K header edits are inert at `Split=false`" a measurement.
- the binary **on disk**: present, **not a symlink**, and **re-hashed** to the pinned sha.
  The generic B1 selector only compares `stat -c '%s'`, and this arm's two known links differ
  in sha at an identical 300 123 792 bytes — precisely the case a size check cannot separate,
  so without the re-hash the PID-shifted twin would have armed silently.

**Guard hygiene.** A named split-K live arm arms `_FR13_M32_GUARD_ACTIVE` like every other B1
live arm, and the two new pin variables are in `_FR13_M32_GUARD_NAMES`, so `.lumo.local.env`
cannot move them underneath a run.

**Tests.** `tests/test_fr14_splitk_arm_launcher_wiring.py`, 39 cases. The two that matter:
the launcher literals are checked **against `fr14_splitk_fa2_build_attestation.json`** rather
than being retyped constants that could drift from the artifact; and the branch is
**extracted and executed** rather than merely grepped — it accepts the real staged binary and
refuses a missing one, a wrong one at the right path, a symlink to the right one, a drifted
kernel digest, a drifted baseline digest, another arm's sha, and another arm's size.

**Arming recipe** (for the §7 generation probe — *not run here*, and the arm stays unarmed):

```
FR13_FA2_QROW32_B1_LIVE_AB_ARM=gqa_pair_splitk \
FORKED_FA2_SO=/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so \
FR13_FA2_QROW32_B1_SO_SHA256=28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857 \
FR13_FA2_QROW32_B1_SO_SIZE=300123792 \
FR13_FA2_QROW32_B1_FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95 \
FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878 \
FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST=3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e \
FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST=fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470 \
  <the usual live-A/B invocation>
```

The launcher's existing live-arm gate still applies on top and is unchanged: the canonical
instance `astropy__astropy-12907`, `FR13_FIXED32_B1_DIAGNOSTIC=1`, hydra27 at
`MAX_NUM_SEQS=1 / SWE_CONCURRENCY=1`, `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`,
`FR13_FA2_QROW32_B1_SOURCE_COMMIT == git rev-parse HEAD`, and the patcher source hash.

**CORRECTION (Arm S, promotion A/B 2026-08-18, commit `550e201cc`).** This section
originally closed with "the plumbing removes the reason the probe could not be run". That
was wrong, and the promotion-A/B runner proved it wrong by trying — three separate
refusals, none of which the bash pin case above could have fixed:

1. **The live boot refused at the launcher.** The split-K pins landed in the bash pin case
   but *not* in the in-container Python qualification map twenty lines of comment away,
   which fell through to `split2`'s pins; and `fr13_fixed32_contract.py` carried no split-K
   constants at all. `fixed32 qrow32 B1 binary identity is not qualified`.
2. **The live-A/B route refused `num_splits=4` by construction** — its
   `_fr13_fa2_qrow32_b1_require_same_reduction` check compares reduction topologies, and
   split-K's differs from the served reference's by design.
3. **That route never returned candidate output anyway** (`qrow16_reference_served=1`,
   `candidate_returned=0`). It is a *shadow* byte gate: it re-calls both arms on captured
   bundles and compares. No served token could have carried split-K attention.

The lesson is the one worth writing down: **I checked that the arm could be NAMED, not
that it could be REACHED.** A pin case that admits an arm is not a serving route, and the
distance between the two was three defects wide. All three are closed in §12.

---

## 12. The Tier-B qualification route — BUILT, and the credential is EARNED

Mark, FR14 pass 64: the Tier-B route is approved in full. Live-A/B serving on a Tier-B
credential now; promoted-default only after exact16 QC parity. The campaign's "lossless"
doctrine is refined to **faithful-to-the-model-within-FP-rounding, proven deterministic
and QC'd** — not bit-identical-to-the-incumbent. Byte-exact Tier-A remains the default
door for everything else.

### 12.1 Bounds, pre-registered before the runner existed

`fr14_splitk_tierb_bounds.json`, committed at `9d294733` — *before* the gate runner was
written — with the margin stated for every bound. Its sha256 is pinned in the sidecar, so
a bound edited to fit a result changes the digest and both the runner and the validator
refuse. A bound chosen after seeing the number it judges is not a bound.

| | bound | measured | margin |
|---|---|---|---|
| **B1** determinism, in-process **and** cross-process | bitwise, all cases | 16/16 × 8 reps, 2 processes | HARD |
| **B2** output within 2 ULP | ≥ 90 % | **93.31 %** | 3.31 pp |
| **B3** max abs delta, in bf16 steps of the tensor max | ≤ 4.0 | **1.384** | 2.89× |
| **B4** LSE max ULP | ≤ 8 | **4** | 2× |
| **B5** LSE max abs delta | ≤ 1e−4 | **3.81e−06** | 26× |
| **B6** argmax-vs-float64-exact | ≤ **the incumbent's** | **1 vs 2** | equal-or-better |
| **B7** output RMS-vs-exact ratio | ≤ 1.10 | **0.961** | 1.14× |
| **B8** LSE RMS-vs-exact ratio | ≤ 1.10 | **0.839** | 1.31× |
| **B9** non-finite disagreements | = 0 | **0** | HARD |

B6–B8 are the bounds a byte gate cannot express and the ones that decide the question:
two approximations disagreeing is not a defect unless the candidate is the *worse*
approximation. B7 exists because a ULP histogram cannot see a systematic bias — a kernel
can be uniformly slightly wrong and still sit within 2 ULP everywhere. A **probe-strength
floor** is pre-registered alongside them (≥ 4 context lengths spanning 20 480–40 960, ≥ 5
seeds, captured operand scale, ≥ 8 determinism repeats, ≥ 2 processes, ≥ 3.5 M elements),
because every bound above gets easier under a weaker probe.

### 12.2 The credential — EARNED

`scripts/fr14_gate_fa2_qrow32_splitk_tierb.sh` → `fr14_splitk_tierb_credential.json`.
**All nine bounds PASS. Determinism PASS in-process and across two independent processes.
Probe strength PASS.** Body digest `d04b65ec…`; file sha256 `2146654f…`.

It binds ten fields — arm, `.so` sha256 + size, source closure, FA2 head, **both** SASS
digests, HEAD, patcher digest, bounds digest — each individually tested, because a
credential that outlives a rebuild authorises numerics nobody measured. Its verdict is
**recomputed** from the recorded measurements at every validation rather than read out of
the file: a test forges `bounds_passed: true` over a measurement that fails B2, and the
validator still refuses.

**It must be re-earned when HEAD moves.** The credential binds `source_commit`, and the
launcher already requires `FR13_FA2_QROW32_B1_SOURCE_COMMIT == git rev-parse HEAD`. That
is the campaign's existing doctrine, not a wrinkle in this one; the gate is offline
kernel work and re-runs in about ten minutes.

### 12.3 Arm S's three refusals, closed

1. **Binary identity.** `fr13_fixed32_contract.py` gains the split-K constants
   (`QROW32_B1_SPLITK_*`, including both SASS digests) and a **named branch** in the
   pin-arm resolver; both launcher families gain the matching entry in the in-container
   Python qualification map, plus an assertion that a tier-b arm resolving a non-tier-b
   binary is a refusal. The defect was a fall-through default silently answering for an
   arm it was not written for; naming every arm is the fix.
2. **`num_splits=4` refused.** The byte gate's `require_same_reduction` is **unchanged**
   and still refuses split-K — correctly. The live route now *routes around* it for a
   tier-b arm and records a characterization instead: ULP distribution, worst
   disagreement in bf16 steps of the tensor's own maximum (comparable with B3), non-finite
   disagreements — on the **real served operands** the offline probe cannot reach. It also
   gains the live determinism gate that matters most: the candidate is called **twice per
   layer in-process**, with fresh split accumulators, and must return the same bits.
3. **Candidate output not returned.** `_fr13_fa2_qrow32_b1_serving_arm()` resolves Tier A
   first and unchanged, so a launch naming no tier-b arm cannot reach one line of the new
   path. Naming a tier-b live arm is still the shadow route it always was; **serving**
   additionally requires `FR13_FA2_QROW32_B1_TIER_B_SERVE=1` *and* a credential validated
   before the first served token — two independent things, so a reused live-A/B invocation
   cannot start serving Tier-B numerics by accident.

### 12.4 Nothing byte-gated became easier

The validator's **first** check is that the arm is tier-b marked — before a single
measurement is read, and regardless of how good the numbers are. Presenting this
credential for `nosplit`, `split2`, `visibility` or `gqa_pair` fails on the fact that
those arms have a byte-exact door, and the message says so.
`_FR13_FA2_QROW32_B1_PRODUCTION_ARMS` is unchanged. The contract's **production**
allowlist is unchanged and still refuses split-K. `require_same_reduction` is unchanged.
34 tests cover exactly these properties.

### 12.5 Arming recipe

```
FR13_FA2_QROW32_B1_LIVE_AB_ARM=gqa_pair_splitk \
FR13_FA2_QROW32_B1_TIER_B_SERVE=1 \
FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL=<credential path in container> \
FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256=<file sha256, re-earned at serving HEAD> \
  <plus the §11 binary/SASS pins and the usual live-arm gate>
```

**§7 is still not discharged.** The route now reaches the kernel with served tokens, which
is what §11 wrongly claimed already. Whether those tokens are *good* is a question only
reading them answers, and Mark's degenerate eyeball on the served generations remains
mandatory and undone.
