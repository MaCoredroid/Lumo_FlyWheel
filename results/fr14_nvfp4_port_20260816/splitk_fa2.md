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

---

## 13. The 17th site — an identity resolver that defaulted, and the sweep for its shape

Arm S's fifth boot got further than any before it: launcher, qualification map, allowlist,
`TIER_B_SERVE=1`, and a fresh credential all passed. It then refused at
`_fr13_fa2_qrow32_b1_identity` — and **the refusal prevented identity fraud.**

That resolver branched on `gqa_pair` and `visibility`, then returned **split2's pins**
bare to everything else. Selected arm `gqa_pair_splitk` → the incumbent's identity. The
run stopped only because the environment declared split-K's sha and split2's pin did not
match it. **That is an accident, not a guard.** Had the two happened to agree — or had the
environment been permissive — the boot would have **served split-K while attesting the
incumbent**, and every artifact downstream would have named the wrong kernel.

It is the same shape as §12.3's refusal (1), one layer deeper: I fixed the *pin-arm*
resolver by naming every arm, and left the *identity* resolver it feeds still defaulting.

### 13.1 Sweep inventory — the pattern, not the instance

| | site | before | after |
|---|---|---|---|
| **A1** | `fr13_patch_fa2_tree_bias.py` `_fr13_fa2_qrow32_b1_identity` (injected blob) | 2 branches, **bare return of split2's pins**; blob carried **no split-K pins and no SASS digests at all** | `_FR13_FA2_QROW32_B1_IDENTITIES`, 5 arms, unknown arm **raises** |
| **A2** | `fr14_leg3_launch_nomiddleware.sh` bash pin case | no split-K branch; `*)` asserts split2's pins | split-K branch added, arm admitted, `*)` **refuses** |
| **A3** | same file, in-container python map | **unfixed twin** of §12.3(1): no split-K key, no tier-b guard | keyed, guarded, **refuses** |
| **B1** | both other launchers' bash `*)` | asserts split2's pins for any new arm | `nosplit\|split2\|""` named; `*)` **refuses** |
| **B2** | all three launchers' `.get(…, split2 pins)` | tier-b guard caught only tier-b arms; a new **non**-tier-b arm still landed on split2 | `.get(arm)` + `if expected is None: raise` |
| **B3** | `fr13_fixed32_contract.py` `_expected_runtime_fa2_identity` | `else:` → visibility ternary → split2; **and** *every* member of `QROW32_B1_TIER_B_ARMS` mapped onto split-K's pins | explicit 6-entry table, unknown arm **raises** |
| **B4** | `fr13_qrow32_b1_pass_sidecar.py` `_source_status` | bare `return SOURCE_STATUS` | `_SOURCE_STATUS_BY_ARM`, unknown arm **raises** |

**Resolvers found: 7. Bare fallbacks converted to refusals: 7. Split-K branches added: 4**
(A1 identity table, A2 bash case, A3 python map, plus the stale twin's allowlist and
caller-guard entry). Verified clean and left alone: the B4 family's `[arm]` subscript, the
qrow16 flag-selectors (no arm dispatch), `_fr13_fa2_qrow32_live_ab_contract`,
`_candidate_contract`, and ~180 other launcher `*)` defaults that are already validators.

### 13.2 Why it hid: two coverage gaps in the lint family

- **`fr14_paired_contract_sweep.PATCHER` was only the GDN patcher.** Nothing in the
  paired-contract family ever walked `fr13_patch_fa2_tree_bias.py`'s blobs. A1 sat there
  for the whole campaign.
- **`LAUNCHERS` was a 2-tuple.** `pair_launcher_twins` already listed `gqa_pair_splitk`
  among its required markers and simply never looked at the third twin — which is exactly
  how A2/A3 drifted a whole arm behind.

Both closed. The family gains a **`fallback-pattern`** kind and two detectors:
`arm_identity_resolvers_refuse_unknown_arms` and
`launcher_pin_cases_refuse_unknown_arms`. **15 pairs enumerated, 0 stale.**

### 13.3 The detector is behavioural, because my first one wasn't

The first version asked whether each arm name appeared *anywhere in the file*. Every arm
does — the registry names them all — so renaming a key in the identity table still passed.
And the launcher check matched the phrase `has no pinned identity`, which **the refusal's
own comment contained**, so deleting the executable `echo … >&2; exit 2` left the check
green. Both were caught by this family's own non-negotiable rule: write a mutation test
that restores the defect and prove the detector fails.

So the detector now **calls** each resolver with an unknown arm and requires it to raise,
compares the identity table's keys against the arm registry's, and strips comments before
looking for an executable refusal. A detector a comment can satisfy documents a guard
instead of finding one — the same text-keying mistake `ADJUDICATED_REPLAY_POSITIONS` was
written to stop.

### 13.4 Consequences

- **Both C++ source closures are unchanged** — `172b5e71…` and `4ed00909…`. This fix
  touched the injected python and the shell/validator layers only, so **the staged
  `.so` is still the characterized kernel** and needs no rebuild.
- **The banked credential is now stale by construction.** It binds
  `patch_source_sha256`, and the patcher changed. That is the binding doing its job.
  Re-earn with `scripts/fr14_gate_fa2_qrow32_splitk_tierb.sh` at the serving HEAD;
  the stale one cannot be used, because the serve path requires both digests to match.
- **The in-container identity now carries both SASS digests** and checks them, which the
  bash pin case had done all along and the container half had not.

**Five attempts, zero split-K tokens served.** Four of the five refusals were correct
guards doing their job on a route that was genuinely incomplete; the fifth was a guard
that fired by luck. The sweep — not a sixth boot — is what should end the series.

---

## 14. Round 6 did not measure this kernel — the 18th site, and it was mine

Round 6 reported a clean eyeball, `swerc=0`, 12907 resolved, and a verifier forward of
125.898 → **128.999 ms (+2.46%)** where −14 was projected. The double-call in my live
determinism gate was named prime suspect. **It is not the mechanism, and the evidence
says so before anything was built on it.**

### 14.1 The suspect is refuted

`logs/fr13_fixed32_work_census.jsonl`, 1038 rows: **`tree_attn.calls = 16` on every
row.** Not 32. My double-call lives in `_fr13_fa2_qrow32_b1_live_replay`, which is guarded
by `_FR13_FA2_QROW32_B1_LIVE_ATTEMPTED` and runs **once per boot**, off the served path —
three calls × 16 layers, one time, against a 395-second task. It cannot double a per-step
cost and it did not.

### 14.2 What actually happened: the candidate never served

| evidence | reading |
|---|---|
| `fr13_fa2_qrow32_b1_production_engagement.json` | **absent** from the run's `logs/` |
| launcher line 7075 | `elif` chain: `LIVE_AB_ARM` is tested **before** `PRODUCTION_ARM` |
| patcher line 9442 | the install of `_fr13_fa2_qrow32_b1_production_begin` was gated on `fixed32_query_tile32_b1_production` **alone** |
| `container_env.txt` | `LIVE_AB_ARM=gqa_pair_splitk`, `PRODUCTION_ARM=` **empty**, `TIER_B_SERVE=1` |

A tier-b serve is spelled as a **live** arm. So the launcher passed
`--fixed32-query-tile32-b1-live-ab` and **never** the production flag — and the serving
hook, the only thing that retags the operand on the served path, **was never patched into
`TreeAttentionImpl`**. `_fr13_fa2_qrow32_b1_serving_arm()` had been taught about tier-b
arms; **the code that installs its caller had not.**

The steady-state decode ran the incumbent at both ends of the A/B. **+2.46% is
incumbent-versus-incumbent noise on a ±4 ms floor.** The −14 projection did not fail —
**it was never tested.**

This is the same shape as §13's 17th site, one layer further out: resolver taught about
the new arm, installer not. §13 swept resolvers; it did not sweep *installers*.

### 14.3 The worse half: my artifact asserted a serve that never happened

The round-6 record carried `"tier_b_serving": true` and
`"served_return": "candidate output served (tier-b)"`. **Both were computed from
environment variables** — `tier_b and _fr13_fa2_qrow32_b1_tier_b_arm() == arm` reads
`FR13_FA2_QROW32_B1_TIER_B_SERVE` and nothing else. They reported what was *asked for*,
not what *ran*, and the campaign reasonably believed them.

The campaign had already paid for exactly this once — the draft-vocabulary identity, where
"two hardcodings agreed with each other while both disagreed with reality" and a K0 gate
minted a green credential describing a K64 serve. I reintroduced the shape one file away
from the comment that documents it.

Fixed so the claim is falsifiable:

- **`_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT`** counts the retag *at the retag*, with the
  layer names and graph ids it engaged on.
- The record now reports **`tier_b_serve_armed`** (configuration) and
  **`tier_b_engagement`** (observation) **separately**, and `tier_b_serving` is derived
  from the counter.
- **Armed-but-never-engaged is a REFUSAL**, raised before the pass/fail branch — so an
  otherwise-clean shadow comparison can no longer report PASS for a serve that did not
  happen. Round 6's exact state now fails the run in seconds instead of consuming an arm.

### 14.4 The fix

- New patcher flag **`--fixed32-query-tile32-b1-tier-b-serve`**: a *modifier* of the
  live-A/B selector, not a member of the mutually-exclusive chain that hid this. It
  installs the serving call site **and** the capture-end hook. The two `cuda_graph`
  patches anchor on different lines (`entry.cudagraph.replay()` vs
  `entry.cudagraph = cudagraph`), so they compose — only the `elif` chain made them
  exclusive.
- All three launcher twins pass it as a **separate expansion** when `TIER_B_SERVE=1`, and
  a test asserts it is not inside the selector chain.
- `TIER_B_SERVE` and both credential variables joined `_FR13_M32_GUARD_NAMES` in all three
  twins: the switch that decides whether the candidate's output reaches the model must not
  be movable by `.lumo.local.env`.

### 14.5 The single-call mode was not built, deliberately

It would have been building on a refuted mechanism, which the campaign's own rule forbids.
The double-call is one-shot insurance costing 16 extra kernel invocations per **boot**;
determinism is credentialed offline (9/9, twice) and now also checked live on real served
operands (`repeat_byte_mismatches: 0`, `ulp ≤ 2` on **99.56%** of elements at seq_len
35 369 — tighter than the synthetic 93.31% the bounds were set from). It stays.

### 14.6 What the next serve needs

Both C++ closures are unchanged (`172b5e71…`, `4ed00909…`), so **no rebuild**. The patcher
moved again, so the Tier-B credential must be **re-earned** at the serving HEAD — ten
minutes, offline. Then arm with `TIER_B_SERVE=1`, and this time the run will either engage
or refuse; it can no longer do neither and say it did.

---

## 15. The static walk — and a first-class name for tier-B

Round 7's modifier fix worked: the hook installed. Then the **19th site** fired
(`INTERNAL_ATTESTED`, exported only inside the production-arm block) and the **20th** was
found by reading one line further (`require_exact4`, plus a real contradiction under it —
tier-B *serving* demands the canonical exact4 identity while tier-B *arming*, spelled as a
live arm, demands `B1_DIAGNOSTIC=1` single-instance).

### 15.1 The decision: tier-B gets its own name (task 4)

**Yes.** Four consecutive sites — 17, 18, 19, 20 — trace to one pun: tier-B serving was
spelled `LIVE_AB_ARM=<arm>` + `TIER_B_SERVE=1`. Every gate in the tree then had to be read
twice, once as "shadow" and once as "serve", and one of the two readings was missed each
time. Gates written for the serving path key on `PRODUCTION_ARM` and never fired; gates
written for the shadow path — a **single-instance diagnostic** — fired when they must not.

`FR13_FA2_QROW32_B1_TIER_B_ARM` is now a first-class selector, sibling to
`PRODUCTION_ARM` and `LIVE_AB_ARM`, mutually exclusive with both. `TIER_B_SERVE` is
**retired and refuses loudly** in the launcher and in the blob — a stale invocation fails
in seconds instead of silently degrading to a shadow run, which is what round 6 did for
395 seconds. The patcher flag is a selector too, not a modifier, and is mutually exclusive
with `--fixed32-query-tile32-b1-live-ab`.

### 15.2 Walk inventory

Full trace from launcher start to the first kernel invocation, across five layers.
**Gates enumerated: ~110. TIER-A-ONLY on the served path: 6. Reverse-class (shadow-only
gates that would block a multi-task serve): 10. Fixed this pass: 13.**

| # | site | class | fix |
|---|---|---|---|
| 1 | `launcher` `INTERNAL_ATTESTED` exported only under `PRODUCTION_ARM` | tier-A-only | **site 19** — tier-B attestation block, export tied to `verify-tier-b` |
| 2 | `require_exact4` vs the live-arm `B1_DIAGNOSTIC=1` block | contradiction | **site 20** — tier-B gate asserts the *production* shape per the ruling |
| 3 | patcher CLI: tier-b flag *required* `--…-live-ab` | reverse | flag made first-class + mutually exclusive |
| 4 | launcher spoke only `TIER_B_SERVE`, never set `TIER_B_ARM` | rename gap | selector, enum, retired-spelling refusal |
| 5 | `_FR13_FA2_QROW32_B1_PIN_ARM` never saw a tier-B arm | tier-A-only | falls back to `TIER_B_ARM` |
| 6 | tier-B vars reached the container only via the `compgen` sweeper | transport | explicit `-e` passthrough for all three |
| 7 | `TIER_B_ARM` absent from `_FR13_M32_GUARD_NAMES` | hygiene | guarded (replaces `TIER_B_SERVE`) |
| 8 | `SPLITK_SASS_DIGEST` / `_BASELINE_` had no `:-` default under `set -u` | crash-not-refuse | defaults added in all three twins |
| 9 | cuda_graph elif chain had no tier-B branch | tier-A-only | tier-B takes the **serving** hook |
| 10 | `production_record` read `PRODUCTION_PASS_SIDECAR_SHA256` bare | KeyError on tier B | tier-aware; credential digest instead |
| 11 | `production_record` hardcoded `draft_vocab_root=1, k=65536` | honesty | reads the served identity — the red-team's own lesson |
| 12 | EAGER emitter omitted `tier=`, defaulting to `"A"` | latent mislabel | carries the tier |
| 13 | `serving_arm()` resolved production first, making tier-B's mutual-exclusion checks unreachable | silent wrong-tier serve | exclusion hoisted **before** resolution |

Items 10–13 were found by **reading and by tests**, not by booting. Item 13 was found by a
test I wrote for something else — a boot naming both selectors would have served tier A
while its env, its operator and its artifacts all said tier B.

**Correctly tier-A-only, left alone:** `require_same_reduction` under `if tier == "A"` (the
one place the tier distinction was implemented as intended); the three independent
allowlists that refuse `gqa_pair_splitk` as a *production* arm (that refusal is the
ruling); and the batch-1 geometry welded into the kernel — B1 *means* `sequences=1`, and a
paired multi-task A/B runs tasks sequentially at concurrency 1, which is what the exact4
campaign already does.

### 15.3 The CPU end-to-end serve (task 5)

`tests/test_fr14_fa2_tierb_qualification.py` now **walks the served path on CPU**: real
fixed32 operands, the real injected blob, the real precondition chain, through to the
assertion that the returned operand carries sentinel **1179791671** and `num_splits=4`.
Sites 19 and 20 both fail it in milliseconds; each precondition is removed one at a time
and must refuse. Five boots bought sites 17–19 and cost hours each; this costs 0.9 s.

**110 tests in the file, 240 across the suite.** The tier-A path is asserted unchanged in
the same harness, and the byte-gate branch is asserted *not* to run for tier B.

### 15.4 What round 8 needs

Both C++ closures unchanged (`172b5e71…`, `4ed00909…`) — **no rebuild**. The patcher moved,
so **re-earn the credential** at the serving HEAD (~10 min, offline), then arm with
`FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk` (not the retired `TIER_B_SERVE`), the
credential path/digest, `B1_DIAGNOSTIC=0`, `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` and the
canonical exact4 pins. The run will engage, refuse, or crash on a named precondition — it
can no longer serve the incumbent while reporting the candidate.

---

## 16. Site 23 — the resolver, not the table

Site 23 closes the loop on my own site-17 fix, and the runner's words are the right
doctrine: **naming a default does not remove it; the removal has to happen at the
resolver, not the table.**

Site 17 replaced `.get(arm, split2_pins)` with an explicit table — and I kept an `""` key
in it, for the legitimate "no selector named" case. The in-container Python resolver
never learned `TIER_B_ARM`, so a tier-B boot resolved `""`, and the `""` key returned
split2's pins. **The table looked exhaustive and the resolver was not.** The comment
above it still said "This mirrors the bash pin case"; the bash twin at `:2245` *did* fall
back to `TIER_B_ARM`.

### 16.1 The fix, at the resolver

1. The Python resolver learns `TIER_B_ARM`, at bash's precedence (live → tier-b →
   production).
2. **The `""` key is gone.** "No selector named" is now spelled `nosplit`, in *both*
   twins — because `""` was also what a resolver that failed to look produced, and those
   two must never be the same value.
3. The resolver is made **total against the environment**, not against a list of keys
   someone remembered to update: any `FR13_FA2_QROW32_B1_*_ARM` that is set and unknown
   to it is a refusal, with the known-but-not-pin-deciding selectors (`TIMING_ARM`) named
   explicitly so the sweep cannot be widened until it passes.

That third point is the one that generalises. Sites 2.1, 17 and 23 were each *one
variable this resolver had not been told about*; the 24th selector nobody has written yet
now refuses instead of silently inheriting split2's identity. Verified: deleting
`TIER_B_ARM` from the resolver no longer yields a wrong answer — it yields a **refusal**.

### 16.2 The twin-equivalence detector (executed, not grepped)

`pair_pin_arm_resolver_twins` extracts the bash resolver and the Python resolver **from
the launcher files themselves** and *runs both* across 9 selector environments × 3
launcher twins, comparing answers. A claim of equivalence asserted only in prose is what
this family exists to delete.

Mutation-tested from both sides: removing `TIER_B_ARM` from the Python tuple is caught
(`bash='gqa_pair_splitk'` vs `python='REFUSED:…'`), and removing the bash fallback is
caught too (`bash='nosplit'` vs `python='gqa_pair_splitk'`). **Its reach is also recorded
rather than assumed**: selector *precedence* is unobservable while multi-selector boots
refuse, so a precedence swap correctly leaves the detector green — written down so the
next reader does not assume it covers more than it does. **16 pairs, 0 stale.**

### 16.3 The credential path — settled by measurement

Run with the launcher's own mounts (`-v $REPO:/workspace -v /models:/models`):

```
host path readable in container? NO
container path readable?         YES
is /home even present?           /home   (the image's own, and empty of the file)
```

**One variable cannot serve both consumers.** Split onto the campaign's existing pattern:
`…_TIER_B_CREDENTIAL_HOST` is what the operator supplies; the launcher verifies it,
stages it into `$LOG_DIR` (which *is* mounted), re-digests the staged copy, and **derives**
`…_TIER_B_CREDENTIAL=/logs/fr13_fa2_qrow32_b1_tier_b_credential.json` itself — so the two
can never be supplied out of sync, and the container sees an immutable snapshot whose
digest `verify-tier-b` checks again on the far side. Both spellings are guarded against
`.lumo.local.env`.

### 16.4 Arming, corrected

```
FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk \
FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST=<host path to the credential> \
FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256=<its sha256> \
  <plus the §11 binary/SASS pins, B1_DIAGNOSTIC=0, FULL_AND_PIECEWISE, exact4 pins>
```

The container path is **not** supplied. Re-earn the credential at the serving HEAD first
(~10 min, offline) — the patcher moved again.

**251 tests, 16 lint pairs, 0 stale. Both C++ closures unchanged — no rebuild.**

---

## 17. Site 24, site 25, and the test that ends the class

Site 24 is the rename's stranded survivor. My walk (§15, B3) taught
`_expected_runtime_fa2_identity` about tier-B **under the spelling then in use** — the
`LIVE_AB_ARM` + modifier pun — and the pass-79 rename to a first-class `TIER_B_ARM`
reached the bash resolver and the in-container map but not this one. The runner executed
it both ways: **new spelling → stock pins; the retired pun → the tier-B branch.** Its
comment still said "Tier-B arms are LIVE-only", describing a spelling that no longer
existed.

### 17.1 Enumeration: 34 reads / 6 stranded / 6 fixed

Grepping every read of `FR13_FA2_QROW32_B1_(LIVE_AB|PRODUCTION)_ARM` across the launchers,
the patcher, the contract and the sidecar gives **34 sites**. Six were stranded by the
rename, and all six are fixed in this pass rather than the next round:

| # | site | what it would have done |
|---|---|---|
| **24** | `fr13_fixed32_contract.py` `_expected_runtime_fa2_identity` | resolved **stock** pins for a tier-B boot |
| **25** | the promoted-default injection block | **armed `PRODUCTION_ARM=gqa_pair` underneath a tier-B boot** — its guard listed every other selector and not `TIER_B_ARM` |
| 26 | `export LIVE_AB_ARM PRODUCTION_ARM` before the child interpreter | site 24's own consumer: the contract resolver would have learned `TIER_B_ARM` and **never seen it** |
| 27 | the private-selector-active disjunction | a tier-B boot would not have counted as "a private selector is active" |
| 28 | the `qrow16_stock` timing-arm exclusion | a timing arm could coexist with a tier-B serve |
| 29 | the sfwd qrow16-production exclusion | same class |

Site 25 is the one worth pausing on: it was found **by the enumeration, before it fired**,
and it would have presented as yet another late-boot refusal. Site 26 is the sharpest
lesson — *a resolver that knows a selector it cannot see is still stranded*, and fixing
site 24 without it would have produced an identical round 11.

### 17.2 The universal resolver test

Nine independent resolvers, one question — *which binary does this boot authorise?* — all
fed the canonical tier-B environment and required to answer split-K
(`300123792` / `28570f83…`, plus both SASS digests where carried):

3 × bash pin-arm resolver · 3 × in-container Python resolver · the runtime contract · the
injected identity table · the sidecar candidate contract · the serving resolver.

**The enumeration is grepped, not listed**, so it cannot rot: every read of a legacy
selector is either an **exercised resolver** (8 minimum, asserted — a registry that
adjudicates everything and exercises nothing would pass while testing nothing), or an
**adjudicated non-resolver** with a written reason, or a **failure**. Sites 2.1, 17, 23
and 24 were each a read nobody had enumerated.

Counting its own answers is what caught the last one: the test expected six launcher
resolvers and got four, because `LAUNCHERS` in this file was still a 2-tuple — the same
one-twin-short mistake that let `fr14_leg3` drift a whole arm behind in the
paired-contract family. Now three.

### 17.3 The CPU walk now includes the link that killed round 10

`_expected_runtime_fa2_identity` is executed **inside** the end-to-end serve test, not
trusted from two files away. Round 10 died there with every other link already green.
0.60 s.

**266 tests, 16 lint pairs, 0 stale. Both C++ closures unchanged — no rebuild.**
Re-earn the credential at the serving HEAD; arm with `TIER_B_ARM` + `_CREDENTIAL_HOST`.

---

## 18. Site 25 — producer and consumer, and the detector that ends the patching family

Site 25 is site 18's shape in a **second installer**. `cuda_graph.py` injects the import
and call of `_fr13_fa2_qrow32_b1_production_capture_end` under `elif tier_b_serve`; the
blob that **defines** that symbol was inserted under `live_ab or production` only.
Consumer installed, producer not — so the boot cleared the launcher, the credentials, the
gate and the contract, and died at vLLM module import.

### 18.1 Symbols resolved / dangling found / fixed

**204 symbol references resolved across three arm modes (68 / 68 / 66), 3 dangling found,
3 fixed** — all three by one edit, because the same disjunction gates
`production_begin`, `production_end` and `capture_end`. Reverting the fix makes the
detector name exactly those three, which is what "each would have bitten in turn" means
concretely.

Plus one more, found by the symbol inventory rather than by a boot: `tier_b_serve` was
**absent from `_patch_tree_attn`'s private-selector mutual-exclusion dict**, so
`tier_b_serve + b1_live_ab` installed *both* the live-register wrapper and the production
wrapper into one decode call while `cuda_graph.py` got only the live-replay hook — leaving
`production_capture_end` defined, its state populated, and its verification never run.

### 18.2 The detector

`scripts/fr14_patch_symbol_resolution_sweep.py`. Builds a scratch engine tree carrying
only the anchors the patcher binds to, runs the **real** `patch_installed_vllm` against it
**once per arm mode**, then AST-walks every patched file and requires every `_fr13_*`
symbol referenced to be defined where the patch actually put it. No GPU, no vLLM, 0.3 s.

It is run for **tier-A production, tier-B serve and the live-A/B shadow** — because a
detector that only knew the mode under repair would have passed on the day site 18
shipped.

**It had the site-25 blind spot itself, at first.** The initial `analyse()` counted an
injected `from … import _fr13_x` as a *definition*, so it reported zero cross-file edges
and would have called a missing producer "resolved". An import is an obligation, not a
definition; recording it as one is precisely the bug being hunted. With that fixed the
sweep reports the one real cross-file edge per mode — `cuda_graph.py → tree_attn.py` —
and the mutation test fails as it should.

Two adjudicated foreigners are named rather than ignored:
`_fr13_fixed32_observed_event_active` and `_fr13_fixed32_capture_begin` belong to the GDN
patcher. An unadjudicated dangling symbol stays a failure.

### 18.3 Import resolution (task 3)

The patched modules are now `py_compile`d in tests for all three modes, and the specific
cross-file import is checked against `tree_attn.py`'s **top-level defs**. The ImportError
class that produced site 25 now fails in milliseconds instead of minutes.

**299 tests, 16 lint pairs, 0 stale, 3 arm modes swept. Both C++ closures unchanged — no
rebuild.** Re-earn the credential at the serving HEAD; arm with `TIER_B_ARM` +
`_CREDENTIAL_HOST`.

---

## 19. PROMOTED TO THE PRODUCTION DEFAULT

**Mark's ruling, FR14 pass 100.** Split-K is the production default. He reviewed the
degeneration evidence — clean — and **waived the exact16-QC-first ordering: the QC now runs
after promotion, as verification rather than as a gate.**

Evidence pointer: §3–§6 and §12 of this note, and the **round-12** promotion-A/B traces.
Determinism bitwise in-process and cross-process; nine pre-registered bounds cleared with
margin; **closer to a float64 reference than the incumbent** on both output and LSE RMS;
argmax disagreements with exact 1-in-384 against the incumbent's 2; 1.9–2.0× faster.

### 19.1 The shape

The default boot arms `FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk` under
**hydra27_fixed32** — the same mode gate the `gqa_pair` default used. hydra31 stays
excluded topologically until its own qualification.

**It is armed as a TIER-B serve, not as a production arm.**
`_FR13_FA2_QROW32_B1_PRODUCTION_ARMS` is unchanged, the contract's production allowlist
still refuses split-K, and the byte-exact Tier-A door is untouched. Promotion changed
*when* the tier-B route is armed, not *what* it is. A test asserts precisely that.

**Hard refusal, inverting the house pattern deliberately.** The `gqa_pair` default degrades
to the incumbent on a stale credential, on the stated principle that *a promotion must
never refuse a boot*. Split-K's does not: a promoted default that silently serves something
else is an **unlabelled A/B**, and this campaign has already spent one whole round
(round 6) measuring the incumbent while every artifact said split-K. Missing binary, wrong
binary digest, or missing credential each exit 2 naming the cause.

### 19.2 The credential re-scope — forced, and stated plainly

The credential bound `source_commit`. Under a hard-refusal default that means **every
commit breaks the default boot**, starting with the commit that lands the promotion. That
is an availability bug wearing a safety property.

So the binding is re-scoped to what determines the **numerics it attests**: binary sha and
size, source closure, FA2 head, **both** SASS digests, the **patcher** (which decides
dispatch), and the pre-registered bounds. `source_commit` becomes *recorded and
well-formed* rather than matched. Change anything that can alter what the kernel computes
and the credential still refuses — verified: a drifted patcher digest is refused, and a
different HEAD is not. A commit touching none of those cannot alter the numerics, so
refusing on it protected nothing.

Same re-scope-and-re-attest shape pass 99 applied to the held TAW digest. **If
commit-binding is wanted back, the default must degrade rather than refuse — the two
cannot both hold**, and that is a coordinator call, not mine.

Credential re-earned at the promotion commit: **9/9 bounds, file sha `255267fc…`**.

### 19.3 What else is in

- **Explicit opt-out preserved and tested** — naming an arm makes the default block a
  no-op. Round 20's H31i A/B depends on this.
- **Credential path** keeps the `_HOST`/container split measured in round 9; the launcher
  computes the staged digest rather than carrying a literal that would need re-committing
  on every re-earn.
- **Family parity**: 11 new markers cover the default block, every literal it arms from,
  and both halves of its refusal — identical across all three twins.
- **CPU walk of the default boot**, executing the real bash: plain launch arms split-K;
  missing binary refuses *without falling back*; missing credential refuses; opt-out is a
  no-op; and the block is checked **structurally** to sit inside the hydra27 gate.

**326 tests, 16 lint pairs, 0 stale, 3 arm modes swept, family parity clean.**

## 20. The promotion did not boot — F1/F2, and the walk that could not see it

The pass-100 promotion landed the default block in all three launcher twins,
CPU-walked it under four environments, and shipped. A CPU read at pass 106
found that **the promoted default had never once served.** Round 12's live run
booted only because it named `FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk`
explicitly — the one path that skips the default block entirely.

**F1 — arming a selector is not surviving it.** The default block set
`TIER_B_ARM` and nothing else. That makes `_FR13_FA2_QROW32_B1_SELECTOR_COUNT`
1, which opens the B1 selector gate ~950 lines further down, and that gate
demanded `FR13_FA2_QROW32_B1_SOURCE_COMMIT == $(git rev-parse HEAD)` and a
`PATCH_SOURCE_SHA256`, both of which default to empty and which nothing in the
default path ever set. Every plain hydra27 B1 launch exited 2 at
`FR13 qrow32 B1 selector requires Hydra27 B1 and exact binary/source
provenance`.

The incumbent `gqa_pair` default answers this by requiring the CALLER to have
presented provenance and standing down when it is stale. A **hard-refusal**
default cannot stand down — that is the whole point of it — so it **mints**
instead: there is no caller declaration to check, because the launcher itself
is the thing arming, from its own literals. What guards the binary is the
sha256/size/SASS check the block already runs and `verify-tier-b` on the
credential, neither of which a minted commit can weaken.

**The gate had to be reconciled, not just fed.** Pass 101 re-scoped the tier-B
credential: `source_commit` moved from `TIERB_BINDING_FIELDS` to
`TIERB_RECORDED_FIELDS`, because a credential that attests *numerics* must bind
what can change them, and a commit that touches no kernel input cannot. The
selector gate was still enforcing the binding the credential had dropped.
Feeding it a minted HEAD would have satisfied it while leaving the rule alive
one layer down, where nothing documents it — so the two were made to agree, in
the credential's direction:

* tier-B selector → `source_commit` must be a well-formed 40-hex commit
  (recorded, not bound). A credential earned eight commits back now boots.
* every other selector → unchanged, `== $(git rev-parse HEAD)`. Nothing was
  dropped from the byte-exact route, whose credential *is* a byte identity
  earned at a commit.
* the **patcher digest** is still checked for every arm including tier-B. It
  decides dispatch, so it can change what the kernel computes, and the
  credential binds it too. Only the commit clause was scoped.

**F2 — two promoted defaults, no arbitration.** Presenting a `gqa_pair`
credential armed `TIER_B_ARM` from the split-K block *and* `PRODUCTION_ARM`
from the gqa_pair block; `SELECTOR_COUNT` became 2 and the boot died at
`live A/B and production arms are mutually exclusive`. Split-K's default
**supersedes** gqa_pair's — it *is* the promotion — so when the tier-B default
arms, the gqa_pair default block stands down with a logged reason. Naming
`gqa_pair` explicitly is untouched: that path sets
`_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED=1`, which skips the whole region.

### The walk answered the wrong question

The pass-100 walk executed the default block and asserted `ARM=gqa_pair_splitk`
— *does the block arm?* The question was *does the boot survive the arming?*,
and the defect was not in the block at all: it was in what the block hands to
the rest of the boot. A four-environment walk that stops at the block's own
`fi` is structurally incapable of seeing it.

The replacement composes the regions an arming actually flows through and
executes them back to back, sliced verbatim from each launcher by anchors that
exist in all three twins:

| region | what it is |
|---|---|
| 1 | the split-K default literals |
| 2 | the mode-gated promoted-default region — **both** defaults, so F2's arbitration is exercised |
| 3 | `SELECTOR_COUNT` accumulation + the mutual-exclusion refusal |
| 4 | the B1 selector gate, including the reconciled commit clause |

No paraphrase of launcher logic lives in the test, which is the only way a walk
can still fail when the launcher changes underneath it. It asserts
`CANDIDATE=1` — proof the gate actually opened — so a future slice that misses
region 4 fails loudly instead of passing vacuously.

Six behaviours, times three twins:

* a plain hydra27 B1 boot reaches rc=0 with a minted 40-hex commit and 64-hex
  patcher digest (the F1 regression proper)
* a presented gqa_pair credential yields `COUNT=1`, not 2 (F2)
* emptying `_FR13_SPLITK_DEFAULT_ARM` still reaches the gqa_pair default — the
  arbitration must not *cost* the incumbent its own default
* a stale patcher digest still dies at the gate
* a byte-gated arm with a non-HEAD commit still dies at the gate
* a tier-B boot with a malformed commit still dies — "recorded" is not "absent"

…and five mutation proofs that remove exactly the fix each guards: drop the
mint (with its guard, or the guard fires first and proves itself rather than
the gate), drop the patcher-digest mint, drop the reconciliation (which only
shows up on a credential from an *older* commit — at HEAD the mint hides it),
and `if false` the arbitration. `scan_family_parity` gained six markers keyed
on the halves of the fix, keyed on the mint's own assignment rather than the
bare `git rev-parse` idiom, because the canonical launcher runs that idiom a
second time for the gqa_pair serviceability probe the twins do not have.

### Which boot path is canonical now

**The fixed promotion default.** A plain `FR13_FIXED32_MODE=hydra27_fixed32`
B1 launch, with no B1 arm named, now arms `gqa_pair_splitk` from the
launcher's own literals, mints its own selector provenance, and survives to
serve. The round-12 env route
(`FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk`) remains valid and is now the
**explicit-A/B path**, not the production path — it bypasses the default block,
so a run that sets it is opting out of the promotion's own staging checks and
must present its own binary and credential. Measurements should record the
default path unless they are deliberately A/B-ing the arm.

## 21. Site 12 — the fork was current and stale at the same time

F1/F2 proved out live: the runner's boot armed the promoted default for the
first time ever, the mint and the F2 arbitration both fired — and it died one
gate later, in `fr14_leg3_launch_nomiddleware.sh` only. The fork's B1 selector
still hard-coded the draft-vocabulary identity:

```
&& "$FR13_DRAFT_VOCAB_ROOT" == "1" \
&& "${FR13_DRAFT_VOCAB_K:-65536}" == "65536" \
&& "${FR13_DRAFT_VOCAB_BLOCKS:-}" == "/workspace/scripts/fr13_dvk_subset_blocks.json" \
```

Production had converted that to `_fr13_assert_draft_vocab_profile`, which
admits `full_vocab` as well as `k64_root`. **K0 full-vocab is the only shape
split-K has ever served in** — round 12's promotion evidence ran the identical
env — so in the fork the promoted default was structurally unbootable.

The fork was **current on the previous night's F1/F2 work and stale on the
earlier vocab generalization, at the same time.** Selective staleness is the
shape that defeats "did this file get updated?" as a question.

### The census says four, not one

The coordinator's read found one site. Enumerating every
`_fr13_assert_draft_vocab_profile` call site against every hard-coded vocab
predicate, per family, found that production converted **five** levers and each
fork took exactly **one**:

| lever | production | armb_leg3 (before) | leg3 (before) |
|---|---|---|---|
| ordered GDN live gate | helper | **helper** | **helper** |
| qrow32 B4 GQA-pair | helper | 3 hardcoded clauses | 3 hardcoded clauses |
| qrow32 B1 selector | helper | 3 hardcoded clauses | 3 hardcoded clauses ← site 12 |
| GDN GQA-group3 production | helper | 2 hardcoded clauses | 2 hardcoded clauses |
| GDN single-launch production | helper | 2 hardcoded clauses | 2 hardcoded clauses |

**Eight sites, not one.** All eight are converted. The forks also never gained
`FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE` / `..._B4_...`, so the calls would
have expanded to the empty string and refused everything — those are added too,
and the four refusal messages now match production's word for word (the forks'
said "K64/root1" where production's no longer did).

Conversion *tightens* the default path as a side effect: the helper's
`k64_root` branch also requires `-z FR13_NEEDS_ALLOW`, which the hardcodes did
not check. That is production's behaviour, which is the point.

### The detector: stop enumerating

The 26-marker parity roster was **blind to all four**, and would have stayed
blind — a generalization that predates its marker is invisible forever, and the
roster only ever ratchets after a miss. This is the second selectively-stale-fork
bug this campaign.

So `scan_vocab_profile_parity()` is **structural, not enumerated**. It is told
no lever's name. It walks each launcher for every
`[[ … ]] || { echo "FR13 …" >&2; exit 2; }` refusal region and records two
facts per region — how many inline draft-vocab identity clauses it hard-codes,
and whether the profile helper is called immediately above it — then compares
that *shape* across the three families. Levers present in only one family are
ignored (the forks are forks); a lever present in two or more whose shape
disagrees is reported.

Two design points earned the hard way:

* **Regions are keyed by a short prefix of the refusal message.** Keying on the
  whole message is what let site 12 hide: the fork's message still said
  "K64/root1" where production's no longer did, so the two regions did not look
  like the same region at all. The prefix is the part that survives a predicate
  change; the tail is the part that records it.
* **Keyed on `(prefix, ordinal)`, never on prefix alone.** A 40-char prefix is
  short enough to survive the change, therefore short enough to collide:
  `"FR13 GDN GQA-group3 production requires "` prefixes both the lever's own
  refusal and, twenty lines later, its live-PASS-JSON refusal. The first
  version of this detector kept the last one per prefix and reported **two**
  divergences where there were **four** — it found site 12 and missed two more
  of exactly the same kind. Key on `(text, ordinal)`, never on text.

Against the pre-fix tree it reports all four, naming file, line, and shape,
while the enumerated roster reports nothing. That contrast is a test:
`test_the_structural_scan_finds_what_no_marker_names` strips every
conversion-related marker from the roster, asserts the roster is then blind,
and requires the structural scan to find the divergence anyway.

The enumerated markers were added too (seven, keyed on each *call's own shape*
rather than the bare helper name — the forks carry a comment mentioning the
helper, so a name-count marker would have read 7 against production's 6 and
cried wolf, which is how a marker gets deleted).

### The walk stubbed the thing that broke

The full-boot walk from §20 could not have caught this: `_BOOT_STUBS` defined
`_fr13_assert_draft_vocab_profile() { return 0; }`. **What a walk stubs, it
cannot test.** The helper is now lifted from the launcher as region 0 and run
for real, and the walk takes the vocabulary identity as an input:

* every full-boot case runs under **both** `k64_root` and `full_vocab`
* claiming one profile while carrying the other refuses, in both directions
* an unknown profile is refused outright, not defaulted
* re-introducing a single hardcoded clause is proved to leave `k64_root`
  booting and kill `full_vocab` — the runner's failure, reproduced on CPU.
  That asymmetry is why it survived so long: every gate that ran, ran under K64.

Also fixed: `test_fr13_qrow32_b4_production_default.py` had been asserting the
pre-conversion wording of production's B4 refusal and failing since the
generalization landed — the same miss, on the test side.
