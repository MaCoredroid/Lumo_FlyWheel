# FR13 — WHY the tree is slower at B=4 SWE (MEASURED, workflow wacoxe6i2, adversarial-verify holds=True)

**Basis = `decode_seconds` (raw vLLM counter `request_decode_time_seconds_sum`), NOT wall** (tree emitted only 1856 vs 2048 tokens — early stops — so wall 2.074x understates). **NOT TPS÷accept** — the banned 1.412x/1.29x hand-roll form is explicitly quarantined and never used.

## The headline decomposition (both factors are raw measured counters)
```
tree/native decode_seconds = 298.497 / 127.755 = 2.336x  slower
  = (MORE FORWARDS)            x (MORE TIME PER FORWARD)
  = (spec_drafts 620/433=1.432x) x (decode_s/spec_drafts 0.4814/0.2950 s = 1.632x)
  check: 1.432 x 1.632 = 2.336   (exact algebraic identity)
log-share:  ~42% accept gap   /   ~58% per-forward tax
```
- `spec_drafts` = forward count (justified: spec_draft_tokens/spec_drafts = exactly 9.000 = num_spec_tokens, both arms). Scraped before/after by `scripts/fr12_deliverable_swe4_probe.py`.
- accept/event corroborated: native **3.794** vs tree **2.024**; tree reject modal at step-0 (accepts spine-only).
- **UNKNOWN (not measured):** decode_seconds is a per-request SUM under B=4 concurrency (overlapping windows), so 1.632x is a valid RELATIVE ratio but NOT single-stream forward latency; no `request_metrics.jsonl`/steptrace in the run dir → per-kernel attribution of the 1.632x is **not** measured.

## Removable-vs-fundamental (per-forward 1.632x tax)
Forward is **weight-bandwidth-bound**: 27 GB fp8 / 273 GB/s = **98.9 ms floor** (GB10). A change moves wall-time only if it cuts HBM bytes or wasted launches.

- **REMOVABLE — the dominant adder: GDN tree-scan per-node 9× state r+w amplification** (+8.95% per-req / **+35.79% at B=4** of the 27 GB stream; LABELED CPU-cost-model estimate `/tmp/gdn_cost_verify.py`). Removable ONLY by **kernel rewrite = WY one-pass / accept-only state commit** — CPU-proven lossless (fp32 **4.19e-9, 0 argmax flips**). **#1 do-now lever.**
- num_warps=8 spill fix **already committed** (`fr10_gdn_tree_kernel.py:582`, a586ac84) — h_cache 128KB register overrun gone. LIVE gates still open: ptxas spill-bytes==0 + bit-exact at N_PAD=16.
- TREE_ATTN / FA2-fork: ~time-neutral (attention ~0.1% of a bandwidth-bound forward, hidden behind the weight stream); the -inf-bias is an in-register add, no extra pass/MMA.
- Eager-launch tax: ≤2.7% of the floor (LABELED estimate); CUDA-graph capture removes it.
- **FUNDAMENTAL** (can't remove without losing the tree): attention node-count (36 vs 24 pos/forward, but hidden behind the floor); no-copy FA2 ~1-bf16-ULP grouping floor (0.0039 on 2 of ~983k elem — **under** the E5 self-noise floor ~0.059, so lossless + free).

## ⚠ CORRECTION (user, superset logic) — the "capped/thin" framing below is WRONG
The tree is a **STRICT SUPERSET** of native's MTP drafter: its spine ≡ native's depth-5 chain, plus branches. A correct verify therefore accepts **≥ native (3.794) on the spine alone**, + branch bonus at temp>0. So tree accept/event **2.024 < native 3.794 is IMPOSSIBLE for a correct verify** — the 2.024 and the "0 net branch accepts" are **CONTAMINATION dropping the spine itself**, NOT a drafter ceiling. The break-even table below computes from 2.024, which is the *contaminated* number → meaningless as a "drafter floor." **Corrected verdict:** fixing the B=4 carrier pulls accept/event from 2.024 → ≥3.794 (spine recovered) + branch bonus, so **lossless and faster CONVERGE** — they are not separate phases. The speed is currently *masked by the bug*, not capped by the drafter/spine. (The 76%-saturated argument only bounds the *branch* upside above native, not the recovery to native.)

## [superseded framing] Break-even arithmetic — valid only IF 2.024 were the true yield (it is not)
Governing identity: `TPS = (accept/event + 1) / time_per_forward`. Break-even accept/event the tree must hit (LABELED structural estimates):
| tree forward vs native | break-even accept/event |
|---|---|
| 1.00x (all tax removed) | ≥ 3.794 |
| 1.10x (GDN-amp removed) | ≥ 4.27 (+13%) |
| 1.13x (clean structural est) | ≥ 4.42 (+16.5%) |
| **1.632x (measured, contaminated)** | **≥ 6.82 — IMPOSSIBLE on depth-5 (ceiling 5.0)** |

Two hard caps:
1. Native MTP-5 = depth-5 chain, **hard ceiling 5.0**; native 3.794 = **75.9% saturated** → lossless tree's upside is a THIN slice.
2. **In every measured run, branches added 0 net accepts** (pos5-8 survival = 0). The caterpillar drafter has two known bugs (propose_tree builds 2 parallel chains not a caterpillar; deep-spine GDN-flat) → branches may be starved/garbage. **This is the user's superset logic: same drafter top-1+top-2 = superset, so 0 branch accepts = OUR drafter bug, not a ceiling.**

## Levers ranked
1. **Remove kernel tax (WY one-pass + accept-only state commit)** — ONLY lever touching the binding per-forward metric; removes 9×→1× GDN-state amp losslessly; targets the ~58% share. WY is **PARKED** (failed *literal-0.0*, never re-measured at the *within-floor* bar the user moved to). The clean re-measure is the unrun arbiter.
2. **Deeper drafter (depth 6/7/8)** — raises the accept/event ceiling (only way past 5.0); DRAFTER change, orthogonal to verify kernel; the genuine route to acc/event > native.
3. **Fix the caterpillar drafter** (recover the 1.432x more-forwards) — caps at ~42% of the gap, gated on the two topology bugs.
4. More branches / wider tree — least headroom vs a saturated spine; net-negative until the drafter is fixed.
5. ~~Multi-spine~~ — CLOSED_NON_SHIP/lossy (directive), not a lossless lever.

## The unrun ARBITER (recommendation)
Clean re-measurement: **FR10_METRICS off, all FR12/FR13 capture-break hooks compiled out, WY kernel, B=4 CUDA-captured, vs E5** — replaces the contaminated 1.632x with the true structural forward floor (est ~1.10-1.15x, UNMEASURED). That number decides win/loss. **Bring the clean pass/fail to the user before declaring win/loss.** If the clean forward floor proves a hard >1.13x AND branches stay at 0 net accepts on a depth-5 spine, lossless-fast is dead unless the drafter is deepened.
