# FR13 hardware-floor series — close-out (2026-08-15)

Campaign: lossless speculative tree decoding, Qwen3.6-27B-fp8, GB10/DGX Spark (48 SMs, 273 GB/s).
War cry at series start: **1.945× floor, 40.9% left to cut.** Fail-closed provenance, Tier-A exact-byte rule, real SWE-bench-Verified traffic throughout.

## What shipped (on main, credentialed)

- **B1 production default = FA2 gqa_pair** (registry flip, line-level credential chain). Honest floor ratio **1.833×** (floor 126.775 ms incl. per-request KV re-reads).
- **B4 production default = sealed padded gqa_pair** (`32e240e15`): +27.03 ms/step sealed (SD 13.77, one-sided 95% LB +10.82), padded-B4 shadow doctrine proven at source and on hardware (poisoned shadow, zero real-byte drift), dual-gated at three successive HEADs. B4 honest floor ratio **2.574× and falling** — the honest gap to B1 is 1.40×, not the 1.55× raw ratios implied.
- **single_launch GDN fold (B4)**: legality PASS (11,088 byte comparisons), price measured directly at b=4: **8.984 ms/step** (2.14× MDE, super-linear in width). Production machinery built as gqa_group3's structural twin, registry default **0** — promotion is a one-token change gated on phase-4 sealing.
- **Token metering moved to our own instrument** (`d44f456ce`): per-request usage on the tamper-evident proxy ledger; the "unattributed traffic" mystery resolved — the SWE agent's self-reported usage under-credits its rejected compactions, discarded first turns, and 0/0 sub-agents. 4-task gates reconciled by luck (P≈0.7), not structure.
- **Copytree disk bomb defused** (`9a2559f15` train): a bare repo copy was 700 GiB apparent → 0.385 GiB; basetemp retention capped.
- **TAW batched committer (this merge)**: the 9.2 ms/step "sampler math" re-attributed to our own tree-accept walk (24 full-vocab softmax+cumsum/step on 4 of 48 SMs). Batched candidate byte-proven **zero-ULP by construction** at B∈{2,3,4} (shape-pinned normalization sums; widened gate covering post-norm rows, cumsum thresholds, accept decisions; authoritative-zero arming). **Unpriced** — the screen was cut at close-out; central expectation ~7 ms/step at width 4, above the sealed 4.20 ms MDE.

## The map of remaining room (width-4 step, honest accounting)

| where | ms/step | verdict |
|---|---|---|
| "other" bucket (83.9 total) | 33–62 reducible | TAW is the first bite (~7 expected) |
| GDN scan+delta | 63.0 vs 4.5 floor (**13.97×**, worst ratio) | single_launch takes ~9; more theoretically there |
| FA2 remaining headroom | 62.9 | **load imbalance** (cost = max(ctx) per wave), not bandwidth — needs scheduling |
| GEMM DRAM efficiency | 38.3 | known-hard |

## Open items (documented, safe by construction)

1. **Production re-gate at current main is REQUIRED before the next production serve** — the merge train staled the HEAD-bound flip credential; the launcher fail-closes until `fr13_run_b34_fa2_qrow32_gqa_pair_live_gate` re-passes at this HEAD (~1.5 GPU-h, known-clean).
2. TAW price + seal/flip: re-earn FA2+TAW credentials at this HEAD, one SC timing pass, then the 4-pass seal (operator ruling).
3. single_launch phase-4 seal (4-task shape recommended, b4-only) — doubles as its workload-matched timing verdict.
4. 13398 tool-cap stays OFF (truncated-trace validator union unbuilt).
5. Caveat on record: `[1,V]` fp32 cumsum is run-to-run non-deterministic on this hardware — the historical B1 TAW byte credential's zero-mismatch rests on a non-reproducible op (mitigated in practice by top-k masking); B=1 batching is structurally impossible and refused in four places.

## Doctrine bank (hard-won this series)

Padded-shadow operands; HEAD-bound credentials (every commit forces a re-gate — batch merges into trains); refuse-at-the-right-layer (qualification refuses, serving degrades); manifest binding is settledness, not `git status`; workload-matching, not same-pool (paired passes absorb trajectory drift); meter on your own instrument, never a third party's self-report; authoritative zero (an explicit 0 vetoes sidecars); boot-refusal probes mirror the runner, not the arm; a gate that can only see the bit-stable op is blind by construction — widen before you trust.
