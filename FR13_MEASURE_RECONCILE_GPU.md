# FR13 canonical-measure GPU reconciliation (2026-06-15)

GPU run of `scripts/fr13_measure_orchestrate.sh` (drives `scripts/fr13_measure.py`) in the canonical
regime, reconciling the MEASURED OFF-mode speed against the banked historic numbers. Serialized boots
on PORT 9950 (prelaunch `recover_host_memory` + assert MemAvailable>=95GiB + docker-empty per boot +
teardown after). Substrate: main HEAD `572b623e`. int-view never atol; pathspec commits only.

## VERDICT — the regime reproduces the historic B=1 SPEED; accept is trajectory-bound (autotune floor)

- **s/fwd REPRODUCES the banked numbers** (the speed regime is correct):
  - native E5: measured **0.21732** s/fwd vs banked **0.218160** -> **-0.38%** (within trajectory-length noise).
  - native E5 B=4: **0.21641** s/fwd -> **-0.42% vs the B=1 number** = s/fwd is ~B-invariant for the
    bandwidth-bound GB10 decode, as the model predicts.
  - cat9: measured **0.22636** s/fwd vs banked **0.2248** -> **+0.69%** (reproduces); tok/draft=9.0 == len(TREE).

`reconcile.json` (canonical MAX_NUM_SEQS=1 records):

| arm | B | measured s/fwd | banked s/fwd | s/fwd delta | measured accept | banked accept | reproduces_banked_accept | fp |
|---|---:|---:|---:|---:|---:|---:|:--:|---|
| native_e5 | 1 | 0.21732 | 0.21816 | -0.38% | 1.5888 | 3.16129 | False (forked) | c25a3c3d |
| cat9 | 1 | 0.22636 | 0.2248 | +0.69% | 2.6693 | 3.18 | False (forked) | 9b265d3e |
- **accept/event does NOT reproduce 3.161 byte-for-byte** because each fresh boot's GB10 fp8/bf16
  realization lands on a DIFFERENT greedy trajectory at an early near-tie. THIS IS THE
  AUTOTUNE/REALIZATION FLOOR (`feedback_no_cross_boot_byte_gate`), NOT a regime bug and NOT an engine
  regression. The infra SURFACES it (`reproduces_banked_accept=False` + a differing
  `served_stream_fingerprint`) instead of silently reporting the forked number as "native accept".

## The token-6 cross-boot greedy fork (the root, definitively pinned)

The original "1.70 vs 3.161" bug was hypothesized to be raw-vs-tokenized prompt framing. That is WRONG
in the mechanism. The real cause is a **same-prefill cross-boot greedy trajectory fork at served token
index 6**, the GB10 autotune floor:

| source | first 6 served ids (shared prefill) | token@6 | trajectory | drafts | accept/event |
|---|---|---:|---|---:|---:|
| banked 3.161 (`fr13_b1_current_gate`) | `[271,248068,271,248069,271,40]` = "</think>\n\nI" | `3172` ("I'll") | coherent | 124 | **3.161290** |
| this boot (MAX_NUM_SEQS=1) | identical | `1144` ("I need") | degenerate `<think>...None` loop | 197 | **1.589** |

- The prefill BYTES are identical (both share `[271,248068,271,248069,271,40]`).
- Each boot is **within-boot deterministic** (rep1==rep2 byte-identical: fp `c25a3c3d` both reps,
  accept 1.589 both reps). So the fork is CROSS-boot, not within-boot nondeterminism.
- `MAX_NUM_SEQS=1` (the banked gold bind) was tested and STILL forks (accept 1.589) — so
  MAX_NUM_SEQS=4-vs-1 is one amplifier but not the sole discriminator; the underlying token-6 near-tie
  argmax is resolved differently by each boot's kernel autotune. This is the documented floor.

Binding lesson (bug-class #12 "non-like-for-like trajectories"): `accept/event` is TRAJECTORY-BOUND and
is NOT apple-to-apple across two free-running boots. `s/fwd` (per-event decode time, ~length-invariant)
IS trajectory-robust, which is exactly why it reproduces while accept forks.

## Regime fix applied to the orchestrator

`fr13_measure_orchestrate.sh` previously inherited the launcher default `MAX_NUM_SEQS=4` for the B=1
native boot (diagnosed amplifier #3). Fixed: `boot_native`/`boot_tree` now pin **MAX_NUM_SEQS=1** (the
gold bind that produced 3.161/3.18); the B=4 co-residency smoke is a SEPARATE boot
(`native-b4`/`tree-b4`, MAX_NUM_SEQS=4) whose accept is the genuinely co-residency-degraded number,
labelled `batch_size=4`. The two must NOT share one boot (mixing perturbed the trajectory).

## B=4 co-residency smoke (sane)

- native E5 B=4 (MAX_NUM_SEQS=4, client batch of 4): s/fwd **0.21641** -> **-0.42% vs the B=1 0.21732**
  = ~B-invariant (bandwidth-bound, as expected). accept/event **1.4954** (labelled B=4, trajectory fp
  `a20d3c9d`). The B=4 measurement is sane: s/fwd is B-invariant, accept is captured per-event with its
  own fingerprint + the co-residency note. (This particular boot's B=1 accept was itself forked low, so
  the B1->B4 accept delta is small here; the apparatus is what is being smoke-tested, and it measures
  s/fwd B-invariant + accept B-dependent correctly.)

## INSTRUMENT ON/OFF separation (measured)

- native E5 diag-residue: OFF s/fwd 0.217199, ON s/fwd 0.220307 -> **instrument tax 1.43%** (within the
  <=2.5% expectation per 46e89f22, MEASURED, `tax_within_expectation=true`).
- cat9 diag-residue: OFF s/fwd 0.226358, ON s/fwd 0.233503 -> **instrument tax 3.16%**
  (`tax_within_expectation=false`) -- the tree arm's capture-q DtoH is heavier; the infra MEASURED and
  FLAGGED it rather than assuming <=2.5%. The ON capture is used ONLY for this tax; the OFF number is the
  deployment speed.

## temp-0.6 distributional capture (the NEW q machinery, non-vacuous)

Both arms `capture-q` (ON, temp 0.6 / top_p 0.95 / top-k 20): q recorded over the **FULL** served stream
(128/128 positions x all 4 prompts on each arm), per-position truncated tail mass recorded (native
median ~0.003-0.005; cat9 median ~4e-5..5e-4), within-boot determinism `[T,T,T,T]` on both, engagement
tok/draft=5.0 (native) / 9.0 (cat9). This is the piece nothing recorded before. The recurrent-oracle p +
per-position TV(softmax(q/0.6),softmax(p/0.6)) reduce is the binding lossless gate (separate in-process
oracle boot, FR12_NO_SPECULATIVE_CONFIG=1 FLASH_ATTN single-token roll, RECURRENT_PATH_ENGAGED asserted).
