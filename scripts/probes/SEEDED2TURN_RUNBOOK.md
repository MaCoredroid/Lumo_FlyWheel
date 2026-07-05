# SEEDED2TURN — FR13 tree+cache DECODE-losslessness 2-turn seeded probe

Instrument of record for campaign §30. Localizes the tree+cache give-up's
losslessness violation with **on-distribution same-seed paired streams** (NOT
teacher-forcing) and answers, from ONE probe:

1. **COLD carrier** (§28) — where cache-config decode numerics first flip a real
   sampled TOKEN on turn-1 (cold), bracketed by the cross-boot autotune floor.
2. **RESTORE carrier** (refold's domain) — whether the hit-turn restore reproduces
   the same boot's miss-turn stream (confound-FREE, same-boot).
3. **REFOLD value** — whether refold measurably improves the hit turn, gated by the
   mechanical `redirect_used>0` liveness check (else the arm is vacuous, §27).

Files (all in this dir):
- `seeded2turn_run.sh` — boots the 4 arms serially, drives the 2-turn seeded
  requests, asserts the hit fired, collects streams + counters + (pass-2) dumps.
- `seeded2turn_reduce.py` — first-divergent-token per arm/turn + floor bracket +
  restore/refold verdict + (pass-2) per-decode-step state localization.
- `apply_decode_capture_patch.py` — wires the DEFAULT-OFF per-decode-step GDN-state
  capture into the patcher + launcher (pass-2 only; reversible).
- `decode_gdn_capture_patch_spec.py` — the canonical capture block + rationale.

GPU: **serial, single GB10 GPU.** The runner refuses to boot on top of any running
container. Recover host memory between arms (built in).

---

## The two passes

### PASS-1 (default) — token streams, NO patch, byte-identical served path
```
bash seeded2turn_run.sh
.venv/bin/python seeded2turn_reduce.py \
    --run-root output/fr13_seeded2turn --tokenizer /models/qwen3.6-27b-fp8 \
    --out output/fr13_seeded2turn/reduce.json
```
This is the PRIMARY readout. It applies NO capture patch and needs no launcher
edit. It yields readouts (1)(2)(3) at token granularity and hard-gates hit-firing +
refold liveness. **Run this first.** Learn the first-divergent decode step `D` from
readout (2)'s `min` (the earliest miss-vs-hit token index maps to the decode step
that produced it; the token index IS a decode-step index in resend mode because
turn-2 decodes one token per step from position 0).

### PASS-2 (only if pass-1 shows an above-floor / restore fork) — per-step STATE

> **The decode-capture instrument is ALREADY in HEAD** (a concurrent worker landed
> it in commits `482bb382` / `dccb221c`, §31): identical env-var names, launcher
> `-e` forwarding, default-OFF + eager-only guards, and payload schema
> (`layer_prefix`,`decode_step`,`last_recurrent_state`,`conv_state_rows`,
> `core_out_spec`,`coresident_rows`,`spec_state_indices`) that this reducer keys on.
> So **pass-2 needs NO patch application** — just `CAPTURE=1`.
> `apply_decode_capture_patch.py` is a presence-aware verifier/fallback: it reports
> "already present in tree (HEAD) — no-op" and refuses to duplicate the launcher
> lines; it only writes edits on a checkout where the instrument is genuinely absent.

```
# 1. (optional) verify the instrument is present (no-op against HEAD)
python3 apply_decode_capture_patch.py            # dry-run: reports present/absent, touches nothing

# 2. re-run the cache arm (B) + the ref (A) with a WINDOW around D (<= 8 steps wide)
CAPTURE=1 STEP_LO=$((D-2)) STEP_HI=$D CAPTURE_LIMIT=4 \
  ARMS="cat8_nocache cat8_cache" bash seeded2turn_run.sh

# 3. localize (streamed, memory-safe)
.venv/bin/python seeded2turn_reduce.py --run-root output/fr13_seeded2turn \
  --self-demote --memmax 12G \
  --state-cache-dir  output/fr13_seeded2turn/cat8_cache/logs/decode_gdn \
  --state-oracle-dir output/fr13_seeded2turn/cat8_nocache/logs/decode_gdn
```
(No revert step needed against HEAD — the instrument is committed and default-OFF.
`--revert` only removes THIS script's sentinel regions, so it is a safe no-op here.)

---

## Arms (VERBATIM route_probe.sh presets; all cat8 TREE, ENFORCE_EAGER=1, temp 0.6)

| arm | role | key env |
|---|---|---|
| `cat8_nocache`      | **A** lossless reference | `FR13_APC_CONFIG_ONLY=1` (cache OFF) |
| `cat8_nocache_b`    | **A'** floor self-check  | identical to A, 2nd boot (brackets cross-boot floor) |
| `cat8_cache`        | **B** give-up config     | `FR13_APC_EXACT_SEED=1`, refold OFF |
| `cat8_cache_refold` | **C** refold             | B + `FR13_APC_BLOCK_REFOLD=1 REFOLD_TO_SNAPSHOT=1` |

`cat8_nocache` (A) MUST be the first arm (default order) — conversation mode freezes
its turn-1 completion as the shared assistant turn.

## The 2-turn construction

Per arm, per seed `k`:
1. `POST /reset_prefix_cache` ONCE → turn-1 is a genuine cold prefill.
2. **TURN-1**: the pinned route-probe payload (system+user, ~12k tokens, 15 tools),
   `seed=k, temperature=0.6, max_tokens=1024, logprobs=true`. Capture the response.
3. **TURN-2** (NO reset): re-POST at the SAME `seed=k`. Two modes:
   - `TURN2_MODE=resend` (**default**): re-send turn-1's exact prompt. The WHOLE
     prompt hits → maximal restore, and turn-2 input is byte-identical across arms
     AND identical to turn-1 → the same-boot miss-vs-hit gate is exact.
   - `TURN2_MODE=conversation`: `payload.messages + FROZEN assistant turn (+tool
     results) + FIXED user turn-2`. The assistant turn is frozen from ref arm A
     (seed `$FREEZE_SEED`) so turn-2 input stays byte-identical across arms; the
     shared ~12k system+user prefix still hits and fires restore.

**Why resend is the default:** it is exactly §30 ("turn-2 re-sends turn-1 and HITS
the cached prefix"), guarantees the hit, and makes readout (2) a pure recompute-vs-
restore byte-identity gate with ZERO cross-boot confound. Conversation mode is the
more deployment-faithful variant when a genuine new user turn is wanted.

## Asserting the hit fired (mandatory — else the probe is vacuous)

Per turn the runner snapshots `/metrics` and records the per-turn delta of
`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` to `hits.jsonl`.
Hard gates (cache arms B, C):
- **turn-2 hit seeds > 0** — else `FAIL: turn-2 prefix cache NEVER hit … VACUOUS`.
- `es_seed_applied > 0` (from the periodic `FR13_OBS_SUMMARY` eng line, atexit
  `fr13_obs_final.json` fallback) — WARN if 0.
- arm C only: **`redirect_used > 0`** — else `FAIL: refold NEVER executed … the
  refold A/B is VACUOUS (§27)`. This is the standing risk: `redirect_used` has been
  0 on every refold run to date; if it stays 0, arm C == arm B and "refold no help"
  would be a lie (refold never ran). The reducer independently re-fails on this.

Config-only arms (A, A') must report turn-2 hits == 0 (sanity: caching truly off).

## Env-threading verification (the H3 lesson — asserted end-to-end)

Every consequential var is asserted in `container_env.txt` or the LIVE vLLM argv
(`/proc/1/cmdline`) before any sampling:
- `ENFORCE_EAGER=1` → launcher `:673` host-eval → `--enforce-eager` in argv **[asserted]**.
- `FR13_ENABLE_APC/CONFIG_ONLY/EXACT_SEED`,`MAMBA_BLOCK_SIZE/DTYPE` → `APC_FLAGS`
  → `enable_prefix_caching=<exp>` in boot log + `--enable-prefix-caching`/
  `--enable-chunked-prefill` in argv **[asserted]**.
- `TREE` → `SPEC_CONFIG` (`speculative_token_tree`) in `container_env.txt` **[asserted]**.
- `FR13_APC_BLOCK_REFOLD/REFOLD_TO_SNAPSHOT=1` (arm C) in `container_env.txt` **[asserted]**.
- pass-2 `FR13_DECODE_GDN_CAPTURE` in `container_env.txt` **[asserted]** — fails loud
  if the capture patch was not applied.

## Memory policy (the 2026-07-05 unified-mem OOM lesson)

The pass-1 token reduce is pure-JSON (light). The pass-2 STATE reduce streams the
`.pt` dumps ONE file at a time (`map_location=cpu`, freed immediately) and never
holds two payloads. Run it `--self-demote` (re-execs under
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`) so a reduce blow-up
kills the scope, not the session. The runner's pass-2 window (`STEP_HI-STEP_LO<=8`,
`CAPTURE_LIMIT` small) bounds the dump to ~few steps × 48 layers.

---

## The decode-capture patch — EXACTLY what it touches (ALREADY IN HEAD; default-OFF)

**Status: already committed to HEAD** (`482bb382`/`dccb221c`). The applier no-ops.
The description below is what it WOULD touch on a checkout that lacks it (and matches
what HEAD already contains). `apply_decode_capture_patch.py --apply` edits two
tracked files:

1. **`scripts/fr10_phase4_patch_vllm_tree_gdn.py`** — inserts ONE post-replacement
   stanza after `text = text.replace(prefill_conv_needle, prefill_conv_replacement,
   1)`. At that point `text` holds the emitted tree-decode scan; the stanza does a
   single guarded `text.replace(<unique tree-scan FR12_SUBKERNEL_CAPTURE anchor>,
   <capture block> + <anchor>)`, placing the block right after the spec scan
   produced `(core_attn_out_spec, last_recurrent_state)` with `conv_state`,
   `spec_state_indices_tensor`, `num_accepted_tokens`, `attn_metadata`,
   `self.prefix` in scope, BEFORE `ssm_state` write-back. Guarded by
   `"FR13_DECODE_GDN_CAPTURE" not in text` (idempotent). Verified: patched patcher
   `py_compile`s; the block `py_compile`s in a 12-space scope; the runtime anchor is
   **unique** (a plain 12-space `if os.environ.get("FR12_SUBKERNEL_CAPTURE"):` is a
   substring of the 16/20-space guards elsewhere → the applier anchors on the
   multi-line `tree_scan_active` block to avoid hitting the wrong site).
2. **`scripts/fr13_launch_forked_fa2_tree_server.sh`** — adds 5 `-e
   FR13_DECODE_GDN_CAPTURE*` lines after the `FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX`
   `-e` line.

**Default-OFF invariant:** with `FR13_DECODE_GDN_CAPTURE` unset the inserted block
is a single `os.environ.get()` returning `None` → body skipped → ZERO tensors
touched → byte-identical served path. It is ALSO eager-only
(`torch.cuda.is_current_stream_capturing()` guard) so it can never perturb a
graph-mode run. Verify flags-OFF byte-identity by diffing a flags-OFF served stream
vs a pre-patch stream (a pass-1 run at CAPTURE=0 with the patch applied must produce
byte-identical `turn*_k.json` to a pre-patch pass-1). **Revert with `--revert`.**

---

## Expected readout (given §22/§29)

- **(1) COLD** — arm B turn-1 forks from ref A at/near the first `<answer>` token
  (route delegate 16/16 → ~3-4/16), EARLIER than the A-vs-A' floor ⇒ confirms the
  §28 cold-decode cache-config losslessness violation as a real, above-floor signal.
  Arm C turn-1 == arm B (refold cannot fire cold).
- **(2) RESTORE** — within each cache boot, turn-1(miss) vs turn-2(hit). If restore
  is lossless these are byte-identical; a fork is a pure restore-losslessness
  failure (refold's domain).
- **(3) REFOLD** — arm C restore-fork LATER/vanished vs arm B ⇒ refold helps; equal
  ⇒ inert; `redirect_used==0` ⇒ VACUOUS (must be reported as "refold never ran", not
  "refold no help").
- **Instrument health:** every cache arm turn-2 `hit>0` + `es_seed_applied>0` (+ arm
  C `redirect_used>0`), and A' brackets a non-degenerate floor. A null (arm B turn-1
  WITHIN the floor) would overturn §28 and must be reported, not smoothed.

## Risks & the ONE thing most likely to make the first run vacuous

**MOST LIKELY VACUOUS CAUSE: arm C `redirect_used==0`.** Refold has never fired
(§27); the probe hard-FAILs arm C on this and the reducer re-flags it. This does NOT
harm readouts (1)(2) for A/A'/B — arm C is last and its abort still leaves full
`arm_meta.json` (written before the assert). If the goal is refold VALUE and C keeps
failing liveness, that IS the finding (refold irrelevant to turn-1; deploy conv-only)
— do not silently downgrade it to "refold tested, no help".

Other risks:
- **Cross-boot floor contamination** — cache-ON vs cache-OFF are irreducibly
  separate boots (`enable_prefix_caching` is engine-construction). The A' floor arm +
  eager + the same-boot readout (2) are the load-bearing mitigations. If the floor
  forks INSIDE the think channel (before `<answer>`), token localization degrades to
  route-rate granularity — lean on the route distribution, not the exact index.
  (NOTE: a shared triton-cache-root would deepen the floor, but the launcher `-e`
  list does not forward `TRITON_CACHE_DIR` and the L0c BI/FLASH_ATTN lever bypasses
  the TREE_ATTN kernel that IS the carrier — so it is unavailable here without a
  further launcher edit; A' is the mitigation instead.)
- **Hit not firing** — if the template re-renders turn-2's prefix non-identically,
  turn-2 won't hit. Resend mode makes this near-impossible (identical bytes); the
  runner hard-FAILs on turn-2 hit==0 regardless. In conversation mode a dangling
  tool_call is avoided by appending tool-result messages.
- **Tree-mode logit row ambiguity** (pass-2) — `FR13_FINAL_LOGIT_CAPTURE` dumps all
  tree candidate rows. The primary token-stream diff avoids this; treat captured
  logits as secondary and row-map via `num_accepted_tokens`/`spec_state_indices`.
- **Seed semantics** — deployment give-up runs `seed=None`; this probe forces
  `seed=k`. Correct for the lossless self-check, but it measures the config's logit
  shift under fixed randomness, not the exact deployment draw. Frame as
  "losslessness localization", not "reproduce the give-up token sequence".
