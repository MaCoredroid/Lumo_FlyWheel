# Track B Round 2 Step 1 — LMCache integration staged + BLOCKED

**Date:** 2026-05-09
**Status:** install + import working in the running vLLM container
(`bc4bd9c`); vLLM-side wiring **BLOCKED** by an upstream
incompatibility on Qwen3.5-27B's hybrid attention.

## Blocker — hybrid KV cache spec unification

After applying the staged wiring (`--kv-transfer-config` pointing
at `LMCacheConnectorV1Dynamic` + `LMCACHE_CONFIG_FILE` env +
`docker/lmcache_configs/track_b.yaml`) and bringing the container
up cleanly through every prelaunch step, vLLM's engine init fails:

```
File ".../vllm/v1/core/kv_cache_utils.py", line 1216,
    in unify_hybrid_kv_cache_specs
    raise ValueError(...)
ValueError: Hybrid KV cache manager is disabled but failed to convert
    the KV cache specs to one unified type.
```

**Root cause:** vLLM 0.19's `KVTransferConfig` path disables the
hybrid KV cache manager and tries to unify all layers' KV specs to
a single type. Qwen3.5-27B is a hybrid model — full-attention
layers and GDN linear-attention layers have different KV cache
shapes (block size, head count, dtype). They cannot be unified
without losing model correctness.

The unification is a vLLM-side requirement of the kv_transfer
contract; LMCache itself doesn't impose it. Any kv_transfer
connector (NIXL, LMCache, or a custom one) hits the same
constraint on hybrid models.

**Confirmed reproduction:** commit cd9cff4 prelaunch fires all four
patches cleanly (memory ✓, PR#39562 ✓, arctic-inference ✓, LMCache
install ✓, nixl removal ✓, parser patch ✓). The failure is
strictly at engine-init time, after model weights are loaded, when
vLLM walks the per-layer KV specs.

## Workarounds (all out of session scope)

1. **Patch vLLM to allow hybrid KV with kv_transfer**: requires
   teaching the LMCache connector path to handle multi-spec KV
   layouts. Multi-week vLLM source work.
2. **Use a non-hybrid model**: drops Qwen3.5-27B → loses every
   regime measurement we just shipped against this exact model
   (v2 Round 0 baseline is invalidated for Round 2 deltas).
3. **Wait for vLLM upstream**: hybrid + kv_transfer support is
   tracked in vLLM issues; not on the 0.19 → 0.20 roadmap as far
   as I can find.

## Where this leaves Round 2

The prefill-is-king finding (62% of round wallclock) still stands
as Round 2's largest open lever. Without LMCache, the levers are
narrower:

- vLLM's built-in `--enable-prefix-caching` is already on; this is
  the only intra-process prefix-cache mechanism we have.
- Cross-session KV reuse is blocked until either (a) we move off
  the hybrid model, or (b) vLLM hybrid + kv_transfer support
  lands.

Round 2's near-term work shifts to decode-side techniques (Steps
3-9 in the harness-coupled spec). Decode is 31% of round wallclock
on the v2 baseline; the technique stack's 2-3× decode acceleration
target translates to ~10-20% e2e wallclock improvement, not the
3-5× cumulative target the v1 spec optimistically projected
(which assumed LMCache compounded with decode-side wins).

## Stage 1 of 2: vLLM kv-transfer-config (non-applicable until blocker resolves)

## Why LMCache now (re-prioritized)

The v2 spec recalibration originally claimed tool-exec-wait was the
largest open lever. Direct measurement (commit f4b8620) found tool-
exec-wait is only 12% of round wallclock; **prefill is 58%, decode
is 29%**. Prefill is 2× decode in absolute wallclock contribution.

Levers that move prefill:
- **LMCache + cross-session KV reuse** (this work).
- vLLM prefix caching is already on (`--enable-prefix-caching`),
  but operates within a single process lifetime and within the
  request scheduler's KV pool. LMCache extends that to a persistent
  cross-session cache backed by CPU RAM (and optionally NIXL/disk
  for L2).
- Reducing per-turn prompt growth (architectural; not a Round 2
  scope change).

Round 2 Step 1 in the original plan was "install LMCache + verify."
Install is done (lmcache 0.4.4 + vLLM connector module + prelaunch
hook); the wiring step is what this stage doc enumerates.

## Stage 1 of 2: vLLM kv-transfer-config

The vLLM 0.19 launcher needs `--kv-transfer-config` pointing at the
LMCache connector module. Append to the `vllm serve` command in
`scripts/run_track_b_loop.py:_track_b_runtime_prelaunch_shell` (or
to whatever launcher the operator uses):

```
--kv-transfer-config '{"kv_connector": "LMCacheConnectorV1Dynamic", "kv_role": "kv_both", "kv_connector_module_path": "lmcache.integration.vllm.lmcache_connector_v1"}'
```

Set the env var pointing at the LMCache YAML config (Stage 2):

```
-e LMCACHE_CONFIG_FILE=/opt/lumo/lmcache/track_b.yaml
```

And mount the config into the container:

```
-v /home/mark/shared/lumoFlyWheel/docker/lmcache_configs:/opt/lumo/lmcache:ro
```

## Stage 2 of 2: LMCache YAML config (local-CPU backend)

Create `docker/lmcache_configs/track_b.yaml`:

```yaml
# Track B LMCache config -- local-CPU L1 only, no L2 (NIXL/disk)
# until we have a measurement of how often the cache exceeds 30 GiB.
chunk_size: 256
local_cpu: True
max_local_cpu_size: 30   # GiB; well below the 117 GiB unified pool ceiling
remote_url: ""           # disabled
remote_serde: ""
pipelined_backend: False
save_decode_cache: False
enable_blending: False
```

Rationale:
- `chunk_size: 256` matches vLLM's default block size; ensures the
  KV blocks LMCache stores are addressable by vLLM's request
  scheduler.
- `local_cpu: True` puts the cache in CPU RAM (which on GB10 is
  unified with GPU memory; the 30 GiB cap leaves headroom for the
  27B-fp8 model + activation buffers).
- `max_local_cpu_size: 30` is a starting estimate. Track B's
  13-task corpus prefixes total ~370K prompt tokens (per the v2
  Round 0 capture) and at fp8 KV cache that's ~6 GiB — well under
  the cap. Raise it later if we see eviction.
- `remote_url: ""` disables L2 (NIXL/Redis/disk) for now. Add later
  if the cache hit rate is improved by spilling.
- `save_decode_cache: False` keeps only prefill KV in the cache;
  decode KV is regenerated from the prefill state. This matches the
  "prefill is 58%" framing exactly.

## Verification plan after wiring

1. Confirm vLLM init line includes the LMCache connector
   (`grep "LMCacheConnectorV1Dynamic" vllm.log`).
2. Run the v2 round 0 sweep against the patched runtime with
   prefix-cache reset BEFORE each task (existing
   `--reset-prefix-cache-url`). Compare:
   - Aggregate prefill_sum_s before vs after.
   - Per-task wallclock median before vs after.
   - LMCache hit/miss counters (LMCache exposes `/lmcache/stats`).
3. Headline metric for Step 1 acceptance: prefill_sum_total_s
   reduction by >= 30% on the 13-task sample. The v2 baseline is
   1976.4s; the target is therefore <= ~1383s.
4. Ladder check: re-run Step 0d after wiring to confirm the
   forced-tool_choice patch still works alongside LMCache. They
   operate at different layers (LMCache = KV; the parser patch =
   tool-call response shaping) so no interaction expected, but
   the gate is cheap.

## Why this is staged, not applied

The running vLLM container `lumo-vllm-track-b-suffix` is the Round 1
production baseline. It carries the v2 Round 0 measurements'
runtime_config_hash and is the cleanest reference dataset until
Round 2 lands a config change. Wiring LMCache requires a vLLM
relaunch (operator-gated through `ModelServer` for the host-memory
recovery sequence); this doc enumerates the exact change so the
relaunch is one step, with a sanity-check plan attached.

The prelaunch hook in `scripts/run_track_b_loop.py` already installs
LMCache idempotently. Adding the kv-transfer-config arg + the YAML
mount is the only delta.

## Open questions for operator

1. **L2 backend**: do we need LMCache's NIXL distributed L2 across
   sessions, or is local-CPU L1 sufficient for the Track B 13-task
   sample? (Recommendation: start L1-only; add L2 if we see
   eviction.)
2. **Cache budget**: 30 GiB starting estimate. Should it be larger
   given GB10's 117 GiB unified pool? (Recommendation: tune after
   first measurement; over-allocation hurts the model footprint.)
3. **Restart timing**: Round 1 baseline is currently serving. Is
   there a maintenance window before the LMCache relaunch should
   happen, or should it be the next operator action?
