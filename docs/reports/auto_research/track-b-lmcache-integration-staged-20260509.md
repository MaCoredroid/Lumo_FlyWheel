# Track B Round 2 Step 1 — LMCache integration staged

**Date:** 2026-05-09
**Status:** install + import working in the running vLLM container
(`bc4bd9c`); vLLM-side wiring + cache backend config staged here for
operator-driven next vLLM relaunch.

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
