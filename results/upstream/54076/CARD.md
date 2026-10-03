# Experiment CARD — item E / issue #54076 boot check
Written 2026-09-21 BEFORE any GPU work. Do not edit after the first launch; corrections go in DEVIATIONS.md.

## Question
jschmied (#54076, 2026-09-20): "If you can name a configuration in which the divergence *is* reachable on
current main, I will run the original cell on it."

Divergence := at the moment `Scheduler._mamba_block_aligned_split` runs,
`self.cache_config.block_size` (the value that function reads) is STRICTLY SMALLER than the
`block_size` of the model's `MambaSpec` KV-cache group.

## What is executed vs. what is read
- EXECUTED: the rig's stock wheel, vLLM **0.28.0** (aarch64), torch 2.13, GB10, TP=1.
  Nothing under `.venv` is modified.
- READ STATICALLY ONLY: `origin/main` @ **382970ee6ca490aeaaaf4e32c53695b581ff61ba**
  (fetched 2026-09-21, tip commit "[Kimi-K3][AMD] Return KDA and MLA projection outputs directly (#50592)").
  No main build is executed. Any claim about main is a source claim, not a measurement.

## The decisive values (and nothing else)
1. `cache_config.block_size` AFTER `EngineCore._initialize_kv_caches` — i.e. after
   `vllm_config.cache_config.block_size = min(...)` over KV-cache groups
   (0.28.0 `vllm/v1/engine/core.py:322-324`; main `:344-353`). This is the value
   `_mamba_block_aligned_split` reads (0.28.0 `vllm/v1/core/sched/scheduler.py:392`;
   main `:431` — identical statement in both).
2. The `block_size` of the `MambaSpec` group in the scheduler's `kv_cache_config.kv_cache_groups`.
3. Every other group's `block_size` + spec class (so the `min` is auditable).
VERDICT RULE: "reachable" iff (1) < (2) for a configuration that boots. Anything else — a startup
INFO line, a config repr printed before KV-cache init, an argv value — is configuration, NOT the answer.
Explicitly NOT claimed by this experiment: what the scheduler then does at runtime with that geometry
(no request is traced through `_mamba_block_aligned_split`; no correctness claim about #54076's bug).

## Static prediction (made before booting)
Chain (0.28.0 file:line, then main file:line at the SHA above):
- (a) min over groups: core.py:322 / core.py:349. main additionally filters to
      `g.kv_cache_spec.prefix_cacheable` groups (core.py:344-348); 0.28.0 has no such filter and no
      `prefix_cacheable` property at all.
- (b) platform floor + mamba padding: `platforms/interface.py` `_align_hybrid_block_size`
      0.28.0 :767 (floor :908-914, `mamba_block_size = block_size` in align mode :918,
      mamba page padded to attn page :931-940) / main :764 (floor :917, align :925, padding :938).
      Same logic in both.
- (c) drafter's own attention group: the draft layers live in the same
      `compilation_config.static_forward_context`, so their spec is built with
      `cache_config.block_size` like every other attention layer
      (`v1/worker/gpu_model_runner.py:7899` get_kv_cache_spec). Per-group block sizes are then only
      ever RAISED, never lowered, by `unify_kv_cache_spec_page_size`
      (0.28.0 kv_cache_utils.py:1036-1097: `new_block_size = block_size * ratio`).
      The drafter's block size is reported at DEBUG by
      `v1/spec_decode/llm_base_proposer.py:1801` "Using block size %d for drafting layers".
- (d) `MambaSpec.block_size = cache_config.mamba_block_size`
      (`model_executor/layers/mamba/abstract.py:64-70`; main :66-72), which in
      `--mamba-cache-mode align` is exactly the post-floor `cache_config.block_size` (see (b)).
- (e) THE ONE PATH THAT LOWERS A GROUP'S BLOCK SIZE: hidden-state cache layers.
      `kv_cache_utils.py` pulls `HiddenStateCacheSpec` layers out of page unification and re-adds them
      with `new_bs = _largest_divisor_at_most(gcd(group block sizes), common_page // per_token)`
      — 0.28.0 :1781-1821 (log line :1812 "Using block size %d for hidden-state cache layer %s"),
      main :2305-2353 (log :2344). `HiddenStateCacheSpec` (main kv_cache_interface.py:722) does NOT
      override `prefix_cacheable`, so it inherits `True` (main :163) and is therefore INSIDE main's
      filtered `min`. These layers exist when the speculative method is `extract_hidden_states`.

Predictions:
- P1 (Arm 1, DFlash2 drafter): NO divergence. Target Qwen3.8-27B full-attn per-token page
  = 2*num_kv_heads(4)*head_dim(256)*2B = 4096 B; DFlash2 drafter per-token page
  = 2*8*128*2B = 4096 B. Equal per-token pages ⇒ `unify_kv_cache_spec_page_size` is a no-op ⇒ the
  drafter's (sliding-window) group carries the SAME block size as the target attention group, and the
  `min` cannot fall below `MambaSpec.block_size`. A separate drafter with its own attention KV group
  is, by itself, NOT sufficient. (This is the part of jschmied's request we were asked to test.)
- P2 (Arm 2, control): identical geometry to Arm 1.
- P3 (Arm 3, `--block-size 816` explicit): the floor at interface.py:908 raises it back to the
  hybrid-required value; geometry unchanged from Arm 1.
- P4 (Arm 4, the predicted-positive, ADDED — see DEVIATIONS.md): with
  `--speculative-config '{"method":"extract_hidden_states", ...,
  "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[...]}}}'` plus the
  ExampleHiddenStatesConnector, a `HiddenStateCacheSpec` group is created whose block size is a
  proper divisor of the attention/mamba block size (path (e)) ⇒
  `cache_config.block_size` < `MambaSpec.block_size`. Arithmetic prediction with N=1 extracted layer,
  hidden_size 5120, bf16: per_token = 1*5120*2 = 10240 B; if the mamba floor lands the attention block
  at B with common_page = B*4096, then hidden block = largest divisor of B that is <= B*4096/10240
  = B/2.5, i.e. at most B/3. Predicted STRICT inequality.

## Arms (one server at a time)
All arms: `Qwen/Qwen3.8-27B-FP8` rev 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, TP=1,
`--max-model-len 4096`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.50`,
`--enable-prefix-caching`, `--mamba-cache-mode align`, `VLLM_LOGGING_LEVEL=DEBUG`,
HF_HOME=exp54928/hf, HF_HUB_OFFLINE=1, PATH prepended with .venv/bin (ninja for FlashInfer JIT),
FLASHINFER_WORKSPACE_BASE=exp54928/flashinfer_ws.
- Arm 1 `spec_dflash2`: + `--speculative-config {"model":"incoai/Qwen3.8-27B-DFlash2",
  "revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","num_speculative_tokens":7}`
- Arm 2 `nospec`: no `--speculative-config`.
- Arm 3 `dflash2_bs816`: Arm 1 + `--block-size 816`.
- Arm 4 `extract_hidden`: `--speculative-config` with method `extract_hidden_states` +
  `--kv-transfer-config` ExampleHiddenStatesConnector (kv_producer, shared_storage_path under the
  work dir). See DEVIATIONS.md for why this arm was added.

## Instrumentation (disclosed; additive, logging-only)
Stock vLLM never logs the POST-`min` `cache_config.block_size`, and `EngineCore.get_kv_cache_group_metadata`
(0.28.0 core.py:419) is dead code with no caller and no HTTP route. Therefore a `sitecustomize.py`
on PYTHONPATH wraps `vllm.v1.core.sched.scheduler.Scheduler.__init__`: it calls the ORIGINAL
`__init__` unchanged and then prints `self.cache_config.block_size`, `self.cache_config.mamba_block_size`,
`id(self.cache_config)`, `self.block_size`, `self.hash_block_size`,
`self.need_mamba_block_aligned_split`, and for every group in `self.kv_cache_config.kv_cache_groups`
its spec class and `block_size`. It sets nothing and returns nothing. The printed
`self.cache_config` is the same object `_mamba_block_aligned_split` reads (`self.cache_config.block_size`,
scheduler.py:392) — the hook prints `id()` of it so that identity is on the record.
The exact file is archived in the evidence tarball.

## Rules held
GitHub read-only, nothing posted. GPU only under `flock /home/mark/shared/exp54928/gpu.lock`.
`sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before each launch (GB10 unified memory:
the CUDA free-memory query counts clean page cache). Servers killed only with `pkill -f '[v]llm serve'`.
Stock wheel; no file under `.venv` is touched. Budget <= 1.5 h GPU, <= 4 h total.
