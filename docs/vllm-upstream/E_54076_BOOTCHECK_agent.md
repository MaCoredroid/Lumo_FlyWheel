> **CORRECTIONS 2026-09-21 (Codex check):** see results/upstream/54076/CORRECTIONS.md — hook wording (it does monkeypatch, logging-only), stale arm-4 line numbers, 'only one difference' on main unsupported, 'dead code' overstated, bounded scope of the DFlash negative.

# Item E — #54076 boot check: is the divergent block-size geometry reachable?
Mark Ma / vLLM upstream campaign. Run 2026-09-21. GitHub read-only; **nothing was posted anywhere.**
Work dir `/home/mark/shared/exp54928/exp54076/` (CARD.md written before any GPU work, RESULT.md,
STATIC_FINDINGS.md, DEVIATIONS.md, DRAFT_COMMENT.md, `logs/`, `runs/`, evidence tarball).

---
# VERDICT

**Yes — and the reachable configuration is not the one #54076 describes.**

1. A **separate drafter with its own attention KV-cache group is NOT sufficient.** Booted with the
   DFlash2 drafter (the exact case jschmied said he had never tested), the geometry is equal:
   `cache_config.block_size = 832`, `MambaSpec.block_size = 832`. Structurally it cannot be
   otherwise: page unification only ever *raises* a group's block size.

2. **The divergence IS reachable, via hidden-state cache layers.** With
   `--speculative-config '{"method":"extract_hidden_states", ...}'` on the same hybrid target, a
   `HiddenStateCacheSpec` group is created whose block size is a proper divisor of the attention /
   mamba block size. Measured on a server that booted and answered a request:
   **`cache_config.block_size = 200` vs `MambaSpec.block_size = 800`.**

3. On `origin/main` this still applies — **argued from source, not measured.** main narrows the `min`
   to `prefix_cacheable` groups; `HiddenStateCacheSpec` inherits `prefix_cacheable = True`, so it
   remains inside the `min`.

## The decisive values, and exactly which ones decide it
`Scheduler._mamba_block_aligned_split` reads **one** number:

    block_size = self.cache_config.block_size     # 0.28.0 scheduler.py:392 | main scheduler.py:431

`self.cache_config` is `vllm_config.cache_config`, whose `block_size` was overwritten with the `min`
over KV-cache groups at `v1/engine/core.py:322` (0.28.0) / `:349` (main). The comparison that answers
#54076 is that number against the `MambaSpec` group's `block_size`
(`= cache_config.mamba_block_size`, `mamba/abstract.py:64-70`). Nothing else — not `scheduler.block_size`
(the LCM), not `hash_block_size` (the GCD), not `--block-size` on the command line.

---
# Scope and honesty limits (read before quoting any number)

- **Executed: vLLM 0.28.0 only** (the rig's stock aarch64 wheel, torch 2.13, NVIDIA GB10, TP=1, driver
  590.48.01). No file under the venv was modified.
- **`origin/main` was read, never built or run.** SHA **`382970ee6ca490aeaaaf4e32c53695b581ff61ba`**
  (fetched 2026-09-21; tip "[Kimi-K3][AMD] Return KDA and MLA projection outputs directly (#50592)").
  Every statement about main below is a source claim with a file:line, and is labelled as such.
- **These are startup/configuration numbers.** They establish what geometry the scheduler is
  *constructed with*. They do **not** establish what the scheduler then does at runtime: no request was
  traced through `_mamba_block_aligned_split`, no Mamba state was inspected, and #54076's correctness
  claim is neither confirmed nor refuted here.
- Every number below is quoted from an archived `server.log` line, cited by line number.
- **Instrumentation disclosed:** stock 0.28.0 logs no line carrying the post-`min`
  `cache_config.block_size` (the one function that would, `EngineCore.get_kv_cache_group_metadata`,
  core.py:419, is dead code with no caller and no HTTP route). An additive, logging-only
  `sitecustomize.py` on PYTHONPATH wraps `Scheduler.__init__` and `EngineCore._initialize_kv_caches`:
  each wrapper calls the original unchanged, then only reads attributes and prints. It prints
  `id(self.cache_config)` so the object it reports is on the record as the one
  `_mamba_block_aligned_split` reads. Code archived in the tarball; full rationale in DEVIATIONS.md §2.
  The platform and hidden-state block sizes quoted below come from **stock INFO lines**, not the hook.

---
# Static findings (both trees; full version in STATIC_FINDINGS.md)

| link in the chain | 0.28.0 | origin/main @382970ee6c | same? |
|---|---|---|---|
| `cache_config.block_size = min(groups)` | `v1/engine/core.py:322-324` | `v1/engine/core.py:339-353`, filtered to `g.kv_cache_spec.prefix_cacheable` | **differs** |
| platform floor `if block_size < attn_block_size` | `platforms/interface.py:909-914` | `:916-921` | same |
| align: `mamba_block_size = block_size` | `interface.py:918` | `:925` | same |
| mamba page padded to attention page | `interface.py:931-940` | `:938-945` | same |
| `MambaSpec.block_size = cache_config.mamba_block_size` | `mamba/abstract.py:64-70` | `:66-72` | same |
| `block_size = self.cache_config.block_size` in `_mamba_block_aligned_split` | `sched/scheduler.py:392` | `:431` | same (byte-identical) |
| hidden-state group block size lowered | `kv_cache_utils.py:1783-1821` (log `:1812`) | `:2305-2353` (log `:2344`) | same |

**Only one difference matters**, and it makes the divergence *harder* to reach on main, not easier:
main excludes non-`prefix_cacheable` groups from the `min` (the in-tree comment names the
GLM-5.3-Flash kpool tail). That closes the kpool route. It does **not** close the hidden-state route:
`HiddenStateCacheSpec` (main `kv_cache_interface.py:722`) declares no `prefix_cacheable` override and
inherits `True` from `KVCacheSpec` (`:163`) through `MLAAttentionSpec` → `FullAttentionSpec` →
`AttentionSpec`, none of which override it.

**Why a drafter can never do it.** All attention layers — target and draft alike — build their spec
from `cache_config.block_size` (`gpu_model_runner.py:7899`). Differing per-token pages are reconciled
by `unify_kv_cache_spec_page_size` (0.28.0 `kv_cache_utils.py:1036-1097`), which for a smaller-page
layer sets `new_block_size = block_size * ratio` — it only raises. Mamba is padded, not rescaled.
The one code path that *lowers* a group's block size is the hidden-state re-add:
`new_bs = _largest_divisor_at_most(gcd(group block sizes), common_page // per_token)`.

**#58021 (added at the coordinator's request, read-only, diff not reviewed).** PR
vllm-project/vllm#58021 touches `resolve_kv_cache_block_sizes`. On main today the min-over-groups logic
#54076 describes **still exists and has not been replaced**: `cache_config.block_size` is the **MIN**
(core.py:349, inside `_initialize_kv_caches`), while `resolve_kv_cache_block_sizes`
(`kv_cache_utils.py:731`, called from `EngineCore.__init__:164`, i.e. *after* `_initialize_kv_caches`
returns) produces `scheduler_block_size` = **LCM** and `hash_block_size` = **GCD**. The scheduler
carries both; `_mamba_block_aligned_split` reads the MIN. Caution on the "16/1600 hybrid geometry"
#58021 cites: in `tests/v1/core/test_mamba_align_chunk_split.py` (main) `ATTN_BLOCK_SIZE = 16` is the
**hash** block size and the stub sets `cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE)` —
i.e. `cache_config.block_size == 1600 == the mamba block`. That test documents
`hash_block_size < mamba block`; it is **not** an instance of the #54076 divergence.

---
# Measured results (all on 0.28.0)

Common to all four arms: `Qwen/Qwen3.8-27B-FP8` @`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, TP=1,
`--max-model-len 4096`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.50`,
`--enable-prefix-caching --mamba-cache-mode align`, `VLLM_LOGGING_LEVEL=DEBUG`, HF offline.
Every arm reached `/health`, `/v1/models` and answered one `/v1/chat/completions` with HTTP 200.

| arm | `cache_config.block_size` | `MambaSpec.block_size` | divergent | groups |
|---|---|---|---|---|
| 1 DFlash2 drafter (k=7) | 832 | 832 | **NO** | 10 Mamba + 4 FullAttn + 1 SlidingWindow, all 832 |
| 2 no speculation (control) | 784 | 784 | **NO** | 3 Mamba + 1 FullAttn, all 784 |
| 3 arm 1 + `--block-size 816` | 1632 | 1632 | **NO** | same 15 groups, all 1632 |
| 4 `extract_hidden_states` | **200** | **800** | **YES** | 3 Mamba + 1 FullAttn at 800, **1 HiddenStateCache at 200** |

### Arm 1 — the case jschmied had never tested. Not divergent.
`runs/arm1_dflash2/server.log`
- `1948: INFO [platforms/interface.py:911] Setting attention block size to 832 tokens to ensure that attention page size is >= mamba page size.`
- `3279: [[E54076]] VERDICT cache_config.block_size=832 MambaSpec.block_size=832 divergent=False`
- `3278: KV_GROUP idx=14 spec=SlidingWindowSpec block_size=832 page_size_bytes=3407872 n_layers=5 first_layer=model.layers.64.self_attn.attn`
  — that is the drafter: 5 layers, named `model.layers.64…68` (the target's own 64 layers are
  `language_model.model.layers.*`), carrying DFlash2's `sliding_window: 2048`. A genuinely separate
  attention KV group whose block size is *equal to*, not smaller than, the target's.
- Arithmetic: both per-token KV pages are 4096 B — target `2*4 kv heads*256 head_dim*2 B`, DFlash2
  `2*8*128*2 B`. Equal pages ⇒ unification is a no-op ⇒ identical block sizes
  (832 × 4096 = 3,407,872 = the `page_size_bytes` printed for all 15 groups).

### Arm 2 — control. Not divergent. Matches jschmied's own result.
`1843: Setting attention block size to 784 tokens`; `2510: VERDICT … 784 … 784 … divergent=False`.
(784 vs arm 1's 832 only because the drafter changes the mamba state shape the platform sizes against.)

### Arm 3 — `--block-size 816`. Not divergent, and the floor overrides the user in an unexpected way.
`1948: Setting attention block size to 1632 tokens`; `1949: Padding mamba page size by 99.51%`;
`3279: VERDICT … 1632 … 1632 … divergent=False`. The user's 816 is consumed as the kernel *alignment
grain* (`kernel_block_alignment_size = max(min supported kernel block, cache_config.block_size)`), so
the floor lands on the next multiple of 816 — 1632 — and pads the mamba page by 99.51 %. This
corrects CARD prediction P3, which expected the geometry to be unchanged from arm 1 (it changed; the
*relation* did not). Recorded in DEVIATIONS.md §6.

### Arm 4 — `extract_hidden_states`. **DIVERGENT.** This is the answer to jschmied.
`runs/arm4_extract/server.log`
- `1855: INFO [platforms/interface.py:911] Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.`
- `2462: INFO [v1/core/kv_cache_utils.py:1812] Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block`
- `2521: [[E54076]] POST_INIT_KV_CACHES id(cache_config)=0xfb3fe21e5400 cache_config.block_size=200 cache_config.mamba_block_size=800 n_groups=5`
- `2533: [[E54076]] SCHEDULER_CACHE_CONFIG id(cache_config)=0xfb3fe21e5400 cache_config.block_size=200 cache_config.mamba_block_size=800 … mamba_cache_mode=align … num_gpu_blocks=458`
- `2534: [[E54076]] SCHEDULER_FIELDS scheduler.block_size=800 scheduler.hash_block_size=200 need_mamba_block_aligned_split=True mamba_partial_cache_hit=True`
- `2539: [[E54076]] KV_GROUP idx=4 spec=HiddenStateCacheSpec block_size=200 page_size_bytes=3276800 n_layers=1 first_layer=cache_only_layers.64`
- `2540: [[E54076]] VERDICT cache_config.block_size=200 MambaSpec.block_size=800 divergent=True`

Same `id(cache_config)` (`0xfb3fe21e5400`) at `_initialize_kv_caches` exit and on the scheduler, so the
`200` is literally the value `_mamba_block_aligned_split` would read. Note `scheduler.block_size` is
**800** (the LCM) while `cache_config.block_size` is **200** — the two numbers the scheduler carries
disagree, which is the shape of the #54076 complaint.

Arithmetic, reproduced from the log: hidden-state per-token cost
`1 aux layer × hidden_size 5120 × 2 B = 10,240 B`; common page `800 × 4096 = 3,276,800 B`;
`3,276,800 // 10,240 = 320`; largest divisor of 800 that is ≤ 320 is **200**; wasted
`3,276,800 − 200×10,240 = 1,228,800 B = 37.50 %` — exactly the percentage vLLM prints at line 2462.
Then `min(800, 800, 800, 800, 200) = 200`.

Exact invocation (arm 4), from `runs/arm4_extract/argv.json`:

    vllm serve Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
      --tensor-parallel-size 1 --enable-prefix-caching --mamba-cache-mode align \
      --max-num-seqs 4 --max-model-len 4096 --served-model-name qwen38 \
      --gpu-memory-utilization 0.50 \
      --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1,
         "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [32]}}}' \
      --kv-transfer-config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer",
         "kv_connector_extra_config": {"shared_storage_path": "<dir>", "allow_custom_save_path": true}}'

---
# What jschmied would need to run

The same command line on his build, on any hybrid mamba model, with `--enable-prefix-caching
--mamba-cache-mode align`. The drafter model is irrelevant — drop `--speculative-config`'s DFlash
entry entirely and use the `extract_hidden_states` method above. He should see, at stock INFO:
`"Setting attention block size to <B> tokens"` and `"Using block size <b> for hidden-state cache
layer …"` with `b < B`; `cache_config.block_size` then becomes `b` while `MambaSpec.block_size` stays
`B`. On main his `resolve_kv_cache_block_sizes` also logs `"kv cache group sizes %s"` and
`"kv lcm block sizes %s"` at INFO (0.28.0 has neither), which shows the per-group sizes without any
instrumentation. The value he actually needs — the post-`min` `cache_config.block_size` — is still not
logged by stock vLLM on main either; on his build the cheapest read is the same one-line wrap of
`Scheduler.__init__`, or simply `min` of the group sizes main now prints, restricted to the
prefix-cacheable ones.

Two caveats to state if this is relayed to him: (i) everything measured here is 0.28.0, and (ii) the
geometry is configuration — whether it actually poisons a Mamba state at runtime is the thing his
"original cell" would test, and this experiment does not answer it.

---
# Evidence
`/home/mark/shared/exp54928/exp54076/evidence_54076_20260921T220949Z.tar.gz`
— CARD.md (pre-run), RESULT.md, STATIC_FINDINGS.md, DEVIATIONS.md, DRAFT_COMMENT.md (not posted),
PROVENANCE.txt, the launch scripts, `hook/sitecustomize.py` (the exact instrumentation), and per-arm
`server.log` + `argv.json` + `env.json` + `meta.json` + `smoke_resp.json` + `decisive_lines.txt`.
Wheel sha256 `817b8181f7f61b4a62dc1d5d9ab39f2bfb60a6cb86c29879a78a147b85756787`
(vllm-0.28.0-cp38-abi3-manylinux_2_28_aarch64.whl); torch 2.13.0+cu130; NVIDIA GB10, driver 590.48.01.

Budget/rules: 4 GPU boots, 21:41:56Z-22:07:29Z ≈ 26 min (limit 1.5 h). Every launch under
`flock /home/mark/shared/exp54928/gpu.lock`, each preceded by `sync; echo 3 > /proc/sys/vm/drop_caches`
("drop_caches ok" recorded in each `logs/run_<arm>.log`), `--gpu-memory-utilization 0.50`, servers
reaped with `pkill -f '[v]llm serve'`. GPU verified free at exit
(`nvidia-smi --query-compute-apps` → no rows). Nothing under `.venv` modified. Nothing posted.
