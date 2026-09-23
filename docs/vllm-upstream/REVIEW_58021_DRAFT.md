# #58021 review comment (funded item G) — v1 (Codex replacement verbatim; agent draft NO-GO; independent verification of the lead trace pending; AWAITING MARK GO)
> Codex: post as ONE PR review with event COMMENT, no merge verdict. F1 (LCM back-off → checkpoint
> consumer) OMITTED: MambaManager already computes checkpoint validity with hash_block_size, so a
> coarser unit / checkpoint=None may be intentional agreement with allocation. Q1 trimmed of
> exclusivity/universal claims. Lead = Codex-traced MRV2 PIECEWISE profiling pre-stamp path (source
> trace, not run); second = reproduction-description question; CPU test attempt disclosed.

---

Static review at `c18f4fd6c9`: could Model Runner V2's PIECEWISE graph-memory profiling reach the new accessor before stamping? `gpu_worker.py:591` profiles before `core.py:368–376`. In `gpu/cudagraph_utils.py`, profiling initializes a minimal cache directly (:965–989), calls `capture_model(profile_only=True)` (:923), and builds attention metadata even for PIECEWISE graphs with `for_capture=False` (:805–835). With no speculation and a capture size greater than `max_num_seqs` (e.g. 32 tokens/16 requests), dummy query lengths exceed one; KDA classifies them as prefills despite the false `is_prefilling` flags. With the FlashKDA backend, `num_prefill_checkpoint_blocks=1`, so `kda_metadata.py:704–709` appears to reach `checkpoint.py:89` while `resolved_hash_block_size` is still unset. Am I missing an earlier stamp or guard? The UNIFORM_BATCH restriction on FULL graphs does not cover this PIECEWISE path.

For the reproduction description, which served KDA configuration produces the cited finer hash unit with `prefix_match_unit=None`? The 16/1600 fixture constructs that geometry; it does not establish that a checkpoint-capable model produces it.

No GPU execution. My CPU propagation-test attempt failed during model inspection because FlashAttention extensions were unavailable, before geometry assertions. AI assistance was used.
