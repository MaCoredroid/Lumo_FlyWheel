# FR13 Results

Date: 2026-06-07 UTC. Branch: `main`.

## Code changes

- Tree-attn metadata builder now declares
  `AttentionCGSupport.UNIFORM_BATCH`.
- Tree-attn cached prefill/decode metadata no longer computes
  `max_query_len` / `max_seq_len` via forward-path GPU `.max().item()`.
- Triton unified attention launch meta is pinned for graph replay.
- `FR13_TREE_ATTN_EXP2_SOFTMAX=1` (default) applies the exp2/log2e +
  reverse-KV softmax path; `0` preserves base-e for ablation.

## CUDA Graph Evidence

Exp2 tree server:

- `TreeAttentionMetadataBuilder._cudagraph_support =
  AttentionCGSupport.UNIFORM_BATCH`
- Unified attention source in container:
  `exp2_count=6`, `base_e_softmax_count=0`, `log2_scale_count=2`
- vLLM log: `Capturing CUDA graphs (decode, FULL)` completed `4/4`
  and `Graph capturing finished`.
- No `AttentionCGSupport.NEVER` downgrade line was found.

Base-e tree server:

- `TreeAttentionMetadataBuilder._cudagraph_support =
  AttentionCGSupport.UNIFORM_BATCH`
- Unified attention source in container:
  `exp2_count=0`, `base_e_softmax_count=2`, `log2_scale_count=0`
- vLLM log: `Capturing CUDA graphs (decode, FULL)` completed `4/4`
  and `Graph capturing finished`.

Native E5 server:

- Plain MTP-5 config: `num_speculative_tokens=5`, no
  `speculative_token_tree`.
- Backend: `AttentionBackendEnum.FLASH_ATTN`.
- vLLM log: `Capturing CUDA graphs (decode, FULL)` completed `4/4`
  and `Graph capturing finished`.

## Large-Sample SWE-4 Artifacts

All arms used B=4, SWE-Verified-4 prompts, `samples_per_prompt=16`,
`max_tokens=64`, `temperature=0.6`, `top_p=0.95`, `seed=1313`.

Artifacts:

- `output/fr13_deliverable_20260606/tree_exp2_swe4_spp16_mt64.json`
- `output/fr13_deliverable_20260606/tree_basee_swe4_spp16_mt64.json`
- `output/fr13_deliverable_20260606/tree_basee_repeat_swe4_spp16_mt64.json`
- `output/fr13_deliverable_20260606/native_e5_swe4_spp16_mt64_a.json`
- `output/fr13_deliverable_20260606/native_e5_swe4_spp16_mt64_b.json`
- `output/fr13_deliverable_20260606/basee_vs_exp2_compare.json`
- `output/fr13_deliverable_20260606/basee_vs_basee_repeat_compare.json`
- `output/fr13_deliverable_20260606/native_e5_self_compare.json`
- `output/fr13_deliverable_20260606/tree_exp2_vs_native_e5_compare.json`

## Base-e vs Exp2 Ablation

Base-e vs exp2:

- Accept/event: `0.9179746835` vs `0.9217577706`
- Accept/event delta: `-0.0037830871`
- Warm decode TPS: `4.7170320807` vs `4.8038294799`
- Token-count TV: `0.1875`
- First-token TV: `0.0`
- Emitted-token bag TV: `0.2549555752`
- Exact token-sequence match rate: `0.0`

Base-e same-mode repeat floor:

- Accept/event: `0.9179746835` vs `0.9157527418`
- Accept/event delta: `0.0022219418`
- Warm decode TPS: `4.7170320807` vs `4.7454139317`
- Token-count TV: `0.15625`
- First-token TV: `0.0`
- Emitted-token bag TV: `0.2522653315`
- Exact token-sequence match rate: `0.0`

Interpretation: exp2 did not produce a large e2e improvement, but this run
does not prove exact no-consequence. The observed base-e-vs-exp2 movement is
close to same-mode tree sampling movement for emitted-token bag TV and
accept/event, but token-count TV is higher (`0.1875` vs `0.15625`).

## E5 vs Tree-Attn Deliverable

Native E5 self-noise:

- Accept/event: `2.6101398601` vs `2.6931311329`
- Accept/event delta: `-0.0829912728`
- Warm decode TPS: `16.4785015071` vs `16.5968583045`
- Token-count TV: `0.0`
- First-token TV: `0.0`
- Emitted-token bag TV: `0.059326171875`
- Exact token-sequence match rate: `0.5`

Tree exp2 vs native E5:

- Accept/event: `0.9217577706` vs `2.6101398601`
- Accept/event delta: `-1.6883820895`
- Warm decode TPS: `4.8038294799` vs `16.4785015071`
- Warm decode TPS delta: `-11.6746720272`
- Token-count TV: `0.25`
- First-token TV: `0.0`
- Emitted-token bag TV: `0.5578341712`
- Exact token-sequence match rate: `0.0`

Verdict: the FR13 implementation fixes the tree-attn CUDA-graph capturability
wiring, but the E5-vs-tree-attn lossless+speed deliverable does not pass on
this large-sample run. Tree-attn remains slower and lower-acceptance than E5.

## Open Items

- The preexisting FR12 GDN diagnostic capture code still attempts CPU copies
  during CUDA graph capture and emits warnings even with no capture output path
  configured. FULL graph capture completed despite this, but the capture-hooks
  path should be gated more strictly before claiming a clean final run.
- Branch-oracle per-layer parity was not re-run in this pass.
