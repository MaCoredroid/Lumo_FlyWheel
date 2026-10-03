# Q1 continuation fixtures v1: independent source/CPU review

Reviewed 2026-09-29 UTC. **Accepted as CPU fixture preparation; no concrete fixture blocker found.** This does not admit a runtime job or establish qualification coverage. The qualification denominator remains null. No model, GPU, Docker, SSH, cache action, implementation edit, or gate change was performed.

## Exact sources and artifact

Paths are relative to `v2/experiments/review-response-20260927/`.

| Artifact | SHA-256 |
|---|---|
| `tools/q1_continuation_fixtures_v1.py` | `16f85ffff80008fdd396e48c9b1e0387770b7b4ee8e0611de566fa05abf8ce9e` |
| `fullmodel/fixtures/token-fixtures.sequences-v1.json` | `cc1c07cf2bfddbfc57875b655eaea149af92f497db56ed2aff185c2f7c426fa4` |
| `tools/tests/test_q1_continuation_fixtures_v1.py` | `04010e397d20e7c2b6dca482444a557dd13109e0040d64ef2ec72a0a4ae40dcf` |
| Input scenario matrix | `df2a4c7b01a6ae09ef06c397f0bb0d2f084f40421b39e6d65f01284429173eed` |
| Input cycle0 fixtures | `607634e19235a58d2b6d73f8260dac3c49f842f9f1cba9a1789f329c56012223` |
| Pinned native geometry record | `8fb996a798d91aaade5cac94debd7996292f40968ef578ba594102a08bc49ec2` |

The generator and saved artifact agree canonically on rebuild. The native geometry source must pass the accepted `N.seal` record check, `valid=true`, the exact reviewed raw-record SHA, target-attention coverage 16, and a unanimous kernel block size of 64. Its geometry is not selected from candidate outcomes. The CLI finishes validation/build before opening the write-once output, so a failed native-record binding does not emit a fixture file.

## Verified semantics and population

- **F2:** The three declared sequences are preserved exactly: longest→root-only→off-spine; longest→shortest; and the declared eight-step frozen walk. Each adjacent pair has `z_i == root_(i+1)` at the same absolute position. The flat trace contains every materialized position once, then consumes only the final pending token separately. All **60 adjacent handoffs** across the saved population occur once. Only cycle0 permits hydration; later cycles forbid rehydration. The next tree root forward is simultaneously the preceding cycle's O2 observation, so an eventual driver must not insert a second independent forward for the same pending token.
- **F3:** All correction-pending aliases use the declared depth-four spine base; all bonus aliases use the declared depth-eleven leaf. For every prefix, the complete token and position chains equal the existing frozen cycle0 fixture exactly. These aliases add no distinct token-chain evidence and must not be counted as independent outcomes merely because they have separate scenario names.
- **F4-B3:** All **18 boundary inputs** use the longest accepted path and independently reconstructed prefix-plus-padding bytes. Padding is minimal, fewer than 64 tokens, drawn deterministically from valid non-special token values in the same prefix. Initial prefix remainders are 63, 61, and 53 for offsets −1, −3, and −11. Materialized positions lie on both sides of the next 64-token block boundary, and each concatenated-prefix SHA matches. This artifact does not claim to implement the other F4 or F5 scenarios.
- **Split isolation:** The saved records inherit the exact prefix IDs, prefix bytes/hash, calibration/evaluation labels and held-out flags. Calibration and evaluation each have three disjoint prefix IDs and raw prefix hashes; each has 24 records and 54 cycle observations. No candidate result is read to choose sequences, tokens, padding or paths.
- **Token/position checks:** Independently checked all 108 cycles against the topology and scenario matrix: 32 physical token rows plus one distinct pending token, valid token IDs outside added-token IDs, exact tree-depth positions, exact accepted-path tokens, contiguous materialized positions, matching extents and a single final pending consumption. Record IDs are unique.

Per split, preparation contains 9 continuous records, 3 correction-pending aliases, 3 bonus aliases and 9 boundary records. Overall **48 records / 108 cycle observations** are input-description counts before process/repeat expansion, not executed experiments, unique evidence counts, or an admitted qualification denominator. Both artifact launch/qualification flags remain false.

## Verification and one documentation follow-up

Independently ran `python3 -B tools/tests/test_q1_continuation_fixtures_v1.py`: all **4 CPU test methods pass**, including changed reset, position, pending token, duplicate consumption, inactive node and wrong-parent refusals. A separate direct audit of every saved record is retained in `q1-continuation-fixtures-v1-independent-audit.json`, SHA `e9eee0558d6489b2987d653ce0c6a149069408ef0be014fdd7819827a4661f75`; it also records dependency hashes.

`PROTOCOL-Q1.md` §5 still says “KV block-boundary (block size 1024, bound)”, while the older matrix describes the size as unknown U6. The new artifact correctly binds **64** from the exact accepted native attention record. Update current qualification prose to this authoritative geometry when integrating the fixture; this documentation inconsistency does not invalidate the prepared inputs.

`validate_sequence` is a coherence helper, not a complete runtime admission validator. This review independently checks the generated artifact's additional row, source, split and position fields. Future execution must authenticate the frozen fixture and enforce continuous state/owner lifecycle; the declarations alone do not prove those runtime properties. No new experiment family, workload attempt, numerical tolerance or qualification claim is introduced here.

## Addendum: allocation blocks, native kernel blocks, and candidate physical pages

**Correction to the documentation follow-up above:** the protocol's 1024 value must not simply be replaced with 64. The values describe different cache representations, and the candidate fixed32 path has a 1024-token physical paged-KV contract. V1 remains accepted as **native-64 boundary input preparation only**; its 18 F4 records do **not** cover a candidate 1024-token physical-page boundary. A direct check finds **0/18** cross 1024. This limits the earlier preparation approval; no candidate boundary qualification was established.

Exact distinction and source binding:

- The accepted native boot attestation records `cache.block_size=1024`, `mamba_block_size=1024`, and the full-attention allocation `FullAttentionSpec.block_size=1024`, while `kernel_block_sizes=[1024,1024,1024,64]`. Its SHA is `1f0630dd53e74ab3d02e7e01722552650f7fcb45db12be330f1d08d14ae4455c` at `runs/q1-native-smoke/q1-native-smoke-recovered-20260928T031726Z/q1_ref/boot_attestation.aligned_nonpacked.pA.json`. The native case's attention tensor is `(2,10912,64,4,256)`; the recorded `kernel_block_size=64` is read from `bt.block_size` and checked against tensor dimension 2 (`tools/q1_reference_hooks_v2_1.py:386–406`).
- Native `FLASH_ATTN` permits only 16/32/64 kernel blocks for a hybrid model with FP32 Mamba cache (`identity/native_source/vllm__v1__attention__backends__flash_attn.py:75–92`, SHA `c949e39a0b0b3dfa316017c422b1cb1368cd3a4a1328a1c691f5d7f44ed571e7`). The captured runner explicitly computes `num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size` and shapes attention tensors using kernel blocks (`identity/generated_source/probe-20260928T035847Z/logs/generated/gpu_model_runner.patched.py:10000–10014,10209–10217`, SHA `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79`). Thus each native 1024-token allocation block is represented as sixteen 64-token attention kernel blocks.
- Candidate `TREE_ATTN` permits `MultipleOf(16)`, which includes 1024 (`workload-plan/inspections/cpu-lumotree-source-20260929T012000Z/result/sources/v1/attention/backends/tree_attn.py:720–744`, SHA `ba7710ae7bc77fe5839ca7636b36f1a6591d4d707ef290903b5608cc967911ed`). The served fixed32 B1 FA2 path explicitly requires `params.page_block_size == 1024` and strides proportional to `2*1024*4*256` (`scripts/fr13_patch_fa2_tree_bias.py:1218–1258,1376–1409`, repo-root path, SHA `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`). Its 64-row K-compute tile is a separate quantity, not a 64-token candidate cache page.
- The retained candidate boot's effective serve settings have `--block-size 1024 --mamba-block-size 1024` and `ATTENTION_BACKEND=TREE_ATTN` (`runs/q1-candidate-stage1/q1-candidate-stage1-20260929T040755Z/q1_effective_serve_settings.txt:18,26`, SHA `2db31db873351ac891851f23be581b8f26fb7c3fa826fd2b036d0de165e430ac`). Its served credential separately records `probe.geometry.page=1024` and `block_n=64` (`fr13_fa2_qrow32_b1_tier_b_credential.json:276–289` in that run, SHA `0a4f7576db4e7e35080b0865e4cdb04fd652a29466b66cd0a8cebb1843664453`). These bind the candidate route's expected geometry; a future valid candidate record must still retain the actual tensor/block-table geometry. No candidate-64 assumption is warranted.

Proposed additive protocol wording, without changing historical configuration:

> Block-size clarification: the frozen serving allocation and Mamba alignment settings remain 1024 tokens. The accepted native causal FA2 reference represents each 1024-token allocation block as sixteen 64-token attention kernel blocks, as verified by its block tables and KV tensor shapes. The candidate fixed32 TREE/FA2 path uses a 1024-token physical paged-KV contract; its 64-row K-compute tile is not the page size. F4-B3 records must name which physical boundary they cross and bind each route's actual runtime geometry. The v1 fixtures cross native 64-token boundaries only. Candidate-boundary successor fixtures at 1024−{1,3,11} also cross native 64-token boundaries using the same logical inputs; execution and qualification remain unadmitted.

The parent will preserve v1 and materialize a separate successor using the same already-declared offsets and candidate-1024 boundaries. This does not itself expand or admit an execution denominator. This clarification used only local source/receipt reads; no runtime operation or protocol/gate edit occurred.
