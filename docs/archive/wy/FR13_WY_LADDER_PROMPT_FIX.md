# FR13 WY live-ladder PROMPT-PIN FIX (monitor, 2026-06-08) — stop guessing, do one fresh paired run

## THE WALL (grounded): the pinned native ref `output/fr13_wy_gateA_20260608T163915Z/native/` has NO recoverable prompt
- `native/response.json` `choices[0].prompt_token_ids = None`, `token_ids = None`. There is **no `request.json`** in the dir. The prompt is **unrecoverable** from the saved artifacts.
- usage = `prompt_tokens: 5`, completion text = `"\n\n<think>\nHere's a thinking process:\n\n1. **Analyze"` (a REASONING trace).
- The "The word ready.<|im_end|>" codex chased is the **MTP DRAFT** (`fr10_mtp_draft_trace.jsonl`), NOT the prompt context — draft ≠ prompt. That is why text guesses ("Say the word ready." → input_hidden drift 0.4336; "Output the word ready.") fail: re-tokenizing a guessed text cannot reproduce the unknown 5-token input.

## THE FIX: abandon reusing 163915Z for the ladder — do ONE fresh PAIRED run
The per-layer ladder compares **tree-vs-native on IDENTICAL input** → the specific prompt is irrelevant to drift. So:
1. `docker rm -f` the current container + `recover_host_memory` (forked exit wedges ~90GB).
2. Pick a FIXED deterministic prompt (any; e.g. a short fixed string) OR pass explicit `prompt_token_ids`. **Save the `request.json`** (and the prompt_token_ids) into the run dir.
3. In ONE run capture the **native arm** (FLASH_ATTN MTP-5) AND the **tree arm** (TREE_ATTN WY) with that SAME input. Pin native ONCE here (now WITH its request) → capture-once holds for all later tree-only re-runs.
4. Per-layer ladder: input → every GDN + full_attn layer → final logits, **spine AND branch** (4 leaf-path oracles `[0,1,3]`/`[0,1,2,5]`/`[0,1,2,4,7]`/`[0,1,2,4,6,9]`, per-node argmax). flag OFF = the pre-tap baseline.
5. Then apply `docs/archive/wy/FR13_WY_OUTPUT_TAP_PATCH.md` (flag `FLA_BF16_OUTPUT_SPLIT` ON): smoke OFF-path byte-identical, then re-ladder → spine GDN output 0.0 + branch single-ULP floor + branch oracle. Bind both to FR13_LADDER_LOG.md.

## NOTES
- The E5 accept/event baseline (`output/fr10_native_mtp5_same8_20260604T210257Z`, 3.076) is SEPARATE and unaffected — this is only the per-layer drift ladder's native ref.
- This is capture-once done RIGHT: the 163915Z capture was botched (no pinned prompt); re-pin ONCE with request.json, then reuse.
- ONE GPU; recover_host_memory between arms; no concurrent --gpus; commit+push; no self-declared PASS.
