# Adjudication: #48475 vs #50021 vs #51508 (GDN/KDA zero-accept spec rows)

Refs read: `origin/main` **80447d2765** (2026-09-17); heads **#48475 f70b0ffe66**, **#50021 9a198c0f84**, **#51508 54b69f5dd5**. Read-only on GitHub; nothing posted.

## 1. The defect at main

`fused_sigmoid_gating.py:106` / `fused_recurrent.py:106` load `i_t = num_accepted_tokens[i_n] - 1`
unbounded, then `state_idx = ssm_state_indices[i_n, i_t]` (`:110`); `state_idx <= 0 → return`
(`:114`) is the *only* skip. The write-back at `:157-166` (`fused_recurrent.py:154-163`) stores into
`ssm_state_indices[i_n, i_t]` for **every** `i_t` in `0..T-1`, so past `:114` the recurrent state
advances regardless of which slot was read.

**A zero is producible on current main.** `vllm/v1/spec_decode/utils.py:284-288` sets
`valid_count = 0` for a discarded request (`:299,312-317` likewise for an all-`-1` row); `:843-845`
writes it straight into `num_accepted_tokens` for rows with `prev_positions >= 0 & prev_drafts > 0`
(`gpu_model_runner.py:2175-2182`). Padding rows are safe (`gdn_attn.py:473` fills 1) and the
warmup-zero @xiatwhu raised on #48475 is already closed (`gpu_model_runner.py:2144,2149-2150`).

**Caveat, the crux of ZJY0516's objection:** `discard_request_mask` is
`optimistic_seq_lens < num_tokens` (`gpu_model_runner.py:2111-2113`), i.e. a mid-chunked-prefill row,
which normally carries **no drafts**, so `participating` is False and the 0 never lands. I could not
establish statically that a row is ever both discarded *and* draft-bearing. The only field
measurement in the threads (@brasrox, #48475, 2026-08-17: `min=1`, 301 rows) found no zero, and he
retracted his earlier report. **No PR presents a production trace of a 0 arriving.** All three harden
a code-supported but field-unconfirmed path.

Also live at main: `mamba_utils.py:389` (`offset = -1` → `state[b, -1:]`, a *valid* Python slice,
short copy, silent) and `:412` (`block_ids[cur - 1]` → wraps). `mamba_ssm.py:334-335` is already clamped.

## 2. Model-free discriminating probe (GB10)

Extracted the `fused_sigmoid_gating` kernel from all four refs as standalone modules (the only vllm
import, `from vllm.triton_utils import tl, triton`, rewritten to plain triton) and ran them in one
process. 2 seqs × 2 tokens, 4 state columns, unique per-block sentinel states, row 0 = stale row with
**live** block ids `[5,6,7,8]`, row 1 live `[9,10,11,12]`. The index tensor is a view into a flat
buffer with a known live id (7) planted at offset −1, which removes the allocation-layout dependence
that would otherwise make main nondeterministic under GB10 unified memory.

`/home/mark/shared/tmp-scratch/B_probe/{probe_zero_accept.py,probe_wrongstate.py}`, run under
`flock /home/mark/shared/exp54928/gpu.lock`. Case **Z** = raw `num_accepted_tokens = [0, 2]`, row
still live (what main/#48475/#50021 see); **N** = row nulled to `NULL_BLOCK_ID`, count `[1, 2]` (what
#51508's builder emits); **C** = healthy control `[2, 2]`.

| variant | C row0 | **Z row0 (stale)** | N row0 | row1 (live), all cases |
|---|---|---|---|---|
| main | wrote [5,6] | **wrote [5,6]** | — | wrote [9,10] |
| #48475 | wrote [5,6] | **wrote [5,6]** | — | wrote [9,10] |
| #50021 | wrote [5,6] | **untouched** | — | wrote [9,10] |
| #51508 | wrote [5,6] | **wrote [5,6]** | — | wrote [9,10] |

Second probe (block-5 content at count 0 vs count 1): **main DIFFERENT (max|Δ| 1.58)** — it resumed
from the planted block 7, an out-of-row read, then overwrote block 5. #48475 and #51508 identical
(correct slot-0 resume, then advance). #50021 differs because it leaves block 5 untouched.

## 3. What this settles

- **"The clamp-only PRs leave the state advanced for a discarded step":** true for **#48475**
  (`tl.maximum(...,0)` → slot 0 is a live block → past `:114` → advances). **False for #50021**, which
  is *not* a clamp: it masks the load with `other=0`, so a 0 count lands in the existing
  `state_idx <= 0` path — state untouched, output deterministically zeroed. Same shape in its KDA
  kernel, and in `causal_conv1d.py` an explicit `(num_accepted < 1) | (num_accepted > seqlen)`
  fail-closed return.
- **#51508's own kernel edits are clamp-only and behave exactly like #48475** (row Z). Its protection
  lives entirely at the builder: `gdn_attn.py` nulls the stale row so `:114` fires. Case N shows main
  already skips a nulled row — the builder fill does the work; the kernel clamps are defence-in-depth.
- So #50021 and #51508 both prevent the advance; they differ in layer, not outcome.

## 4. Is #51508 correct and complete on current main?

**Correct.** `git merge origin/main` into `pr-51508` conflicts in **one test file only**
(`tests/models/kimi_k3/test_kda_metadata.py`); all seven source hunks auto-merge and land in the
right place. Its in-place `masked_fill_` is safe: both GDN branches build
`spec_state_indices_tensor` via boolean-mask indexing (`gdn_attn.py:306-308, :327-329`), which
copies; the KDA override correctly uses out-of-place `masked_fill`. The fill sits at `:365`, before
the cudagraph buffer copies (`:437-441, :469-473`), so eager and cudagraph both inherit it. Nulling
the whole row also covers conv: the GDN layers pass `conv_state_indices=spec_state_indices_tensor[:, 0]`
(`qwen_gdn_linear_attn.py:1360`, `olmo:359`, `kimi_gdn:480`) and `causal_conv1d.py:834-839` returns on
a null id before reaching the offset at `:874`. Builder coverage is complete:
`KimiK3ROCmKDAMetadataBuilder` (`amd/kda_metadata.py:88`) overrides only `_build_chunk_metadata` and
glm5next consumes `GDNAttentionMetadata` (`common/kda.py:467`), so both inherit the GDN fix.

**Gaps, all minor.** (a) glm5next's KDA kernels (`nvidia/.../fused_recurrent.py:138`, `amd/...:116`)
lack the defence-in-depth clamp, though the builder covers them. (b) A skipped row's **output** is
left uninitialised (pre-existing main behaviour); #50021 zeroes it. (c) The `mamba_utils.py` clamps
#51508 adds are themselves clamp-to-slot-0, the semantics it argues against elsewhere — still better
than the `-1` at main, but inconsistent. (d) Open, unsettled: a nulled row produces no state update
and garbage output; whether the engine also discards that row's current-step output needs an e2e run.

## 5. Ownership

No maintainer is adjudicating. ZJY0516 (MEMBER) commented once on #48475 (2026-07-13) and has not
returned; maxpla3 @-mentioned @tdoublep and @ZJY0516 on 2026-08-19 with no reply; #51508 has zero
human reviews. #40738 (tdoublep), #55504, #56531 are a **different** defect (the ngram slot-0 /
speculative-path convention mismatch, counts > 1), though #56531 independently closed the
warmup-zero producer.

## 6. Tests (all at each PR's own head, GB10, under the GPU lock)

| suite | result |
|---|---|
| #51508 `test_fused_sigmoid_gating_delta_rule.py -k "null_state_row or zero_accepted"` | 4 passed |
| #51508 `test_gdn_metadata_builder.py` + `test_kda_metadata.py` | 42 passed (37s) |
| #50021 `-k invalid_accepted` (sigmoid gating) + `test_mamba_utils.py -k "invalid_block_table"` | 6 passed |
| #50021 `causal_conv1d -k invalid_accepted`, `mamba_ssm -k accepted_count`, `kda -k invalid_accepted` | 6 passed |
| #48475 `-k zero_accepted` | 2 passed |

#48475 asserts count 0 ≡ count 1 bitwise, which *pins the state advance in place*. #51508's
`test_spec_decoding_null_state_row_leaves_state_untouched` passes an already-nulled row, which main
also satisfies (case N); its discriminating coverage is the builder test, not the kernel test.

Tree restored to `fix/modelopt-lmhead-quant-gaps`, clean. `B_done.marker` touched.

---

## DRAFT review comment for #51508 (NOT POSTED, 176 words)

> Traced the three PRs against main @80447d2765 and ran a model-free probe of the
> `fused_sigmoid_gating` kernel from all four refs in one process (2 seqs, 4 state columns, sentinel
> states, stale row holding live block ids).
>
> With a raw `num_accepted_tokens == 0` reaching the kernel: main reads out of row and advances the
> stale row's state; **#48475** clamps to slot 0 and still advances it; **#51508**'s kernel hunk is
> also a clamp and also advances it — its protection is entirely the builder-level `NULL_BLOCK_ID`
> fill, which main's existing `state_idx <= 0` guard then honours. **#50021** is not a clamp: it
> masks the load with `other=0`, so the row falls into that same guard and the state is left
> untouched at the kernel layer.
>
> So #50021 and #51508 both prevent the advance, at different layers; #48475 does not.
>
> Merging main into 54b69f5 conflicts only in `tests/models/kimi_k3/test_kda_metadata.py`; all source
> hunks apply. New tests pass at each head (51508: 4+42; 50021: 12; 48475: 2).
>
> One thing none of the three shows: a production trace of a `0` actually arriving.

## DRAFT 2-line notes for the other two threads (NOT POSTED)

**#48475** — Probe at main@80447d2765: with count 0 and a live index row, the clamp reads slot 0
correctly but the write-back loop still advances the stale row's state (blocks written, same as
unpatched main); your test's count-0 ≡ count-1 assertion pins that behaviour rather than excluding
it. #50021 and #51508 both avoid the advance, by masking the load and by nulling the row at the
builder respectively.

**#50021** — Worth noting on the neighbouring threads: your FLA/KDA change is repeatedly described
there as "clamp-only", but the masked load with `other=0` routes a zero count into the existing
`state_idx <= 0` return, so the recurrent state is *not* advanced — verified directly against main,
#48475 and #51508 in one process. The one thing #51508 covers that this does not is the CPU
align-mode `-1` indexing at `mamba_utils.py:389,412`.
