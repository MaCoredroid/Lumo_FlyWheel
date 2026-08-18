# FR14 LANE 5 — precision levers: NVFP4 verifier `lm_head`, then fp8-KV

Agent: lane 5 (model-loading / quantization / head territory). Date 2026-08-18.
Brief: REDTEAM pass 41 — `NVFP4 lm_head (-6.7 sfwd-head + ~-2.6 cfwd)`, then
`fp8-KV (~-10)`, both unparked under Mark's condition that generation traces be
READ for degeneration at every stage that produces text.

---

## 0. Headline

**Sub-lever A is not a lever to build. It shipped on 2026-08-16 and its −6.7 ms
is already inside the pinned floor.** The four loader gaps are closed, the patch
is in the production boot flow fail-closed by default, and the NVFP4 head has
now routed correctly on four independent boots (two today). The standing
question "does the modelopt NVFP4 lm_head load?" is **closed: yes.** What this
lane adds is the evidence that was missing underneath that claim — a
logit-space characterisation against the true BF16 head, a degeneration eyeball
on real traces, and an isolated microbench of the head GEMM itself.

Three findings worth Mark's attention:

1. **The realised head-GEMM saving is 7.37 ms, slightly MORE than the 6.695 ms
   the floor credits it.** Measured in isolation on the real weights: BF16 head
   10.883 ms vs NVFP4 head 3.514 ms at M=16. The two heads do *not* run at the
   same fraction of roofline (BF16 85.6% of the 273 GB/s pin, NVFP4 74.5%), so
   the campaign's `81.6% roofline` blend is a blend — but the in-serve
   attribution's 8.200 ms lands inside this microbench's measured 7.37–8.69 ms
   range, which is an independent instrument agreeing with it.

2. **Every argmax flip the NVFP4 head causes is confined to positions where the
   BF16 head was itself nearly indifferent.** On 1,024 real captured decode
   positions: 27 flips (2.64%), and the flip rate at reference top-1 margin
   ≥ 1.0 — 792 of the 1,024 rows — is **exactly 0.0000**. The arm-A FP8-derived
   head the campaign already serves has the same shape at a third of the
   magnitude (7 flips, 0.68%, also 0.0000 above margin 1.0).

3. **Sub-lever B's ~10 ms is overpriced by roughly 7×, and this can be settled
   without writing a kernel.** LANE 4's own TreeAttn-v2 §12 measurement
   decomposes today's attention at 23k context into a **2.98 ms byte term** and
   a 10.46 ms parallelism term. fp8-KV halves the byte term and touches nothing
   else, so its ceiling is **~1.49 ms/step** — and **~0.50 ms** if split-K lands
   first, because split-K already cuts the byte term to 0.99 ms. Meanwhile the
   cost is a fork of the FA2 mainloop, which has *no* dequant stage, *no* scale
   plumbing, and a single `Element` type shared by Q/K/V/O. **STOPPED at the
   plan stage, as briefed. Recommendation: do not build.**

---

# SUB-LEVER A — NVFP4 verifier `lm_head`

## A.1 Loader-gap closure status: CLOSED, and it was closed before this lane opened

The brief anticipated four loader gaps making the modelopt NVFP4 head
unloadable. They were found, fixed, tested and GPU-proven by the arm-B constant
train two days ago. Present state, verified today rather than taken on trust:

| gap | what it was | where it is fixed |
|---|---|---|
| G1 | `qwen3_5.py` / `qwen3_5_mtp.py` build `ParallelLMHead(...)` with **no** `quant_config`, so `VocabParallelEmbedding.__init__` pins the head to `UnquantizedEmbeddingMethod` for *every* scheme | `scripts/fr14_patch_nvfp4_lmhead.py` `LMHEAD_NEEDLE` |
| G2 | ModelOpt's `get_quant_method` dispatches only on `Attention`/`LinearBase`/`FusedMoE`; `ParallelLMHead` is a `VocabParallelEmbedding` and falls to `return None` | same file, `MODELOPT_NEEDLE` |
| G3 | checkpoint keys the head as bare `"lm_head"`, runtime prefix is `"language_model.lm_head"`, and `WeightsMapper`'s `"lm_head."` (trailing dot) never matches | same file, basename-fallback branch |
| G4 | `input_scale` / `weight_scale_2` are 0-dim on disk, `PerTensorScaleParameter` allocates `[1]`, and the `VocabParallelEmbedding` loader asserts exact shape equality | same file, `VPE_NEEDLE` numel-preserving reshape |

Verification performed by this lane, today:

* `tests/test_fr14_lmhead_patch.py` + `tests/test_fr13_hardware_floor_ledger.py`
  — **18 passed** (anchor-exactly-once, fail-closed baked not env-read,
  idempotent within a mode, refuses to cross modes, launcher ordering, runtime
  manifest closure).
* **Live boot 2026-08-18 01:42:57Z**, fresh container, production patcher,
  `FR14_REQUIRE_NVFP4_LMHEAD=1`:

  ```
  INFO 08-18 01:42:57 [modelopt.py:2225] FR14_LMHEAD_NVFP4 ParallelLMHead prefix=language_model.lm_head resolved_algo=NVFP4
  INFO 08-18 01:42:57 [qwen3_5.py:511]  FR14_LMHEAD_QUANT_ROUTE lm_head quant_method=ModelOptNvFp4LinearMethod
  INFO 08-18 01:42:57 [__init__.py:683] Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
  ```
* And, independently of the log line, the capture patch recorded the *live*
  method on every one of 7,821 `compute_logits` calls:
  `"quant_method": "ModelOptNvFp4LinearMethod"`, `"logit_widths": {"248320": 7821}`.

### The brief's "env-flagged, default OFF" is inverted here, deliberately

`scripts/fr13_launch_forked_fa2_tree_server.sh:600` already carries
`export FR14_REQUIRE_NVFP4_LMHEAD=${FR14_REQUIRE_NVFP4_LMHEAD:-1}` and line 6853
runs the patcher unconditionally. **Making the NVFP4 head default-OFF would
break the pinned floor, not protect it**: arm B's `FR13_MANDATORY_WEIGHT_BYTES`
assumes the 4-bit head, so a default-OFF head would serve BF16 against a floor
6.695 ms of which is then fiction. That is precisely the fail-open class the
constant-train audit already fixed by baking the requirement at patch time.

The default-OFF, env-flagged thing this lane *did* add is the **observation**
path, not the head: `results/…/fr14_lane5a_capture_patch.py` is a no-op unless
`FR14_LANE5A_CAPTURE` is set, and the production launcher does not mount it.

## A.2 Floor re-derivation

`scripts/fr13_hardware_floor_ledger.py --derive-from-checkpoint` re-run today
against the real three-shard checkpoint: **PASS**, every pinned byte term
reproduced from real safetensors spans.

```
target_model_bytes           16,892,610,688     (1,841 tensors)
mtp_forward_bytes_per_pass      849,398,784  x5 = 4,246,993,920   (15 tensors)
lm_head NVFP4 on disk           715,161,608     (4 tensors)  =  2.6196 ms
lm_head BF16 reference        2,542,796,800                  =  9.3143 ms
                              -------------                    ---------
head delta                    1,827,635,192                  =  6.6946 ms
checkpoint total             21,921,428,072  (2,194 tensors) — equals RadixArk's
                                              own qualification.json exactly
```

**The floor with the NVFP4 head, and the counterfactual without it** (same
checkpoint, same everything, only the verifier head's precision changed):

| arm | NVFP4 head | BF16 head | delta |
|---|---|---|---|
| `root_64k` (the fixed32 pin) | **25,210,209,416 B / 92.3451 ms** | 27,037,844,608 B / 99.0397 ms | −1,827,635,192 B / **−6.6946 ms** |
| `full_vocab` (B1 contract) | **25,430,574,256 B / 93.1523 ms** | 36,396,385,408 B / 133.3201 ms | −10,965,811,152 B / **−40.1678 ms** |

Read the second row before pricing anything: under full-vocabulary drafting the
head is read six times per step, so the *same* lever is worth 40 ms there and
6.7 ms under K64. This is also why arm B nearly retires K64's reason for
existing (full_vocab 93.152 vs K64 92.345 = +0.807 ms, against +34.3 ms in the
fp8 era).

**Nothing moved.** The pinned constants already are the NVFP4-head numbers; this
re-derivation confirms them rather than replacing them. `PHASE2` (reading the
five K64 draft slices *as* NVFP4 rather than dequantising them) remains a
labelled projection at 22,798,484,616 B / 83.511 ms, −8.834 ms, unbuilt.

## A.3 Head-GEMM microbench — the honest numbers

`results/…/fr14_lane5a_head_gemm_microbench.py`, run in the pinned image on a
free GPU with **only the two heads resident** — no engine, no KV reservation, no
model body. The NVFP4 side goes through vLLM's own
`ModelOptNvFp4LinearMethod.create_weights → process_weights_after_loading →
apply`, which selects `FlashInferCutlassNvFp4LinearKernel`; re-implementing the
GEMM would have measured a kernel the serve does not run. Resident NVFP4 bytes
came back as 715,161,616 — the pinned 715,161,608 plus the 8 bytes the two
0-dim scalars gain when allocated as `[1]`.

| M | BF16 ms | NVFP4 ms | delta ms | speedup | BF16 GB/s | NVFP4 GB/s | BF16 %of 273 | NVFP4 %of 273 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.613 | 3.500 | 12.113 | 4.46× | 162.9 | 204.4 | 59.7% | 74.9% |
| 4 | 10.916 | 3.487 | 7.429 | 3.13× | 232.9 | 205.1 | 85.3% | 75.1% |
| 8 | 11.003 | 3.631 | 7.372 | 3.03× | 231.1 | 197.0 | 84.7% | 72.1% |
| 16 | 10.883 | 3.514 | **7.369** | 3.10× | 233.7 | 203.5 | 85.6% | 74.5% |
| 32 | 12.179 | 3.591 | 8.588 | 3.39× | 208.8 | 199.2 | 76.5% | 73.0% |
| 64 | 12.356 | 3.663 | 8.694 | 3.37× | 205.8 | 195.3 | 75.4% | 71.5% |

Against the pinned floor prices (2.620 ms NVFP4 / 9.314 ms BF16 / 6.695 ms
delta), three things follow, and only the first is comfortable:

1. **The lever over-delivers.** Measured delta 7.37–8.69 ms against 6.695 ms of
   credited floor. The campaign's in-serve attribution (`other` splits by
   checkpoint: radixark 7.456 ms mean over 96,299 decode steps vs unsloth
   15.655 ms — an **8.200 ms** delta) sits inside this range. Two instruments,
   one isolated and one in-serve, agree; the 6.695 ms in the floor is
   conservative.

2. **The two heads are not equally efficient, so `81.6% roofline` is a blend.**
   `host_dfwd_characterization.md` derived 81.6% as `floor_delta / measured_delta`
   = 6.695/8.200. The microbench decomposes it: the BF16 head runs at 85.6% of
   the 273 GB/s pin, the NVFP4 head at 74.5%. The NVFP4 GEMM is the *less*
   bandwidth-efficient of the two in isolation and still wins by 3.1× because it
   moves 3.56× fewer bytes. Nothing downstream breaks — but a future claim of
   the form "the NVFP4 head runs at 81.6% of roofline" would be wrong; 74.5% is
   the number.

3. **NVFP4 is flat in M and BF16 is not.** NVFP4 stays at 3.49–3.66 ms from M=1
   to M=64 (byte-bound throughout). BF16 costs 15.6 ms at M=1 — a 43% penalty
   over its own M=4 point — so the M=1 speedup of 4.46× flatters the lever, and
   the M=4–16 rows are the ones to quote for a verifier head.

## A.4 Logit characterisation on real hidden states

The comparison that matters is **the NVFP4 head against the true unquantised
BF16 head of the same base model**. That head is on disk: Qwen's own FP8-3.8
repo (`base_model: Qwen/Qwen3.8-27B`, the same base RadixArk quantised, rev
`e13a4f0e…`) ships `lm_head.weight` as **BF16 [248320, 5120]** — the FP8 recipe
quantises the body and leaves the head alone.

Two checks that this is really the reference, run rather than assumed:

* **Grid test.** An FP8-per-channel head dequantised to BF16 has at most 256
  distinct magnitudes per row. Sampled 8 rows: the FP8-3.8 head has
  **869–908** distinct magnitudes per row (native BF16); arm A's head — which
  `lmhead_surgery.py` produced by dequantising an FP8 head — has **88–95**.
* **Amax.** Dequantising the NVFP4 head and taking the max absolute value gives
  **0.341796875**; the reference head's amax is **0.341796875**. Ratio 1.0000.
  (This also settles the ModelOpt dequant direction empirically:
  `w = e2m1 · e4m3(block) · weight_scale_2` with `weight_scale_2 = 1.2716e-4`.
  The divide form would have produced an amax of 2.1e7.)

### Weight space (`lane5a_lmhead_weight_characterization.json`, all 1.27e9 elements)

| | NVFP4 head | arm-A FP8-derived head (CONTROL — already served) |
|---|---:|---:|
| relative Frobenius error | **9.483%** | 2.662% |
| RMSE | 1.304e-3 | 3.659e-4 |
| max abs weight delta | 1.921e-2 | 7.813e-3 |
| max relative row error | 9.955% | 2.850% |
| min row cosine | **0.995034** | 0.999597 |

NVFP4 carries **3.56×** the weight error of the FP8-derived head the campaign
already serves in arm A. That is the right frame: the question is not "is 4-bit
lossy" (it is, by construction) but "is it outside the precision envelope this
campaign has already accepted".

### Logit space (`lane5a_logit_characterization.json`)

Hidden states are **real**: 7,827 pre-`lm_head` rows captured from the live
serve at `logits_processor.py:96` — the exact call site whose GEMM the port
replaces — over 5 SWE-flavoured generations, of which a deterministic 1,024-row
spread was characterised. The engine's own top-2 logits were banked next to
every row.

**Device-kernel control first**, because everything else depends on it: the
offline NVFP4 head model reproduces the *device* kernel's argmax on
**995 / 1024 = 97.17%** of rows, and every one of the 29 disagreements is at a
device top-1 margin of **≤ 0.375** with most at exactly **0.0** — i.e. exact
ties, at or below the bf16 logit granularity. The offline model is sound.

| | NVFP4 vs BF16 ref | arm-A FP8 vs BF16 ref (CONTROL) |
|---|---:|---:|
| max abs logit delta | 1.2717 | 0.5245 |
| mean abs logit delta | 0.1411 | 0.0430 |
| p99 abs logit delta | 0.4753 | 0.1473 |
| (reference logit std) | 2.3684 | 2.3684 |
| argmax flips / 1024 | **27 (2.64%)** | 7 (0.68%) |
| top-32 overlap, mean | **0.9655** | 0.9884 |
| top-32 overlap, min | 0.7500 | 0.8750 |
| total variation, T=1.0, mean / max | 0.0221 / 0.2066 | 0.0072 / 0.0798 |
| total variation, T=0.6, mean / max | 0.0246 / 0.3251 | 0.0080 / 0.1396 |

**The flips, conditioned on how decided the position was.** A pooled flip rate
conflates "the model changed its mind" with "the model was flipping a coin and
the coin landed the other way", and the conflation always flatters whoever is
quoting it:

| reference top-1 margin | rows | NVFP4 flips | FP8 flips |
|---|---:|---:|---:|
| ≥ 3.0 | 553 | **0** | 0 |
| 1.0 – 3.0 | 239 | **0** | 0 |
| 0.25 – 1.0 | 160 | 6 (3.75%) | 0 |
| < 0.25 | 72 | 21 (29.2%) | 7 (9.7%) |

**Confident-flip rate (margin ≥ 1.0): 0.0000 for both heads, over 792 rows.**
Not one position the BF16 head had actually decided was decided differently by
the NVFP4 head.

And what the flips swapped, verbatim from the artifact (`Ġ` is the BPE spelling
of a leading space):

```
'Ġboth'       -> 'Ġeither'        margin=0.0167
'Ġbe'         -> 'Ġuse'           margin=0.0295
'ĠDetection'  -> 'Ġdetection'     margin=0.0718
'ĠIDs'        -> 'Ġnames'         margin=0.1522
'Ġrunners'    -> 'ĠCI'            margin=0.0546
'Ġcarefully'  -> 'Ġpolished'      margin=0.3338
"('"          -> '("'             margin=0.1009
'Ġsorted'     -> 'Ġcopying'       margin=0.0852
```

Near-synonyms and whitespace/quote variants inside reasoning prose. Not one flip
lands on a numeral, an identifier, a control-flow keyword or a delimiter.

**Structural note for acceptance.** Under exact rejection sampling the served
output distribution is the *verifier's*, so the NVFP4 verifier head is what
determines output quality; the NVFP4-derived K64 drafter slice affects only the
acceptance rate, i.e. speed. The T=0.6 total-variation column (mean 0.0246) is
the quantity that bounds the distributional shift a paired A/B would see.

## A.5 GENERATION PROBE + EYEBALL VERDICT

Two GPU windows, NVFP4 head live and fail-closed, `--enforce-eager`, seed
20260818. Eight traces total across greedy (T=0) and the campaign's serving
sampler (T=0.6 / top_p 0.95 / top_k 20), including a deliberate 2,560-token
repetition trap and the tool-call path under the production parser.

### VERDICT: **NO DEGENERATION SIGNATURE. PASS. Lane may proceed.**

Read in full, not sampled. Mechanical signatures were computed only to point at
where to look.

**`greedy_exactness` (finish_reason `stop`)** — the cleanest single piece of
evidence, because it is checkable rather than impressionistic:

```
We need to respond to user: "Reply with exactly the line: FR14 lane5A NVFP4 head
alive. Then on a new line state the value of 17*23, and on a third line the value
of 2**16."

Need final exactly three lines:
FR14 lane5A NVFP4 head alive
391
65536
Ensure no extra. 17*23=391. 2**16=65536.
</think>

FR14 lane5A NVFP4 head alive
391
65536
```

Exact string reproduced, both arithmetic facts correct, three lines, clean stop.

**`greedy_bugfix`** — correctly names *both* planted bugs and writes correct
Python:

```
Boundary bug: overlapping intervals that touch at endpoints? ... For closed
intervals, [1,2] and [2,3] overlap at point 2, should merge to [1,3]. If they
mean overlapping maybe touching counts? Boundary bug likely using < instead of
<=. ... Mutation bug: intervals.sort mutates input list; out = [intervals[0]]
references original interval list/tuple? If intervals are lists, last[1] mutates
original interval in input.
```

and its final formulation is the cleanest correct implementation of the three it
considered:

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    out = []
    for start, end in sorted_intervals:
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return out
```

**`greedy_traceback`** — correct root cause (`len(self._devices) == 0`) and a
well-formed unified diff:

```
Minimal patch:
--- a/pipeline/loader.py
+++ b/pipeline/loader.py
@@ -185,7 +185,9 @@
     @property
     def device(self):
-        return self._devices[self._rank % len(self._devices)]
+        if not self._devices:
+            return "cpu"
+        return self._devices[self._rank % len(self._devices)]
```

**`sampled_repetition_trap`** (the deliberate bait: 40 distinct items, 2,560
tokens) — no loop; the enumeration stays distinct across all of it:

```
1. Shared mutable global state across packages.
2. Test order dependence within package.
3. Database state not isolated.
...
14. Hash randomization.
15. Port conflict.
16. Docker/container state.
17. Mock patch leak.
```

Signature table (numbers point at where to look; they are not the verdict —
a numbered 40-item list is repetitive by construction and fluent nonsense would
pass every column):

| trace | regime | words | TTR | max line repeat | top 8-gram count | tail-repeat frac | non-ASCII frac | unbalanced `{}` `[]` `()` ``` | finish |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| greedy_bugfix | greedy | 836 | 0.374 | 5 | 3 | 0.500 | 0.00000 | 0/0/0/0 | length |
| greedy_traceback | greedy | 841 | 0.396 | 6 | 3 | 0.476 | 0.00000 | 0/0/0/0 | length |
| greedy_exactness | greedy | 60 | 0.667 | 2 | 1 | — | 0.00000 | 0/0/0/0 | **stop** |
| sampled_long_code | sampled | 1040 | 0.445 | 3 | 1 | 0.031 | 0.00000 | 0/0/0/0 | length |
| sampled_repetition_trap | sampled | 1640 | 0.460 | 2 | 1 | 0.000 | 0.00000 | 0/0/0/0 | length |
| p2_greedy_bugfix_to_completion | greedy | 118 | 0.822 | 1 | 1 | 0.000 | 0.00000 | 0/0/0/0 | **stop** |
| p2_greedy_tool_call | greedy | — | — | — | — | — | — | — | **tool_calls** |
| p2_sampled_tool_call | sampled | — | — | — | — | — | — | — | **tool_calls** |

**The one thing that looked like a signature, and what it actually was.** The
two greedy long traces show `tail_repeat_fraction ≈ 0.5`. Reading them: this is
the model re-deriving the *same* corrected function in three different
formulations while deliberating — restatement, not a decode loop. The top
repeated 8-gram occurs 3 times in ~840 words; a genuine loop pins that counter
into the tens or hundreds. Zero non-ASCII characters across every trace, zero
delimiter imbalance, zero mid-word breaks on reading.

**Phase 1's disclosed limitation, and how phase 2 closed it.** Four of five
phase-1 traces ended with `finish_reason: length` *inside* the reasoning block,
so phase 1 read the model's deliberation but never its final answer — and the
answer is where truncated code, unbalanced delimiters and mid-word breaks would
show. Phase 2 re-ran one at 6,144 tokens. It finished (`finish_reason: stop`,
2,259 completion tokens) with a complete, correct, cleanly formatted answer:

```
**Boundary bug:**
`start < last[1]` fails to merge intervals that touch at an endpoint, e.g. `[1, 2]`
and `[2, 3]`. For closed `[start, end]` intervals, this should be `start <= last[1]`.

**Mutation bug:**
The function mutates the input:

- `intervals.sort(...)` sorts the caller's list in place.
- `out = [intervals[0]]` stores a reference to the original interval.
- `last[1] = ...` then mutates the original interval object.
```

followed by a balanced, correct fenced code block. Both planted bugs named
precisely; markdown well-formed; type-token ratio 0.822; max line repeat 1.

### Tool calls — Mark's named "malformed tool call" signature

Phase 1's tool request returned **HTTP 400** because the probe had booted
*without* the production serve line's parser flags. That is a probe defect, not
a model result, and it is recorded rather than hidden. Phase 2 re-ran under
`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
— the same flags as `fr13_launch_forked_fa2_tree_server.sh:7168` — so a tool
call this probe rejects is one the real serve would also reject.

**Both regimes emitted well-formed calls. `finish_reason: tool_calls`.**

Greedy, verbatim from the artifact:

```json
{"name": "run_tests",
 "arguments": "{\"path\": \"tests/test_loader.py\", \"keyword\": \"shard_rank\", \"verbose\": true}"}
```

Sampled — two calls, and in the order the prompt asked for ("Before changing
anything, read … and then run …"):

```json
{"name": "read_file",
 "arguments": "{\"path\": \"pipeline/loader.py\", \"start_line\": 180, \"end_line\": 220}"}
{"name": "run_tests",
 "arguments": "{\"path\": \"pipeline\", \"keyword\": \"rank\"}"}
```

Every call: JSON parses, function name is one of the two declared, **zero**
unknown parameters, types correct (`verbose` a real boolean, line numbers real
integers). No malformation of any kind.

Temperature-0 over-deliberation on a reasoning model is a known artifact of
greedy decoding, not a precision symptom — the campaign serves at 0.6/0.95/20 —
but a probe that never sees a completed answer cannot claim to have checked one,
which is why phase 2 exists.

**External corroboration.** RadixArk's own `qualification.json` for this exact
checkpoint records gsm8k 1319/1319 attempted, **97.27%** correct, `stop_rate
1.0`, `truncations 0`, `empty_generations 0`. And the campaign's own SGLang
recon banks NVFP4 gsm8k at 96.82% with bf16 KV.

## A.6 What is genuinely open on sub-lever A

* **The `~-2.6 ms cfwd commit logits` half of the brief is not attributable to
  the head bytes.** cfwd reads no model weights on the floor ledger, and the
  verifier head appears once per step in `LEGACY_MANDATORY_WEIGHT_BYTES`. Either
  the commit path reads the head a second time — in which case the ledger is
  missing a term worth another 6.695 ms and that is a floor bug, not a lever —
  or the −2.6 ms belongs to a different lever. **Flagged for Mark; not claimed
  here.** cfwd's own 20.66 ms/step is precision-invariant full-vocab activation
  traffic, and shrinking it is a separate target.
* **PHASE 2 draft-head FP4 reads** (−8.834 ms of floor, 22,798,484,616 B) remain
  a labelled projection needing an FP4 GEMV unit and its own byte gate.
* **No paired ≥20k-step A/B.** Out of scope by the brief; this lane's evidence is
  offline plus two bounded GPU windows.

---

# SUB-LEVER B — fp8-KV: FEASIBILITY ASSESSMENT (stopped at the plan stage)

## B.1 What exactly rejects fp8 KV — four stacked surfaces, only three removable

The campaign's kv-scheme surgery exists because the checkpoints *ask* for fp8 KV
and the tree kernel cannot take it. Going the other way means clearing all four:

| # | surface | site | removable by config? |
|---|---|---|---|
| 1 | checkpoint declares fp8 KV → engine sets `cache_dtype='fp8_e4m3'` | `attention.py:231-239` (arm A) / `arg_utils.py:1616`→`torch_utils.py:279` (arm B) | yes — this is what the two surgeries undo |
| 2 | `TreeAttentionBackend.supported_kv_cache_dtypes = ("auto","float16","bfloat16")` | raised at `platforms/cuda.py:303` | only by patching vLLM, and the guard is *correct* |
| 3 | `ModelServer._initial_kv_cache_dtype` rewrites `fp8_e5m2 → auto` | `scripts/p5b_fp8_kv_purity_attestation.py:300-330` | yes, in-repo |
| 4 | **the FA2 fork itself** | see B.2 | **no** |

Surface 2's refusal, verbatim from
`results/…/nvfp4_fixed32_boot1_treeattn_kv_refusal.log:134`:

```
(EngineCore pid=154) ERROR 08-16 19:40:14 [core.py:1129] ValueError: Selected backend
AttentionBackendEnum.TREE_ATTN is not valid for this configuration.
Reason: ['kv_cache_dtype not supported']
```

Emitted by `CudaPlatform.get_attn_backend_cls`, reached from `Attention.__init__`
at layer-construction time — **before a single weight is read**. Note the engine
banner on that same run still printed `kv_cache_dtype=auto`: the fp8 forcing
happens per-`Attention`-layer, downstream of the banner, which is why the
refusal is invisible from the config line.

## B.2 The FA2 fork has no fp8 path, and nowhere to put one

Exhaustive search of the fork for `fp8|e4m3|float8|dequant|descale|k_scale|v_scale`
returns **zero hits** in the mainloop. Concretely:

* **One element type for everything.** `Flash_kernel_traits` carries a single
  `Element`, and the KV cache is read by reinterpreting the raw pointer as it
  (`flash_fwd_kernel.h:845,849`). KV cannot be narrowed independently of Q and O.
* **No scale plumbing.** `Flash_fwd_params` has no `k_scale`/`v_scale`. The only
  dtype state that reaches the kernel is `params.is_bf16`, a **bool**.
* **The C++ API refuses first.** `flash_api.cpp:647-651`:
  `TORCH_CHECK(q_dtype == kFloat16 || q_dtype == kBFloat16, "FlashAttention only support fp16 and bf16 data type")`.
* **The dispatch switch is two-way.** `FP16_SWITCH` (`static_switch.h:79-88`) has
  exactly two arms, half and bfloat16. There is no third.
* **All six tree traits instantiate `cutlass::bfloat16_t` only**, and all seven
  generated TUs are named `…_hdim256_bf16_sm80.cu`.
* **Static strides are baked at 2 bytes/element.** `params.k_row_stride == 4*256`,
  `k_batch_stride == 2*1024*4*256` are element-count constants asserted in every
  private tree launcher; smem sizing and the cp.async vector width are all
  `sizeof(Element)` arithmetic.

So the minimal path is not "add a dequant" — it is: add a KV element-type axis to
the traits, add descale plumbing to `Flash_fwd_params`, add a dequant stage
between the cp.async landing and the MMA, re-derive every smem/copy-width
constant, re-derive the static stride asserts, and re-instantiate all six trait
sets. That is a rewrite of the same mainloop LANE 4's split-K work is rewriting.

## B.3 The price is wrong — settled by LANE 4's measurement, not by argument

This is the finding that should decide the lever. TreeAttn-v2 §12 fitted a
two-term model to real measurements across a 2× context range (α stable at
**0.649–0.664 ms per staged GB**, 2.3% spread) and decomposed today's attention
at 23k context:

| configuration | bytes term | parallelism term | total |
|---|---:|---:|---:|
| G=2 (today) | **2.98 ms** | 10.46 ms | 13.44 ms |
| hypothetical zero-byte, 12 CTAs | 0.00 | 10.46 | 10.46 (**−2.98, hard ceiling**) |
| split-K, 48 CTAs (LANE 4) | 0.99 | 2.61 | 3.61 (−9.83) |

Today's attention is **78% parallelism-bound**, and the implied staging bandwidth
is **1.52 TB/s against 273 GB/s of DRAM** — the re-staged bytes are
overwhelmingly L2 hits.

fp8-KV halves the KV element and touches nothing else, so it halves the bytes
term and leaves the parallelism term alone:

* **Before split-K:** 2.98 → 1.49 ms. **Ceiling ≈ −1.49 ms/step.**
* **After split-K:** 0.99 → 0.50 ms. **Ceiling ≈ −0.50 ms/step.**

Against the briefed **~−10 ms**. The estimate is off by ~7× standing alone and
~20× if split-K lands first, and it is off for a documented reason: REDTEAM
pass 22 already flags that "all pool numbers are fp8-era-scaled", and the −13 ms
head-merge lever priced by the *same* byte-proportional model was refuted by
this same measurement. A second, independent refutation exists:
`fr13_b4_honest_floor_20260814/README.md` — *"The marginal KV token costs ≤ 0.32 ns
of FA2 time against a 15.0 ns DRAM floor — 47× cheaper than physics allows if the
kernel were reading that KV from DRAM. Proportionality between FA2 time and KV
bytes is rejected at width 4."*

Note also that the floor ledgers contain **no KV byte term at all** — they are
weight-bytes-only — so fp8-KV moves the floor by exactly zero and would have to
be justified entirely on measured wall.

## B.4 Plan, and the STOP

**STOPPED at the plan stage, per the brief's instruction**, because the minimal
path requires forking the FA2 mainloop concurrently with LANE 4. Beyond the
concurrency conflict, the lever does not survive its own repricing.

**Recommendation to Mark: do not build fp8-KV.** Sequencing if it is ever
revived:

1. **Wait for LANE 4.** Whichever mainloop wins (split-K at 48 CTAs, or F2
   cluster/DSMEM) is the one an fp8 KV path must be built on. Building in
   parallel guarantees a merge against a mainloop that no longer exists.
2. **Re-price before any kernel work**, on the winning mainloop's own α fit.
   Pre-registered kill rule: if the predicted saving is under 1.0 ms/step at the
   B1 operating context, do not build. On today's numbers it predicts 0.50 ms
   post-split-K and fails that rule.
3. **If it is built anyway**, the non-negotiable evidence set is: (a) the
   checkpoint's KV scales are *uncalibrated* — RadixArk's own `qualification.json`
   warns `"Using FP8 KV cache but no scaling factors provided. Defaulting to
   scaling factors of 1.0"` and ships **zero** `k_scale`/`v_scale` tensors across
   all 2,194 — so calibration is a prerequisite, not a detail; (b) SGLang's own
   measurement on this checkpoint puts fp8 KV at **96.44%** gsm8k vs **96.82%**
   for bf16 KV, i.e. a real −0.38 pt; (c) a degeneration eyeball at **long**
   context specifically, since that is where a KV precision loss compounds and
   where this lane's short-context traces say nothing.
4. **The capacity benefit is separate and unpriced here.** Halving KV bytes
   doubles the KV pool at fixed memory (196,608 → ~393,216 tokens). That is a
   concurrency/context-length argument, not a step-time one, and if fp8-KV is
   ever revived it should be revived on that argument rather than on ~10 ms.

---

## Artifacts

Tools (all default-OFF, none in the production runtime closure):

* `results/fr14_nvfp4_port_20260816/nvfp4_lmhead_characterization.py` — weight- and
  logit-space characterisation, CPU-only.
* `results/fr14_nvfp4_port_20260816/fr14_lane5a_capture_patch.py` — container-side
  hidden-state + device-argmax capture; no-op unless `FR14_LANE5A_CAPTURE` is set.
* `results/fr14_nvfp4_port_20260816/fr14_lane5a_generation_probe.sh` — boot +
  generation probe + capture, production preflights, unconditional teardown.
* `results/fr14_nvfp4_port_20260816/fr14_lane5a_prompts.py` — the prompt set.
* `results/fr14_nvfp4_port_20260816/fr14_lane5a_eyeball.py` — degeneration
  signatures beside the traces.
* `results/fr14_nvfp4_port_20260816/fr14_lane5a_head_gemm_microbench.py` +
  `fr14_lane5a_run_microbench.sh` — isolated head GEMM, real kernel, real weights.

Results:

* `lane5a_lmhead_weight_characterization.json`
* `lane5a_logit_characterization.json`
* `lane5a_head_gemm_microbench.json`
* `lane5a_capture_meta.json` (phase 1, 7,827 rows) /
  `lane5a_capture_meta_p2.json` (phase 2, 2,517 rows)
* `lane5a_generations_eyeball.json` (phase 1 traces, verbatim) /
  `lane5a_generations_eyeball_p2.json` (phase 2, incl. tool calls)

Reproduce:

```bash
# offline, CPU only
python3 results/fr14_nvfp4_port_20260816/nvfp4_lmhead_characterization.py \
    --phase weights --chunk 8192 --out <out>.json
python3 results/fr14_nvfp4_port_20260816/nvfp4_lmhead_characterization.py \
    --phase logits --hidden <capture>.f32 --max-rows 1024 --chunk 8192 --out <out>.json

# GPU, docker-empty required, container removed on exit
bash results/fr14_nvfp4_port_20260816/fr14_lane5a_run_microbench.sh --iters 30
bash results/fr14_nvfp4_port_20260816/fr14_lane5a_generation_probe.sh
SUFFIX=_p2 PROMPT_SET=--phase2 bash results/fr14_nvfp4_port_20260816/fr14_lane5a_generation_probe.sh
```

GPU discipline for this lane: three bounded windows (23 min, 2 min, 9 min), each
preceded by a docker-empty check and a unified-memory gate, each tearing its
container down unconditionally on exit. Zero containers left at every boundary.
