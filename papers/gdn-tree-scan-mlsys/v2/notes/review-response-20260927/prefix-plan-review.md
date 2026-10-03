# Q1 prefix preparation: independent CPU review

Reviewed 27 September 2026. Scope: the finalized `experiments/review-response-20260927/prefix-plan/` draft and its bound inputs. No owner file was changed. No model weights, GPU, API request, server, Docker container, or benchmark execution was used. One read-only SSH Python process recomputed the six selected renderings/tokenizations using the existing tokenizer assets; it wrote no remote files.

**PASS for source-bound input preparation. No material preparation defect found. This is not Q1 qualification, a frozen scenario denominator, or permission to launch.** The explicitly pending actual-runtime token-ID gate and protocol-class decision remain necessary before the planned qualification.

## Reviewed identities

Paths below are relative to `experiments/review-response-20260927/prefix-plan/` unless otherwise stated.

| Artifact | SHA-256 |
|---|---|
| `MANIFEST.json` | `7fbf73db7d31b4a78fe26f2aa95d70bb160c1d779d47172c6d95f244f958bacf` |
| `prefix-manifest.draft.json` | `f0450942008e12ee0bbf99bbdc6de0db3c324d77c0b23feafd788437cd537eaf` |
| `PREFIX-PLAN-DRAFT.md` | `ad65867aec6f884214ef2f7ee722f3f9113067573536c2c14214f46dbdee247f` |
| `inventory-raw.json` | `dfae2633a4446f603ea9f60dbce6a7818854ec0ed031d13a554820fa04cd96a5` |
| `select_prefixes.py` | `b03acad702b65f738da332dad6acdf86231622365671690c51b17b010903251f` |
| `inventory_prefixes_readonly.py` | `ae08a81979a512cc14bd6fae589d254ce9150cf2162c4376044797bac00d8468` |
| `verify_artifacts.py` | `0e3c0c93750361b541e5aa5bd91d8b73e2f59d22f8177e46baa90ac39eea246b` |
| `test_selection.py` | `58fcad6d7965fcfeb967cd0513846c8e13120e1ca1e224ef369f069a48f71104` |

All 50 manifest members independently matched their SHA-256 and byte count; paths were unique, and the directory contained exactly those files plus `MANIFEST.json`. All 24 selected payload files matched their receipts. The manifest was rehashed after review and remained unchanged.

## Selection and separation

I independently reconstructed the task ranking, alternating assignment, token-count tertiles, anchors, unused-task constraint, and final nearest-anchor selection without calling the owner's selector. The six selected source hashes matched exactly.

- All 390 inventory rows are eligible single-task text candidates under the stated rules. Each `split_group` equals `task:` plus its parsed task ID. The 16 task identities divide into eight calibration and eight evaluation groups; all requests from a task inherit the same assignment across archived runs.
- The six selected prefixes use six different task IDs. Calibration/evaluation task overlap is zero. This prevents nested prefixes from the same issue crossing the split; it does not turn a historically observed Astropy diagnostic set into an unseen benchmark sample.
- The bound E3 `workload-plan/task-selection.json` matches the recorded hash. Its actual `phases` contain 120 distinct task IDs. The exclusion list equals those phase IDs, and the selected Q1 task intersection is empty. Historical exclusions listed elsewhere in the E3 file are correctly not treated as E3 selected tasks.
- Source-family eligibility and selection code read request content/identity, source hashes, current-rendered lengths, and parsing compatibility. They do not open evaluator outcomes, timing summaries, patches, or benchmark gold/test-patch files to choose inputs. Historical tool observations remain part of the original conversations, as the plan discloses. Selection is a size-stratified diagnostic proposal, not a random or outcome-blind fresh-task evaluation claim.

The available classes are honest relative labels, not the former absolute short/medium classes:

| Class | Observed token range | Records | Anchor |
|---|---:|---:|---:|
| Short available | 13,373–26,112 | 130 | 13,373 |
| Medium available | 26,347–31,851 | 130 | 28,941 |
| Long available | 31,875–60,083 | 130 | 60,083 |

There are zero inputs below 1,000 tokens and zero between 4,000 and 8,000 tokens. No truncation or synthetic padding supplies missing classes. `qualification_denominator` remains null, and `PREFIX-PLAN-DRAFT.md:37` explicitly requires a protocol amendment before execution.

## Independent original → normalization → rendering → token checks

The supplied verifier preserves each original source SHA, all original messages and tool definitions, and the separate non-executable replay-input schema. I additionally checked the replay template kwargs against the selected row's bound `render_kwargs` after removal of `add_generation_prompt`.

To avoid merely repeating the preparation normalizer, I extracted and executed the actual pinned vLLM functions `_parse_chat_message_content_mm_part`, `_parse_chat_message_content_part`, `_parse_chat_message_content_parts`, `_parse_chat_message_content`, and `_postprocess_messages`. The text-only harness supplied a no-media tracker and the source-equivalent text dictionary parser; it did not import vLLM. All selected content was checked to be within the accepted strings/text-arrays/null subset. All six normalized-message hashes matched the inventory.

The read-only SSH check then freshly hashed the four tokenizer assets and custom template, reread all six original remote source requests, and used those pinned parser functions plus the image-extracted `_cached_compile_jinja_template`. It reproduced every rendered-prefix SHA and complete little-endian uint32 token-buffer SHA, not only the token counts. `AutoTokenizer` loaded tokenizer files only, offline and with remote code disabled. Torch/TF and CUDA visibility were disabled; no model weights or vLLM modules were loaded.

| Prefix | Task suffix | Messages | Exact reproduced tokens |
|---|---:|---:|---:|
| Calibration short | 14598 | 2 | 13,487 |
| Calibration medium | 13977 | 22 | 28,941 |
| Calibration long | 13398 | 35 | 60,083 |
| Evaluation short | 13033 | 2 | 13,373 |
| Evaluation medium | 12907 | 14 | 28,930 |
| Evaluation long | 13453 | 34 | 48,688 |

The current custom template SHA is `c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`. The four freshly checked tokenizer hashes match `WEIGHTS-TOKENIZER-LOCK.json`. Libraries reproduced the preparation versions: transformers 4.57.1, tokenizers 0.22.1, Jinja2 3.1.2. Independent SSH stdin SHA-256: `3f6ae5714350e6a715cf1519571c9cff7fd8c82efe788a8daad623bd2babdd17`; command `ssh -o BatchMode=yes -o ConnectTimeout=8 mark@100.103.10.122 python3 -B -`, return code 0.

The source extracts retain their image identity `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. The recorded Cqc10 content-format receipt identifies `string`; the draft also retains the different OpenAI-array rendering as a negative control. This establishes the declared CPU rendering, not the behavior of a future fully patched HTTP path.

Five current files in `source-bindings.json` still match outright. `WORKLOG.md` has since been appended: its original first 10,842 bytes still match the recorded `e36f08a7608cc9307354af16eefb5aa3c76c34b7a2de104052744507e9c9e44f`. Thus this is a recoverable historical source binding, not unexplained input drift. No manifest rewrite is needed for that append.

## Executed checks and remaining boundary

From the prefix-plan directory:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B test_selection.py
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B verify_artifacts.py
```

Results: **10/10 tests pass**; verifier reports 390 inventory records, six prefixes, 24 checked payloads, four matching tokenizer assets, and zero qualification executions. Existing negative controls cover altered source/render/token buffers, outcome metadata, input order, absent strata, and content-format changes. Independent reconstruction additionally checked actual E3 hash/IDs, all task assignments, complete manifest inventory, replay kwargs, pinned-parser normalization, and source-to-token reproduction.

`PREFIX-PLAN-DRAFT.md:47,58` explicitly requires prompt IDs or a full-buffer hash **after the actual runtime tokenizer/HTTP adapter**, compared with each bound token file before current-route operand qualification. That gate is a stated future obligation; neither this review nor the CPU preparation implements or executes it. Any mismatch must block input identity and require versioned regeneration/review, rather than silently changing the prefix or counting it as a numerical experiment failure.

The remaining protocol-class/denominator freeze and runtime parity check are already disclosed pending work, not new defects. No additional corpus search, synthetic prefix, GPU run, or new benchmark attempt is requested by this preparation review.
