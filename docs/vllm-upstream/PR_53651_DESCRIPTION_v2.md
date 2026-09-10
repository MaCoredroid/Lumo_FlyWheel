# PR #53651 — description rewrite (v2.1, Codex round-1 applied) + single status ping
> **POSTED 2026-09-10 (Mark greenlight, Codex final GO):** title + body applied via REST PATCH at 22:55:03Z (gh pr edit failed on a GraphQL deprecation and was retried); ping https://github.com/vllm-project/vllm/pull/53651#issuecomment-5626523964

> Plan v3 item P11. Head 66ab2231c, rebased over #54160 on 2026-08-29.
> Residual on current main (verified 2026-09-10): `_get_untied_lm_head`
> still gates on `UnquantizedEmbeddingMethod` only; the materialized-weight
> guard is absent. Codex corrections applied: #54160 DID touch the cast site
> (so "two sites #54160 did not touch" was false); the duplicate search now
> returns #53651 and #55494 (packed-BF16 head wrapper; unwraps at both sites,
> does not add either residual behavior) — acknowledged as overlap; the
> guard RESTRICTS entry (materialized 2-D weight), so "only widens" was
> wrong; local results ≠ CI (pre-run check fails, pre-commit skipped).
> Should-fix (Mark's call): retitle to
> "[Bugfix] Handle excluded lm_head re-tying and unavailable cast weights".
> Mark GO needed for (a) description edit, (b) optional title edit, (c) ping.

---

## Description (replace the PR body with this)

## Purpose

After #54160 admitted unquantized-linear heads in the `head_dtype` cast
path, two cases remain:

1. `_get_untied_lm_head` still recognizes only
   `UnquantizedEmbeddingMethod`. A head excluded from quantization carries
   `UnquantizedLinearMethod`, so an otherwise eligible head can be excluded
   from weight re-tying and its memory reclaim. This PR recognizes both
   method classes there.
2. The cast path reads `lm_head.weight` directly. Some CPU fused-GEMM
   configurations release that weight after loading. This PR requires a
   materialized 2-D weight before casting and raises a descriptive
   `ValueError` when it is unavailable.

The shared `is_unquantized_method()` helper expresses method classification;
weight availability is a separate check in `LogitsProcessor`. Existing
materialized heads supported by the cast path remain supported.

## Related work

- #54160 merged the cast-path method acceptance. This PR retains it, adds
  the materialized-weight check, and updates weight re-tying.
- #52883 changes a separate DFlash2 candidate-selector call site.
- #55494 adds a packed-BF16 head wrapper and unwraps its fallback at the
  same cast/re-tying sites. It does not add the two residual behaviors here;
  these overlapping sites will need reconciliation when either PR lands.

## Validation and limits

Recorded local validation for head `66ab2231c`:

- `tests/v1/sample/test_head_dtype.py`: 9 CPU tests passed. The added tests
  exercise a real excluded `ParallelLMHead` through post-loading weight
  processing and check casting or clean refusal, plus refusal for an
  unavailable weight. No new test directly covers weight re-tying.
- Local ruff and mypy checks passed.
- GB10 smoke: `LLM(model="facebook/opt-125m",
  hf_overrides={"head_dtype": "float32"}, enforce_eager=True)` followed by
  greedy generation completed. This is a basic cast-path smoke, not an
  end-to-end excluded-ModelOpt-head test.

At the September 10 check, DCO and formatting passed; the pre-run eligibility
check failed and pre-commit was skipped. These local results do not establish
that upstream CI has run successfully.

## Disclosure

AI assistance was used in preparing the change. The submitter reviewed the
changed lines and ran the recorded validation above.

---

## Status ping (one comment, only AFTER the description edit)

@22quinn — the description now leads with what remains after #54160:
excluded-head classification for weight re-tying and the materialized-weight
guard for `head_dtype`. It includes the recorded local validation, the
re-tying test-coverage limit, and the overlap with #55494. This is ready for a
review of those residual changes; upstream pre-commit remains gated.
