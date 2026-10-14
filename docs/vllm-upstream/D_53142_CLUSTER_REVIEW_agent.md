# Cluster map: align-mode Mamba seed divisor (verified 2026-09-17, read-only)

## 1. Live state

| Item | Author | Created | State / head | Labels | Human maintainer? |
|---|---|---|---|---|---|
| PR #53798 | ptorsten | 08-25 | OPEN, head `af5357c2b` (unchanged) | bug, ready, verified, mrv2, needs-rebase | **Yes** — AndreasKaratzas [MEMBER] 09-03 LGTM; `ready`+`verified` same minute |
| PR #55507 | Karl0007 | **09-05** (09-14 = last update) | OPEN, `14e86625f`, MERGEABLE/BLOCKED | bug, mrv2 | No |
| PR #55601 | pondzikk | 09-06 | OPEN, `7cd542548` | bug, mrv2 | No |
| PR #53803 | ptorsten | 08-25 | OPEN **draft** ("Draft while #54076 is open") | bug, needs-rebase, mrv2, kv-cache | No |
| PR #55688 | askliar | 09-07 | OPEN, `89f5ad734` | needs-rebase, ci/build, mrv2, kv-cache-manager | No |
| Issue #53142 | mjungnickel18 | 08-20 | OPEN, 12 comments / 8 participants | **`quantization` only** | No, unassigned |
| Issue #55600 | pondzikk | 09-06 | OPEN, **0 comments**, no labels | — | No |

`review_requested` (WoosukKwon/njhill/yewentao256) is automatic on all three fix PRs — not a signal; only #53798 has a human maintainer touch. Adjacent: #54076 (scheduler chunk split, OPEN 09-17), #54173 (independent sm_121 report), #54199 (retracted dup), `wtdcode/vllm-backport#77` ("resolve on read"). **Bug still live on `main`** (`mamba_hybrid.py:122`).

## 2. Diffs

- **#53798** — `mamba_spec.block_size` from `get_mamba_groups(kv_cache_config)` in a **new `ModelState.set_kv_cache_config` hook**, called from `initialize_kv_cache` **before any `add_request`**. Keeps the all-groups-agree assert. +1 new unit test (880 vs 16), 3 updated. 6 files.
- **#55507** — `self._mamba_spec.block_size` **if not None, else `cache_config.block_size`** (lazy; `_mamba_spec` binds only on the first `preprocess_state`/`prepare_attn`), plus a comment-only fix in `_ensure_align_ctx`. **No tests.** 1 file.
- **#55601** — `self.cache_config.mamba_block_size`. No binding order at all. **No tests** ("No unit test is included"). 1 file.
- **#55688** — verified: `state_block_size = cache_config.mamba_block_size` **only under `if self._use_flashinfer_replayssm:`**; Triton/GDN/KDA align keeps `cache_config.block_size`.

**Equivalent?** Numerically yes on `main`: `MambaSpec.block_size` is built in `mamba/abstract.py:get_kv_cache_spec` from `cache_config.mamba_block_size`; align mode's `Platform._align_hybrid_block_size` sets `mamba_block_size = cache_config.block_size` *before* `EngineCore._initialize_kv_caches` lowers `cache_config.block_size` to `min(prefix-cacheable groups)`. `unify_kv_cache_spec_page_size` **pads** Mamba pages rather than scaling their `block_size` (#53798's "scale past" wording is loose). They differ in robustness:
- #55601 adds a second source of truth — diverges if a Mamba group's `block_size` is ever rewritten post-config (packed/hidden-state/DCP `replace(spec, block_size=...)` already do this for other spec types) or if Mamba groups differ. wickist flagged this.
- #55507's fallback has a narrow but real hole: an external KV connector (`num_external_computed_tokens`) can deliver `num_computed_tokens > 0` in the **first** batch, before `_mamba_spec` is bound — silently reproducing the bug (local prefix caching cannot; cache is empty on batch 1).
- #53798 has neither, at the cost of a new interface method.

## 3. Smallest honest statement

Diagnosis and one-line fix are **mjungnickel18's, #53142, 08-20** — the issue body already holds #55507's exact code shape. First PR: **#53798, 08-25** (eager binding, only unit tests, only maintainer LGTM). Davan-Etelamaki re-offered the snippet 09-05; #55507 opened hours later (crediting @zebgop-ops); #55601 followed 09-06 from an independent GLM-5.3-Flash repro with the strongest e2e evidence (3-hour soak, 133/133) plus its own issue #55600. Parallel discovery, not bad-faith duplication.

## 4. Would our test pass/fail? (reasoned from diffs, no GPU)

Our fixture calls `state.set_kv_cache_config(...)` and fakes `cache_config = SimpleNamespace(block_size=..., mamba_cache_mode="align")`:
- **#53798: passes** (written against that API).
- **#55507: errors as-is** (`AttributeError: set_kv_cache_config`); swap that line for `state._mamba_spec = mamba_spec` → **passes**; leave it `None` → **fails**, catching the fallback hole.
- **#55601: errors as-is**; add `mamba_block_size=M` to the fake cache_config → **passes**.
- **Old divisor (`main`, #55688 non-FlashInfer): fails on contents** (seeds column 6, oracle needs 2→3).

One small fixture edit per target; it discriminates a wrong divisor everywhere but cannot rank the three correct ones.

## 5. DRAFT comment for #55507 — NOT POSTED (165 words)

> Cluster note, read-only; no position on which change should land.
>
> The same one-line defect in `MambaHybridModelState.add_request` now has four live fixes:
>
> - #53142 (2026-08-20) is the original report and already carries this diagnosis; #55600 is a second, independent report.
> - #53798 (2026-08-25, first PR): divides by `MambaSpec.block_size`, bound eagerly via a new `ModelState.set_kv_cache_config`; adds unit tests; has a MEMBER LGTM and `ready`/`verified`, currently `needs-rebase`.
> - This PR: `self._mamba_spec.block_size` when bound, else `cache_config.block_size`.
> - #55601 (2026-09-06): `cache_config.mamba_block_size`.
> - #55688 makes the same correction, but only under `_use_flashinfer_replayssm`; the Triton/GDN align path there still divides by `cache_config.block_size`.
>
> On current `main` all three divisors resolve to the same number, since `MambaSpec.block_size` is constructed from `cache_config.mamba_block_size`, so the open question is where the value is bound, not the arithmetic. One maintainer decision closes all of them.
>
> A model-free worker-level test that byte-compares the restored state under unequal geometry and fails on the old divisor: https://github.com/vllm-project/vllm/compare/af5357c2b90b37bd2033578bbc97d0ddfa6cc69f...MaCoredroid:vllm:p8-restore-fidelity — usable against whichever lands.
>
> AI assistance was used.

## 6. Placement (1 line)

**Post on issue #53142, not #55507** — it is the hub every PR cross-references, has no maintainer triage, is mis-labelled `quantization`, and a dedup map on one of four competing PRs reads as advocacy for it.

## 7. Quote provenance

- Ours, #53798 09-13: "Following up on the review: here is a two-commit, test-only addition atop `af5357c2b` … If useful, cherry-pick `ca1d410ae` then `bb9d7569d`; otherwise no action needed."
- mjungnickel18, #53142 body: "Happy to turn this into a PR if wanted."; 08-20: "Happy to send it as a PR."
- Davan-Etelamaki, #53142 09-05: "**I can open one** if no one else is on it."
- AndreasKaratzas [MEMBER], #53798 09-03: "LGTM -- Probably somebody else should take a look into this though as well."
- wickist, #55601 09-06: "worth a line tying `cache_config.mamba_block_size` to the same single-grid expectation".
