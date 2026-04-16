# release-readiness oracle bundle v0.3

Ground truth for the `release-readiness` pilot variant of the
`report-cli-markdown-evolution` scenario family in Codex-Bench's M1
quality uplift.

**Status: v0.3 — 4 of 5 quality gates complete. Only calibration remains.**

## Gate status summary

| M1 PR spec §14 gate | Status | Evidence |
|---|---|---|
| Ground-truth oracle (two-patch) | ✅ VERIFIED | `patches/solution.patch` + `patches/solution_followup.patch` apply cleanly; `probe_trap.py` empirically demonstrates partial/complete discrimination |
| Hidden test suite (5 layers + mutation-kills) | ✅ VERIFIED | 49 tests. 49/49 on round-2, 38/49 on round-1, 18/49 on broken. Clean three-state discrimination |
| Red-team suite (≥ 6 exploits, all rejected) | ✅ VERIFIED | 6 exploits. `red_team/run_all.sh` reports 6/6 correctly rejected |
| Mutation testing (≥ 0.85 kill rate) | ✅ VERIFIED | **17/17 killed, score 1.000** — see `mutation_report.json` |
| Calibration (3 models × 20 attempts) | ❌ NOT RUN | Requires model access + compute; mechanical once environment is set up |

## Discrimination matrix (verified)

| Repo state | Hidden tests result | Interpretation |
|---|---|---|
| `broken_repo/` (agent-facing start) | 18 pass, 31 fail | Markdown doesn't exist yet |
| `oracle_round1/` (partial fix) | 38 pass, 11 fail | Markdown works; latent `format_count` bug surfaces via differential + followup + zero-count property + exhaustive format_count tests |
| `oracle_round2/` (full fix) | 49 pass | Complete solution |

## What's in this bundle

```
release_readiness_oracle_bundle/
├── README.md                           ← this file
├── broken_repo/                        ← agent starting state
│                                         (ruff ✓, mypy --strict ✓, 20/23 visible tests pass)
├── oracle_round1/                      ← + markdown added; latent bug still present
├── oracle_round2/                      ← + format_count fixed upstream
├── patches/
│   ├── solution.patch                  ← diff: broken → round1 (119 lines)
│   └── solution_followup.patch         ← diff: round1 → round2 (26 lines)
├── hidden_tests/                       ← authoritative grading
│   ├── conftest.py
│   ├── test_example_based.py          (11 tests — SWE-bench-Verified layer)
│   ├── test_property_based.py         (6 tests — hypothesis invariants)
│   ├── test_differential_oracle.py    (10 tests — byte-equal vs oracle)
│   ├── test_regression_guard.py       (4 tests — PASS_TO_PASS on JSON)
│   ├── test_followup.py               (7 tests — round-2 trap catcher)
│   ├── test_mutation_kills.py         (13 tests — close mutation gaps)
│   └── _differential_fixtures.json
├── red_team/
│   ├── 01_delete_tests.sh
│   ├── 02_shadow_pytest.sh
│   ├── 03_hardcode_stub_markdown.patch
│   ├── 04_conftest_xfail_all.patch
│   ├── 05_local_zero_workaround.patch ← the key trap-case exploit
│   ├── 06_bypass_registry.patch
│   └── run_all.sh
├── mutation_test.py                    ← hand-curated mutation harness
├── mutation_report.json                ← 17/17 killed, score 1.000
├── probe_trap.py                       ← empirical trap demonstration
├── gen_differential_fixtures.py        ← regenerate _differential_fixtures.json
└── oracle_README.md                    ← maintainer docs for the trap design
```

## The six test layers (this is what exceeds public benchmarks)

| Layer | Tests | Analogue | What it catches that public benchmarks miss |
|---|---|---|---|
| Example-based | 11 | SWE-bench Verified | Nothing new — this is the baseline |
| Property-based | 6 | None in public bench | Edge cases in generated inputs |
| Differential oracle | 10 | None in public bench | Byte-equal output vs oracle |
| Regression guard | 4 | SWE-bench PASS_TO_PASS, stronger | Byte-identical JSON on hidden inputs |
| Follow-up (round 2) | 7 | None in public bench | Second-turn requirement mutation |
| Mutation-kills | 13 | None in public bench | Directly kills mutations that survive layers 1–5 |

## The trap — empirical demonstration

An agent who writes the obvious round-1 fix passes all visible tests but
emits wrong output for zero-count owners:

```
## Owner Totals
| Owner | Total |
| Alex  | 0 item  |     ← wrong: singular form for zero
```

After round-2 follow-up, the naive fix is to patch `markdown_renderer.py`
locally with `if total == 0: emit "0 items"`. This fails:
- `test_format_count_is_the_fix_site_not_markdown_renderer`
- `test_format_count_exhaustive_small_integers`
- `test_format_count_singular_and_plural_are_distinct_strings`
- Four differential fixtures with zero-count owners
- Several followup tests

The correct fix is the 3-line change in `core/formatting.py` delivered by
`solution_followup.patch`.

Reproduce: `python probe_trap.py`

## Mutation testing results (v0.3)

17 hand-curated mutations across the three target modules; **17/17 killed,
score 1.000**, well above the 0.85 floor.

Score progression:
- v0.2 (layers 1–5): 11/17 killed, score 0.647 — below floor
- v0.3 (+ `test_mutation_kills.py` with 13 tests): **17/17 killed, 1.000**

The 6 previously-surviving mutations were:
- 2× `join_with_commas` (untested in hidden layer)
- 4× registry variants (reset, sort order, register no-op, None return)

Each is now directly asserted against. See `mutation_report.json`.

## Red-team results (v0.3)

6/6 exploits correctly rejected. Exploit 05 (local zero workaround) now
fails 4 hidden tests (up from 2 in v0.2) thanks to the exhaustive
format_count checks.

## How to reproduce all gates

```bash
cd release_readiness_oracle_bundle

# Gate 1 — ground truth
python probe_trap.py

# Gate 2 — three-state discrimination (expect 18/38/49)
for state in broken_repo oracle_round1 oracle_round2; do
    cd $state && pip install -e . --quiet
    pytest /path/to/hidden_tests/ -q | tail -1
    cd ..
done

# Gate 3 — red team (expect 6/0)
bash red_team/run_all.sh

# Gate 4 — mutation (expect Score: 1.000)
python mutation_test.py
```

## Dropping into the Lumo_FlyWheel repo

```
scenario_families/report-cli-markdown-evolution/variants/release-readiness/
├── repo/                               ← copy broken_repo/*
└── oracle/
    ├── solution.patch                  ← from patches/
    ├── solution_followup.patch         ← from patches/
    └── README.md                       ← from oracle_README.md

verifier_data/report-cli-markdown-evolution/release-readiness/
├── hidden_tests/                       ← copy hidden_tests/*
├── red_team/                           ← copy red_team/*
├── mutation/
│   └── mutation_report.json            ← from this bundle
└── calibration.json                    ← NEXT (requires model access)
```

## Remaining work

Only calibration:
- 20 attempts × 3 models (local Qwen 27B, Sonnet 4.6, Opus 4.7)
- Produce `calibration.json` per M1 PR spec §12
- Target for pro tier: frontier model solves round 2 at < 25%

Calibration is not authoring work — it needs model access and ~2 hours of
compute. When that runs, the pilot variant meets all five §14 gates and is
freeze-eligible.
