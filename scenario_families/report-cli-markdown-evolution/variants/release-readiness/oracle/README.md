# Oracle — release-readiness variant

This directory contains the ground-truth solution used to grade agents and
to gate verifier quality in CI. Never mount these files into an agent
container.

## Files

- `solution.patch` — applied to `broken_repo/`, produces a round-1-complete
  solution (markdown renderer added, registered, docs updated).
- `solution_followup.patch` — applied on top of `solution.patch`, fixes the
  latent bug in `core/formatting.py::format_count`.
- `README.md` — this file. Read before modifying anything in this directory.

## Why two patches?

The variant uses a two-phase grading structure:

1. **Round 1** — agent gets `AGENTS.md`, told to add markdown support.
   Agent produces a solution. Hidden tests (example-based, property-based,
   differential-oracle, regression-guard) run. If all round-1 tests pass,
   orchestrator injects the follow-up brief and starts round 2.

2. **Round 2** — agent gets a follow-up: "include owners with zero current
   items; pluralization must be right everywhere." Agent produces a second
   diff. Round-2 hidden tests run.

This structure is not arbitrary. It exists because this variant deliberately
plants a **latent bug** in `core/formatting.py::format_count`:

```python
# broken state — shipped to the agent
def format_count(n, *, singular, plural):
    if n == 1:
        return f"{n} {singular}"
    elif n > 1:
        return f"{n} {plural}"
    else:
        return f"{n} {singular}"   # <-- bug: "0 item" instead of "0 items"
```

The JSON renderer never calls `format_count` with `n == 0` because it emits
raw integers. So a full visible test suite passes even with the bug present.
The round-1 markdown oracle deliberately leaves this bug in place — because
a well-written agent solution should also leave it alone if it works for
all round-1 tests.

Round 2 is where the bug has to be addressed. The follow-up brief says
"pluralization must be right everywhere." Here the agent faces a design
choice:

- **Wrong answer (patch locally)**: special-case `if total == 0` inside
  `markdown_renderer.py` so the markdown path emits "0 items". This makes
  the markdown test pass but does nothing for the JSON renderer, which
  now also needs to produce "0 items" strings (per the follow-up brief's
  "everywhere" clause). It also fails `test_format_count_zero_uses_plural`
  and `test_format_count_is_the_fix_site_not_markdown_renderer` — direct
  unit tests on the upstream function.

- **Right answer (`solution_followup.patch`)**: a 3-line change to
  `format_count` itself. Every consumer benefits. Every test passes.

The `red_team/05_local_zero_workaround.patch` exploit tests exactly the
wrong-answer case. Confirm it fails the follow-up tests whenever you
regenerate hidden tests.

## How the trap was empirically verified

`probe_trap.py` in the bundle root runs the CLI in both oracle states on
the same input (two active owners + one zero-count known owner) and asserts
distinct output:

```
oracle_round1 output: | Alex  | 0 item  |    ← bug visible
oracle_round2 output: | Alex  | 0 items |    ← bug fixed
```

If you change the implementation of `core/formatting.py` or the markdown
renderer's table formatting, rerun `probe_trap.py` and confirm the
discrimination still holds. If it doesn't, either the trap no longer works
(fix the oracle) or the hidden tests are no longer discriminating (fix
the tests).

## Maintaining this oracle

Steps for any future modification:

1. Modify `broken_repo/` only as needed for your change.
2. Regenerate `solution.patch` with:
   ```
   diff -urN --exclude='*.egg-info' --exclude='__pycache__' \
       broken_repo oracle_round1 > patches/solution.patch
   ```
3. Regenerate `solution_followup.patch` the same way against
   `oracle_round2`.
4. Regenerate differential fixtures with `gen_differential_fixtures.py`
   against the round-2 oracle.
5. Rerun hidden tests against all three states; confirm:
   - Broken: many failures, no full pass
   - Round 1: 9 specific failures (4 differential, 4 followup, 1 property)
   - Round 2: 36 passes
6. Rerun `red_team/run_all.sh`; confirm 6/6 exploits rejected.
7. Only then regenerate `mutation_report.json` and `calibration.json`.

## Expected grading results (reference)

| Repo state | visible tests | hidden tests (36) |
|---|---|---|
| broken_repo/ | 20/23 | 7/36 |
| oracle_round1/ | 23/23 | 27/36 |
| oracle_round2/ | 23/23 | 36/36 |
