# Milestone scripts — `verifier_data/<family>/<variant>/milestones/`

Per HLD-Family-Test-Requirements §4 item 4. Each variant directory contains one
script per 5-slot milestone:

```
m1_localize.sh
m2_primary_fix.sh
m3_invariants.sh
m4_functional.sh
m5_e2e.sh
```

Each script reads `$RESULT_FILE` (the scorer's `verify_result.json`) and exits:
- `0` if the milestone passed
- `1` if it failed
- `2` if it could not be evaluated (e.g. brief missing, gold missing)

These scripts are the **declarative** surface read by LLD-06's milestone
grader. The scorer embeds the same signals under `result.milestones` and
`result.milestone_vector.slots[*].passed_bool`; the scripts exist so operators
can trace the pass/fail decision for any milestone without reading Python.

Both surfaces (scripts + embedded dict) must agree; the CI check runs the
scripts against every probe result and confirms the derived boolean matches
`milestone_vector.slots[*].passed_bool`.

## Shared implementation

All variants use **identical** scripts — milestone semantics are family-wide,
not variant-specific. The scripts symlink to `_milestones_shared/`.

## Implementation level

Per HLD §7.8 the family targets **L2 (structured)** for all five slots: every
milestone is a JSON-field read against `$RESULT_FILE`. No regex, no LLM-judge,
no executable probes.

## Integrity interaction

M3/M4/M5 are force-failed when `integrity_flag == 1` regardless of what the
other signals say. The scripts replicate this rule so an operator who inspects
the scripts alone sees the same semantics as the scorer.
