# Candidate A

Strategy: normalize separator-heavy labels in `src/report_filters/cli.py`.

Strength:
- quickly makes the visible CLI regression pass

Weakness:
- leaves `service.compile_filters(...)` unchanged, so non-CLI callers
  continue to emit unnormalized report keys
