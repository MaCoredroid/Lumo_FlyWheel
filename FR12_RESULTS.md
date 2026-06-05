# FR12 Results

## WY Tree Recurrence Gate

Command:

```bash
python3 scripts/fr12_wy_tree_recurrence_check.py --json-out output/fr12_wy_tree_recurrence_check.json
```

Scope:
- Uses FR10 tree descriptors from `scripts/fr10_gdn_tree_algebra_reference.py`.
- Runs the gated delta recurrence in float64 to avoid the vLLM CPU oracle's fp32 floor.
- Validates parent-inherit plus one-reflector append T against rebuilding WY on each path.
- Validates full per-node state/output against serial per-path GDN semantics.

Result:
- Verdict: `PASS` at threshold `1e-8`.
- Max append-vs-rebuild T/basis error: `0.0`.
- Max append-vs-rebuild operator error: `0.0`.
- Max homogeneous S0 map error: `3.3306690738754696e-16`.
- Max full state vs serial error: `3.0531133177191805e-16`.
- Max output vs serial error: `8.673617379884035e-18`.

Interpretation:

`TREE_ANCESTRY_T_RECURRENCE_CONFIRMED`

The FR12 WY tree recurrence is algebraically exact at float64 floor for the tested FR10 tree shapes. This validates the parent T inheritance plus append rule before building the Triton kernel.
