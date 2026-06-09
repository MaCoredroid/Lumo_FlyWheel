# FR13 Corruption Gate Bind

Date: 2026-06-09

HEAD before bind: `e09abd5b`

File: `scripts/fr13_corruption_gate.py`

Status: committed as the CPU reducer for the three-arm tree-vs-native-vs-native-noise corruption gate.

Validation in this turn:

```bash
python3 -m py_compile scripts/fr13_corruption_gate.py
```

Result: compile passed.

Purpose:

- Compare TREE vs native served tokens with native self-noise masking.
- Enforce bag-TV floor.
- Enforce accept/event superset (`tree >= native`, within slack).
- Fail closed on prompt identity mismatch, missing arms, empty records, unpaired record keys, or missing accept/event.
- Optionally use per-event tree/native traces when present.

Important limitation for the next measurement:

- The script expects `fr13_e2e_measure.py`-style arm directories containing `tree_greedy_probe.json`, `native_greedy_probe.json`, and `native_greedy_probe.json` for native-noise.
- The prior SWE B4 diagnostic used `fr12_deliverable_swe4_probe.py` filenames (`tree_b4_swe4_probe.json`, `native_b4_swe4_probe.json`), so the next three-arm run must either produce the expected filenames directly or normalize the fresh run directory before invoking the gate.
