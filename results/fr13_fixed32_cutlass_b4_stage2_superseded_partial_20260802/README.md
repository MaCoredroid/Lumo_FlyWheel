# B4 Stage2 superseded partial diagnostic

This reduced record preserves the useful comparator result from an interrupted
real SWE-Verified exact4 Hydra27 diagnostic. The campaign used the prior
Stage2 binary and was stopped after a rebuilt minimal-state binary received a
new identity pin. It does not qualify the rebuilt binary.

- The comparator reached its 320-call ceiling.
- All 320 returned tensors were byte-equal to the stock reference.
- No mismatching comparisons or differing bytes were observed.
- The campaign did not produce an exact4 PASS and is not timing, production,
  or floor-acceptance evidence.
- Raw prompts, responses, patches, task identifiers, logs, environment data,
  process identifiers, and container identifiers are intentionally omitted.

See `summary.json` for the machine-readable reduced record.
