# E7a step 2 — pilot-derived freeze (REQUIRED before any confirmation boot)

Status: FROZEN 2026-09-22T01:36:16Z — structured record `PILOT_FREEZE.json` sha256 21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce (validated by `check_pilot_freeze.py`); pilot harness of record `out-20260922T013313Z-e7a-fresh-pilot-v2/` summary.json sha256 7744f133df77bf07c0ff9dff31609a462033c5d6e0c14fb86d191bd07d81c4a1, SUMMARY.md sha256 07cb3c1d2be846c186d8822d81a251e450d698869745733698726e6450c84144; 8 hash-bound pilot captures (p072 p017 p015 p021 p085 p095 p058 p083), inspector 59c5b2b9…; criteria in `PILOT_FREEZE.criteria.json`. The five items below are FILLED by the structured record (this file is the human-readable declaration; the JSON is the gate).

Pre-declared (independent of pilot results; already in STATUS "Fresh-prefix pilot design decisions"):
- Unit of analysis / resampling unit: the frozen prefix (one boot, one request, one one-shot payload per prefix); never
  the layer, file, or operand row. Confirmation resamples prefixes only (32 disjoint frozen prefixes, same layer 62 unless
  the freeze below says otherwise).
- Reference conventions (never merged): fp64 serial oracle for algebra; pinned native spec-update kernel (fp32 store) for
  the production comparison; one-token decode kernel reported separately with its declared bf16 seams.

To be filled FROM the pilot (8 verified-fresh payloads; each value cites `out-…/SUMMARY.md` rows):
1. **Candidate selection** — which realization(s) go to confirmation (scan / TreeWY-fs / Bole-Neumann; fp32-ieee store
   only, or also bf16 store) and why (pilot agreement vs the pinned native reference; any candidate whose pilot worst-case
   exceeds the tolerance below is dropped BEFORE confirmation, not after).
2. **Tolerances** — per comparison class, from the pilot's observed distributions (max |Δ| and ULP on significant
   elements over the 8 prefixes): (a) fp32/ieee vs fp64 oracle; (b) production scan fp32 store vs native spec-update
   (expected bit-identical: tolerance 0 ULP, integer view); (c) bf16 store agreement floor; (d) B/C compact-commit state
   vs native. State the pilot max and the frozen tolerance (pilot max × declared margin, margin declared here).
3. **Unacceptable decision changes** — outcomes that would change the paper's claims: any fp32 row not bit-identical
   under (b); any bf16 agreement below the frozen floor; any non-finite cell; any tf32 row promoted; any change of the
   winner ordering between realizations; provenance FAIL on any confirmation prefix (counted as a failure, never dropped).
4. **Timing precision target** — from the pilot's CUDA-event medians: target relative half-width (e.g. ≤ 5 % of the
   median over repeats) and the probe-drift ceiling (≤ 5 %, as in v3); confirmation timing rows that miss the target are
   reported as imprecise, not excluded. Timing stays "shared-device diagnostic" unless the run's own inventory shows only
   the owned container PID.
5. **Cost / variance** — pilot boot cost (health latency, request latency, payload size), per-prefix harness cost, and
   the observed prefix-to-prefix variance of each metric; these justify (or shrink) the 32-prefix plan and set the
   expected confirmation duration.

Sign-off: the freeze is recorded by editing the status line above, adding the utc and the sha256 of the pilot SUMMARY
files it was derived from; the confirmation loop copies this file into its run root before the first boot.
