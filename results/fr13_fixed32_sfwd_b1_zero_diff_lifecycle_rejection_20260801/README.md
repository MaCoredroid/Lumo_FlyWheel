# SFWD real-B1 zero-diff lifecycle rejection

This artifact records the authenticated real SWE-Verified B1 SFWD byte gate
launched at `2026-08-01T11:45:54Z`.

- Source commit: `f35b73725a5a35cd44f77ac77309201156a34dae`
- Task: `astropy__astropy-12907`
- Run root:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T114554Z`
- Wrapper result: exit code 15
- Candidate: `fixed32_sfwd_state_fusion_v1`
- Physical rows per request: 32

The candidate completed a reference-first byte comparison on the first
authenticated real decode. All 48 GDN layers passed:

- 48 of 48 comparator records have `status=pass` and `zero_diff=true`.
- Each layer compares 655,360 bytes of `conv_out` and 737,280 bytes of
  `commit_source_stage`.
- Total compared bytes: 66,846,720.
- Differing bytes: 0.
- The reference result was always served.

The task did not complete. After the comparisons, the engine failed closed in
`sample_tokens` with:

`RuntimeError: FR13 fixed32 KV16 compact/full row-map drift`

The SWE harness subsequently rejected real-task provenance because engine
completion metrics did not reconcile. This is therefore a lifecycle rejection,
not a completed byte gate. The candidate is not production eligible, and this
run supplies no timing, TPS, acceptance, confidence-bound, or hardware-floor
result.

## Evidence bindings

- Byte comparator JSONL SHA256:
  `a9fd9d2d223e15aeeb8a899539eea1d9cfbcc9afe13ec0cab2a967a012835674`
- Live-pass JSON SHA256:
  `7ccfaf5cc907909b0646b752b94027e250b234a3b98bf461de61e6ae70f31782`
- Arm runlog SHA256:
  `f915132d2ab0006607cd80cef765bf70b0a77416163d315c961bd2973f595e3c`
- B1 diagnostic manifest SHA256:
  `fb4a790715612dd18dbdeb464d33e4852fcf9d7469a368b2ee8ed2a2d1faaa2d`
- Source-head record SHA256:
  `c21d8c5e139f53cb19c1261b33ed69f9df5bb57f3fd680532e2c76b691a5a8bc`
- Docker-after-tasks log SHA256:
  `672d0223e1e7d857ad65d61d0382bdfb940bd2460793cc67d6bf5df054224ad4`

