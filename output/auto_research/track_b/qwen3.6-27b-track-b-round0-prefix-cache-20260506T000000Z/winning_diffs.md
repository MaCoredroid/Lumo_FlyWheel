# Winning Diffs

## Candidate 001 - Serving Batching / Concurrency

Status: accepted for speed exploration, not promoted.

Speed evidence:

- `candidates/001/throughput.json`
- serial warm decode: `6.966699403752739 tok/s`
- 4-way aggregate warm decode: `24.373630400693298 tok/s`
- target: `15.0 tok/s`
- prefix-cache hit rate during measurement: `0.7952931629133698`

Quality evidence:

- `candidates/001/b1_result.json`
- B-1 strong-equivalence serial-vs-batched match rate: `1.0`
- B-2: not run
- B-3: not run

Implementation surface:

- No model weights changed.
- No kernel math changed.
- Candidate uses existing vLLM `max_num_seqs=4` scheduler shape to amortize warm decode across concurrent cache-hit requests.

Promotion blocker:

Full B-2/B-3 quality gates remain required before this can be considered a promoted Track B winner.

## Candidate 004 - Controller-Promoted Auto-Research Winner

Status: promoted by the Track B controller.

Authoring path:

- candidate was written by `codex exec` through `scripts/run_track_b_loop.py`
- parent/controller owned measurement and gates
- candidate directory: `candidates/004`

Speed evidence:

- `candidates/004/throughput.json`
- serial warm decode: `7.032551575325713 tok/s`
- 3-way aggregate warm decode: `19.471946265799282 tok/s`
- target: `15.0 tok/s`
- speedup over 7.5 tok/s baseline: `2.596259502106571x`

Quality evidence:

- B-1: `candidates/004/b1_result.json`
- B-2: `candidates/004/b2_result.json`
- B-3: `candidates/004/b3_result.json`
- all serial-vs-batched workload-equivalence gates passed at match rate `1.0`

Why this won:

- Concurrency 8 and 6 both cleared speed but failed B-3 serial-vs-batched equivalence.
- Concurrency 3 still cleared the 2x target while preserving deterministic workload-derived outputs in the controller gates.
