# FR13 B4 Superset CPU Gate Bind

Date: 2026-06-09

HEAD before bind: `d2fa073c`

Run root: `output/fr13_swe_verified_b4_diag_20260609T190931Z`

Reducer: `scripts/fr13_b4_superset_cpu_gate.py`

Reducer output: `output/fr13_swe_verified_b4_diag_20260609T190931Z/b4_superset_cpu_gate.json`

Command:

```bash
python3 scripts/fr13_b4_superset_cpu_gate.py \
  --run output/fr13_swe_verified_b4_diag_20260609T190931Z \
  --out output/fr13_swe_verified_b4_diag_20260609T190931Z/b4_superset_cpu_gate.json
python3 -m py_compile scripts/fr13_b4_superset_cpu_gate.py
```

## Result

The B4 accept drop is a bug in the tree verifier/target-row surface, not a drafter-quality structural floor.

### 1. Topology

Topology is present and engaged.

- Runtime `speculative_token_tree` equals the expected top-1 spine plus top-2 branch tree:
  `[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,1), (0,0,1), (0,0,0,1), (0,0,0,0,1)]`.
- Path0/top-1 spine nodes are `[0,1,3,5,7]`.
- Sibling branch nodes are `[2,4,6,8]`.
- `tree_sampler_debug.jsonl` has `193/193` `gpu_tree_metadata` rows with `mode=tree_mtp`, `tree_len=9`, `reason=ok`, `has_tree_parent_indices=true`, and `has_draft_token_indices=true`.

So the tree topology itself is not the break.

### 2. Spine Accept

Using the measured tail selected by `tree_summary.spec_drafts=620`, the tree-side committer/verifier rows classify as:

- spine path accepted: `250/620`
- branch path accepted: `132/620`
- root reject: `238/620`
- accepted-length counts: `{0:238, 1:60, 2:85, 3:74, 4:37, 5:126}`

The tree summary remains the deployed diagnostic number:

- TREE accept/event `2.024`
- Native E5 accept/event `3.794`

The native B4 artifact has aggregate spec counters and served token IDs, but no native per-event MTP draft/accept trace. Therefore exact native per-event accepted depth cannot be reconstructed CPU-only from this run. Still, the aggregate gap plus the tree trace shows the tree verifier rejects or leaves the spine too often despite the superset topology being present.

### 3. Committer vs Verifier

The committer is following the verifier scores; it is not missing the topology.

Measured-tail committer facts:

- Accepted source-index counts: source `0` = `1098`, source `1` = `132`.
- Branch/source-1 accepts: `132`.
- In `116/132` branch/source-1 accepts, the verifier assigned zero probability to the top-1/spine child.
- Reject steps total: `367`; root-step rejects: `227`.
- Reject steps with both candidates at zero probability: `332`.

This is a verifier/target-row contamination or target-row alignment break. It is not a topology break and not a committer-stops-short break: the committer is obeying rows where the tree verifier says the spine child is impossible or where a branch child is the only accepted candidate.

## Served Output

Raw served-token comparison remains:

- exact sequences: `0/16`
- first mismatch examples:
  - prompt0/sample0 position `15`: tree `5759`, native `1970`
  - prompt1/sample0 position `15`: tree `5759`, native `1970`
  - prompt2/sample0 position `21`: tree `1970`, native `3425`
  - prompt3/sample0 position `31`: tree `12305`, native `44675`

## Classification

- `topology_break=false`
- `committer_break=false`
- `verify_or_target_row_break=true`

The next GPU work, if approved by the parallel CPU workflow, should capture the verifier seam, not re-open drafter-quality or forward-cost hypotheses.
