# E1 as-executed deviations from the frozen settings (append-only; the frozen plan, cells order and campaign snapshots are NOT rewritten)

## T1 — engine seed scope differs between the tree arm and the native arms (recorded 2026-09-22T10:45:17Z)
Source: independent first-tree-timing review `p0/monitor/e1-first-tree-timing-redteam.md` (sha 21148b1095b8f5e464b3194e58d57436d414b32ddacc2a079a4e47793549a2c2); verified on the campaign's own artifacts (`out-20260922T100028Z-e1-18cells/`).

| Cell (sealed at the time of writing) | container Cmd `--seed` | container env SEED (consumed host-side by the launchers; never forwarded) | API request seed(s) |
|---|---|---|---|
| `cell_01_b1_native-5_B1_a1` | 20260921 | not passed | [20260921] |
| `cell_02_b1_native-5_B4_a1` | 20260921 | not passed | [20260921] |
| `cell_03_b1_native-11_B4_a1` | 20260921 | not passed | [20260921] |
| `cell_04_b1_native-11_B1_a1` | 20260921 | not passed | [20260921] |
| `cell_05_b1_tree_B1_a1` | absent (vLLM default 0; server log: seed 0) | not passed | [20260921] |

- Mechanism (source): the cell driver passes `SEED=20260921` to both launchers; `e1/e1_native_launch.v2.sh` forwards it (`--seed '${SEED:-0}'`, line 253), whereas `e7a/e7a_capture_launch.v7.sh` (= v6 + explicit `--no-enable-prefix-caching`) carries NO `--seed` handling at all, so the tree arm's vLLM EngineCore runs with the vLLM default engine seed 0 while the native arms run with engine seed 20260921. The API request seed is 20260921 for every request in every arm. The frozen settings table (`e1/e1_cells.v1.json`, `E1_FREEZE.md`) lists `seed 20260921`; the tree arm's engine seed does not comply with that line. The same tree launcher lineage (v3–v7) served every E2/E7 tree boot in this worktree, so those pilots also ran with engine seed 0.
- Later tree cells (6, 7, 8, 15, 16) run the same launcher v7 from the same campaign snapshot and therefore carry the same engine-seed deviation; they are not re-listed per cell.
- Bounded interpretation (the reviewer's source chain, not a claim of equivalence): the workload is temperature 0 (greedy) with `return_tokens_as_token_ids`; the tree committer's choice among equal-overlap duplicate draft sources uses its own explicit fresh generator, not the global engine RNG, so no corrupted sampling or corrupted rate evidence is currently indicated and no replacement boot is scheduled. I do not claim identical engine seeds across arms, nor full compliance with every frozen setting.
- Handling: the frozen 18-cell campaign continues unchanged (serial, same snapshot); this file and STATUS carry the deviation; the parent integrates it into the final paper/artifacts. Any future tree launcher version must forward `SEED` explicitly and the verifier must check `--seed` in the Cmd (not done retroactively for this campaign).
