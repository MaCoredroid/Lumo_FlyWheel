# Four-attempt manuscript scope review

Reviewed 2026-09-28 UTC. **Bounded PASS; no material issue or required correction.** This is a read-only review of the four named manuscript/front-matter files against `p0/monitor/review-response-20260927/before-four-attempt-paper-20260928T0627Z`, the approved amendment, prospective workload scope/decisions, and the existing source-allocation note. No experiment, GPU operation, SSH, manuscript edit, or full-paper re-review was performed.

## Scope and claim checks

- `abstract.tex:2` removes the canceled fixed-build confirmation proposal and describes the existing rates as retrospectively selected development deployments. Its reported numerical results and their task/deployment scope are unchanged.
- `main.tex:336–340` accurately replaces the old workload plan with one predetermined task (`scikit-learn__scikit-learn-9288`), AR → chain MTP → SGLang EAGLE → LumoTree, and one attempt per method. It retains method-specific serving configurations/depths, pending final configuration/measurement checks, all outcomes including failures, no task replacement/tuning/repetition, and task-specific rather than benchmark-wide inference. The separate component comparison remains pending. This does not reinstate the canceled workload confirmation campaign.
- The prospective declarations agree with the amendment and `SCOPE.json`. The readiness decision binds the current scope hash and has `launch_authorized: false`. Its seed decision is explicitly declared-absent compatibility mode: the proposed scheduling seed is not a delivered request seed. The paper does not claim shared delivered seeds, a closed launch gate, completed workload attempts, or paired-seed reproducibility. Final common settings must still be checked before execution, as the text says.
- `README.md:5–7` states that the old pilot/tuning/confirmation plan is superseded and separates accepted component runs and untimed initialization from still-pending full-model qualification, timed mechanism measurements, and all four workload attempts. The remaining dated revision narrative is identified as provenance. The initialization receipt supports collection/publication only, not numerical acceptance or timing; the new status paragraph respects that distinction.
- `paper.config.yaml:3,16–20` records the pending state, approved amendment, exact workload scope, and four attempts. Retaining the original campaign plan alongside its explicit amendment does not authorize superseded phases.
- The four-file diff changes scope/status prose only. No existing quantitative result or result-file source is modified by these edits. Independent render/build and result-file hashing remain the parent's separate QA, not a claim of this review.

## Memory wording

`main.tex:215–221` is unchanged by this scope amendment and is consistent with `m1-source-live-allocation-review.md`. The equation `4 N_c B_v d_k` is expressly a logical export payload for one head/value tile. The following text distinguishes the reusable full-matrix-per-node scratch allocation from which cut states are written, and distinguishes carried replay state from shape-dependent operands/outputs. It does not turn logical publication volume into resident allocation, measured traffic, whole-process peak, or an SRAM-capacity/no-spill claim. It does not derive a smaller runtime memory floor or a speed claim from these quantities. The allocation note's overlapping lifetimes and lack of a measured peak remain compatible with this prose.

## Reviewed SHA-256 bindings

| Current file | SHA-256 |
|---|---|
| `abstract.tex` | `bce4fc01561b3fdf2d1cff0e14acf0cef00e2bd6f8581a360361b9a61015b2ea` |
| `main.tex` | `fffb02ca7c1fcd98a75dddf2510ab6585582003f62c8a10f5a2c30adfb5a1be9` |
| `README.md` | `c454221fd8d836d81ad1e42a2416825a4bde0c26faf780c24417c69cf77cfb8c` |
| `paper.config.yaml` | `33871cdc4d4b3362bf334a5a63fb16f62f0accae25e0b6543cd6760ac90c30ad` |
| `plan/2026-09-27-four-attempt-workload-amendment.md` | `27bafdf7ec176ded0868b8b420784acc14e522df0b3639f44edc3c4acacc39b4` |
| `experiments/review-response-20260927/workload-case-study-v1/SCOPE.json` | `f9a9d7cd4a0be3da43a9f6a92f6125a199af96515707863aafee4acb9e3ac822` |
| `p0/monitor/review-response-20260927/WORKLOAD-PROSPECTIVE-READINESS-DECISIONS.json` | `e0835abed7061b7d264ccc2144c3073f6dc0037f4c8d88ce5062d981585c861a` |
| `notes/review-response-20260927/m1-source-live-allocation-review.md` | `971bd1b42171b86fbe11a1bde330a333fb202884cad8efddd6b6a45f145ef448` |

Snapshot hashes, in the same four-file order (abstract/main/README/config):

```text
c1cb90f8d7f93808a2b7ea1805186304f0f3ad863b4af08380a9e0da02a97aed
fb1571cf1dd4fb67d71d3122dc63dc9f1928dc2b026795f85cd43b1939c40013
f30ccc93ad5133af52190167c8e1b6b1314c473ede2bfb28726a21b6cc32eff8
8cf82a5ed716b76e3debf73f56b0f14920dc6aa686a8d352021d1ace71d558a2
```
