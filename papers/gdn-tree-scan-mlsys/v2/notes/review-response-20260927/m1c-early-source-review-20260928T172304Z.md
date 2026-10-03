# M1-C early source review — provisional

At the single bounded remote inventory, no final M1-C package freeze/manifest was present. The executable exists and contains a populated `FROZEN_FUNCTION_SHA256` table. This note reviews that source snapshot, **not a final package**, and gives no execution or calibration approval. No further polling was performed.

Snapshot: `p0/monitor/review-response-20260927/m1c-early-reviewed-20260928T172304Z/`. The retained repository path is `repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/m1/m1_c_baseline_v1.py`; **86,495 bytes**, SHA-256 **`79c44327fe1e37989fa69ecb3de915542b6ac032ecb7c4f0dc87e08eea0d8200`**. The copied bytes match a separate remote SHA-256 read. `SNAPSHOT.json` and `PROVISIONAL-SOURCE-CONTROLS.json` retain the identity and local control outputs.

## Sealed-function findings

All **25/25** listed function-source hashes match their actual source bodies. No concrete arithmetic defect was found in the bounded source pass over these functions:

- `prep_c1` / `c1_step` / `run_c1` (lines 252–325) implement the accepted per-port preparation, FP32 persistent state, quantization of the two reduced products, unquantized prepared-key FP32 outer update, scaled query before projection, and one BF16 output store. Replay surfaces omit the readout and keep FP32 state. Author/default versus aligned normalization and beta-storage distinctions remain explicit.
- `prep_c2p` / `c2p_step` / `run_c2p` (362–416) prepare separately from pristine inputs, use float64 recurrence, and continue publication history through 1/6/11 updates. The source uses a separate float64-to-BF16 quantizer for ideal-policy storage points. This is source inspection, not a numerical verification of that quantizer or a tensor calibration.
- `build_witnesses` / `evaluate_layer_surface` (454–531) use the comparator's own states/outputs for no-op, stale, head permutation and sibling controls, with their declared applicability. Sensitivity variants rerun the recurrence. The surrounding draft aggregation requires a nonempty eligible population and zero missed eligible witnesses; it does not treat vacuous witnesses as powered evidence.
- The exact C1 declaration now includes all six fields and stored/prepared dtypes (535–631). Lumo has independent C2p only; its actual native C1 is explicitly unimplemented/unsubstituted (635–645).

The actual source's stdlib code-sharing audit reports disjoint C1/C2p function closures. Four focused AST seed controls refuse 20260928, 20260929, an unauthorized calibration seed and duplicate seeds before operand generation. These controls do not establish the unfinished final package's import/execution readiness. No Torch import, tensor generation, GPU, model, author kernel or calibration was run.

## One concrete issue in the surrounding draft runner

**Partial populations can avoid the completeness rejection used for `RESOLVED_CANDIDATE`.** This is in the still-unsealed orchestration portion, not a defect attributed to the 25 sealed numerical functions.

`cell_population_check` (958–971) compares an arbitrary requested layer subset to the same subset of the generator. At fewer than 48 layers it returns `sealed_manifest_entry=None`. The aggregate check at lines 1117–1120 rejects only a false subset equality or `sealed_manifest_entry is False`; `None` is accepted. `compute_receipt` and the CLI permit 1–48 layers, and seed authorization does not itself require both frozen calibration seeds. In addition, lines 1103–1106 synthesize declared extents for surfaces that were not evaluated. These records disclose their origin, but they cannot supply observed full-population evidence.

A focused stdlib control executed the **actual population-check AST** with the pinned manifest/generator, without tensors:

| Input | Cell count | Subset equality | Sealed entry | Aggregate population rejection |
| --- | ---: | --- | --- | --- |
| One layer, seed 20260930, 28 outputs/3 states | 1,488 | true | null | **false** |
| Full 48 layers, same seed | 71,424 | true | true | false |
| 48 layers with one output node missing | 69,120 | false | false | true |

This does **not** assert that the planned final invocation will use a partial population, or that witness checks would pass. It shows that the current population condition alone allows partial data to escape the missing-coverage disposition.

Minimal closure before the final seal: bind baseline calibration to exactly seeds **20260930/20260931**, all **48 layers**, and the declared evaluated surfaces; require actual evaluated population matches rather than fallback schedule extents for any completeness/resolution claim. Synthetic/subset controls may remain available, but their disposition must be partial/diagnostic and must not resolve the frozen calibration domain. A frozen caller plus an explicit completeness requirement can close this; no new experiments, tolerances or design expansion are needed.

## Disposition

The remaining C1 wording gap from the v3.5 review is implemented in the sealed source functions. Continue package preparation, preserve constants and unopened qualification seed, and close the narrow population issue in the final caller/reducer binding. Final manifest authentication, complete package source binding and the final execution path remain unreviewed because that package was absent. Await the parent's final handoff rather than repeatedly checking remote state.
