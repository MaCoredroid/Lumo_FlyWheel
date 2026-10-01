# Kernel benchmark summary

Times are CUDA-event medians (p10-p90) in ms over the timed repeats; one step = 48 GDN layers, B1, 32-row fixed32 tree.
Errors are max |cand - C2(float64)| over 48 layers x 28 active nodes x 48 heads (outputs) or over 48 layers x 48 heads (committed state); not a gate.

| method | mode | regions (median ms) | step median (p10-p90) | per-layer first region | transient peak | scratch alloc | out max-abs err | state max-abs err (n31) |
|---|---|---|---|---|---|---|---|---|
| lumo_fixed32 | graph | verify=12.898, commit=4.491 | 17.688 (17.182-17.841) | 0.2687 | 0.0 MiB | 132.5 MiB | 4.82e-04 | 2.62e-06 |
| lumo_fixed32 | eager | verify=12.964, commit=4.492 | 17.583 (17.417-18.147) | 0.2701 | 0.0 MiB | 132.5 MiB | 4.82e-04 | 2.62e-06 |
| naive_native_paths | graph | verify=29.710, commit=2.697 | 32.411 (32.360-32.669) | 0.6190 | 0.0 MiB | 5040.0 MiB | 4.82e-04 | 2.62e-06 |
| naive_native_paths | eager | verify=31.026, commit=2.701 | 33.726 (33.684-34.118) | 0.6464 | 0.4 MiB | 5040.0 MiB | 4.82e-04 | 2.62e-06 |
| naive_torch_node | graph | verify=52.316, commit=2.696 | 55.023 (54.943-59.267) | 1.0899 | 0.0 MiB | 5040.0 MiB | 4.82e-04 | 6.51e-07 |
| naive_torch_node | eager | verify=78.328, commit=2.698 | 81.037 (80.898-81.746) | 1.6318 | 1.0 MiB | 5040.0 MiB | 4.82e-04 | 6.51e-07 |
| treewy_author_default | graph | fused_commit_prev_and_verify=11.015 | 11.015 (10.997-11.036) | 0.2295 | 0.0 MiB | 192.8 MiB | 6.31e-04 | 5.10e-03 |
| treewy_author_default | eager | fused_commit_prev_and_verify=11.712 | 11.712 (11.696-11.737) | 0.2440 | 0.4 MiB | 192.8 MiB | 6.31e-04 | 5.10e-03 |
| weaver_aligned_local | graph | verify=4.045, commit=1.398 | 5.445 (5.418-5.498) | 0.0843 | 0.0 MiB | 31.1 MiB | 5.18e-04 | 8.55e-07 |
| weaver_aligned_local | eager | verify=7.715, commit=1.405 | 9.129 (9.087-9.214) | 0.1607 | 1.1 MiB | 31.1 MiB | 5.18e-04 | 8.55e-07 |
| weaver_author_default | graph | verify=2.068, commit=1.385 | 3.454 (3.430-3.506) | 0.0431 | 0.0 MiB | 25.1 MiB | 5.28e-04 | 5.06e-03 |
| weaver_author_default | eager | verify=4.479, commit=1.396 | 5.884 (5.853-5.933) | 0.0933 | 1.1 MiB | 25.1 MiB | 5.28e-04 | 5.06e-03 |

TreeWY (graph): verify-only (sentinel) 10.635 ms; derived commit estimate 0.380 ms; final flush 10.510 ms.

TreeWY (eager): verify-only (sentinel) 11.315 ms; derived commit estimate 0.398 ms; final flush 10.625 ms.

## LumoTree cut-state memory

N_c = 5 cut nodes [0, 1, 4, 9, 14]; allocated export buffer 96.0 MiB (shared by all layers); logical bytes written per layer 15.00 MiB; paper formula M_cut = 4*N_c*B_v*d_k = 20480 B per head/value-tile x 768 tiles = 15.00 MiB per layer.

## Sweep: authors (complete)

| treewy | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |
|---|---|---|---|---|
| chain12 | 12 | 2.107 (0.0439)  | 2.237 | 3.83e-04 |
| tree8 | 9 | 1.971 (0.0411)  | 2.119 | 4.00e-04 |
| tree16 | 17 | 6.297 (0.1312)  | 6.444 | 4.00e-04 |
| tree27 | 28 | 9.095 (0.1895)  | 9.217 | 4.31e-04 |

| weaver | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |
|---|---|---|---|---|
| chain12 | 12 | 1.707 (0.0356)  | 2.596 | 4.72e-04 |
| tree8 | 9 | 1.585 (0.0330)  | 2.627 | 3.11e-04 |
| tree16 | 17 | 1.687 (0.0352)  | 2.597 | 3.36e-04 |
| tree27 | 28 | 1.829 (0.0381)  | 2.610 | 4.72e-04 |


## Sweep: lumo_generic (complete)

| lumo_generic | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |
|---|---|---|---|---|
| chain12 | 12 | 4.163 (0.0867)  | 4.299 | 2.39e-04 |
| tree8 | 9 | 3.594 (0.0749)  | 3.774 | 2.42e-04 |
| tree16 | 17 | 7.486 (0.1560)  | 7.723 | 2.42e-04 |
| tree27 | 28 | 10.998 (0.2291)  | 11.254 | 2.42e-04 |


## Sweep: naive (complete)

| naive_native_depth | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |
|---|---|---|---|---|
| chain12 | 12 | 13.325 (0.2776)  | 15.810 | 2.39e-04 |
| tree8 | 9 | 8.377 (0.1745)  | 9.036 | 2.42e-04 |
| tree16 | 17 | 15.116 (0.3149)  | 16.221 | 2.42e-04 |
| tree27 | 28 | 25.651 (0.5344)  | 28.209 | 2.42e-04 |

| naive_torch_node | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |
|---|---|---|---|---|
| chain12 | 12 | 20.227 (0.4214)  | 31.433 | 2.39e-04 |
| tree8 | 9 | 17.363 (0.3617)  | 26.203 | 2.42e-04 |
| tree16 | 17 | 30.858 (0.6429)  | 45.360 | 2.42e-04 |
| tree27 | 28 | 49.103 (1.0230)  | 69.266 | 2.42e-04 |

