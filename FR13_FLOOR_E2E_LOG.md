# FR13 floor e2e log

## 2026-06-07T16:28:57Z — run `output/fr13_floor_e2e_20260607T162857Z`

Commit under test before measurement tooling: `fe21cb73+`.
Measurement tooling commit: `e5f7b4f9`.

### Step 1a: Gate-2 no-bias kernel check

Command artifact: `output/fr13_floor_e2e_20260607T162857Z/gate2/no_bias_compare.json`.

Forked `_vllm_fa2_C` with no `tree_bias` was compared against the pristine stock `_vllm_fa2_C` on identical q/k/v.

| case | torch_equal | max_abs | nonzero |
|---|---:|---:|---:|
| float16 | true | 0.0 | 0 |
| bfloat16 | true | 0.0 | 0 |

Result recorded for the user table only: no-bias kernel path is byte-exact to pristine stock FA2.
