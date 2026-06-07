# FR13 GDN Clean-Input Conv Replay

Date: 2026-06-07
Commit under test: cff79759 plus replay-tool update

## Method

Only captures with exact `pre_conv_path0 == 0.0` are valid for causal-conv
kernel tuning. Contaminated deeper captures are still reported, but excluded
from aggregate variant selection with:

```bash
python3 scripts/fr13_conv_replay_multilayer.py --require-clean-input ...
```

Clean-input replay target:

- L0 from `output/fr13_gdn_conv_multilayer_capture_20260607T193044Z`
- L45 from `output/fr13_gdn_l45_fullstate_20260607T175434Z`

Replay artifact:

- `output/fr13_clean_input_l0_l45_replay.json`

## Results

| layer | pre_conv max_abs | window max_abs | bf16 tap max_abs | captured conv max_abs | per-layer best |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | `bf16,0123,torch,torch_bf16` |
| 45 | 0.0 | 0.0 | 0.0 | 0.0009765625 | `bf16,0123,torch,tie_positive_down` |

Clean aggregate best remains nonzero:

```text
tap_product=bf16, order=0123, silu=torch, store=torch_bf16
max_abs=0.0009765625, nonzero=1
first_mismatch=[6,430], lhs=0.177734375, rhs=0.1767578125
```

The apparent L45-only fix (`tie_positive_down`) is not valid globally: it fixes
the L45 midpoint but regresses L0 (`max_abs=0.125`, `nonzero=16`) on positive
midpoints where native rounds up.

## Classification

The remaining mismatch is an activation/bf16 midpoint case. CPU torch produces
an exact midpoint for the L45 element:

```text
layer=45 row=0 col=430
acc=0.30756378173828125
torch_silu=0.17724609375
bf16 rounded=0.177734375
bf16 previous=0.1767578125
native=0.1767578125
```

L0 also has positive midpoint cases, but native rounds those up. A simple
positive-midpoint tie rule is therefore not a valid production fix.

No live ladder was run. Offline clean-input conv remains nonzero.
