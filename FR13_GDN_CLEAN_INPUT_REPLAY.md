# FR13 GDN Clean-Input Conv Replay

Date: 2026-06-07
Commit under test: cff79759 plus replay-tool update

## 2026-06-08 update: ex2.approx SiLU replica

Commit under test: `cb669425`

Native source read:

- vLLM `causal_conv1d_update` source applies activation as
  `acc / (1 + tl.exp(-acc))`.
- Cached production PTX for `_causal_conv1d_update_kernel` lowers that path to
  bf16 tap multiplies, fp32 accumulation, `ex2.approx.f32`, `div.full.f32`,
  and `cvt.rn.bf16x2.f32`.

The tree conv path now uses our own Triton helper,
`triton_ex2_silu_bf16`, for SiLU. This keeps the verifier in our tree kernel
path; it does not call native `causal_conv1d_update`.

Replay command:

```bash
docker run -i --rm --gpus all --ipc=host --entrypoint /usr/bin/python3 \
  -v "$PWD:/workspace" -w /workspace \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  scripts/fr13_ex2_silu_replay.py \
  --tree \
    output/fr13_gdn_conv_multilayer_capture_20260607T193044Z/tree/logs/gdn_conv_replay.call0.pt \
    output/fr13_gdn_l45_fullstate_20260607T175434Z/tree/logs/gdn_l45.call0.pt \
  --native \
    output/fr13_gdn_conv_multilayer_capture_20260607T193044Z/native/logs/gdn_conv_replay.call0.pt \
    output/fr13_gdn_l45_fullstate_20260607T175434Z/native/logs/gdn_l45.call0.pt \
  --out output/fr13_clean_input_l0_l45_ex2_replay.json
```

Replay artifact:

- `output/fr13_clean_input_l0_l45_ex2_replay.json`

Clean-input result:

| layer | pre_conv max_abs | captured tree conv max_abs | ex2 helper conv max_abs | ex2 helper nonzero |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0.0 | 0.0 | 0 |
| 45 | 0.0 | 0.0009765625 | 0.0 | 0 |

Clean aggregate:

```text
layers=[0,45]
max_abs=0.0
nonzero=0
```

Status: the clean-input conv edge is fixed offline. No live full-ladder pass is
claimed here; the required next step is one strict live ladder
(spine+branches+logits+gate-2 no regression).

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

This CPU-torch/tie-rule section is retained as the negative control that forced
the ex2.approx helper. It is superseded by the 2026-06-08 clean-input replay
above.
