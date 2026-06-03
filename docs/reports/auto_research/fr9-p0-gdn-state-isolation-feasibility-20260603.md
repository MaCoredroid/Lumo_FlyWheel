# FR9 P0 GDN state isolation feasibility

Date: 2026-06-03 UTC

Branch: `fr9-spine2-lossless-winner`

Decision: **P0 did not pass. Stop before building isolation or selector.**

## Operator Gate

Addendum 11 requires a cheap feasibility gate before writing the co-design spec or implementation:

1. snapshot/restore a GDN/Mamba recurrent state;
2. run a batch-1 isolated single-step forward seeded from that state;
3. prove it is bit-reproducible and independent of co-scheduled rows.

The live container is `lumo-vllm-track-b-suffix`, image `lumo-flywheel-vllm:26.01-py3-v0.19.0`, with vLLM `0.19.0`.

## What exists

vLLM has an internal device-to-device Mamba/GDN state-copy helper, but it is scoped to scheduler-owned cache blocks:

- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py:65` defines `MambaCopyBuffers`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py:97` builds copy metadata from a source block index and destination block index within a request's `block_ids`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py:136` launches the batch memcpy.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py:147` `preprocess_mamba` computes the current running-state block from scheduler token counts and copies previous state to that scheduler-selected block.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py:222` `postprocess_mamba` copies accepted speculative state into aligned blocks after the verifier decides accepted-token count.

The GPU model runner owns the live state index and copy buffers:

- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:860` has `self.mamba_state_idx`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:958` has `_get_mamba_copy_bufs()`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:3957` calls `preprocess_mamba`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:4198` updates states after sampling, then `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:4201` calls the repo-owned debug exporter.

The repo-owned P2B hook can export recurrent state for debugging, but it is one-way export. It does not provide restore or seeded replay.

## What is missing

I did not find a vLLM 0.19 data-plane primitive that can:

- allocate an arbitrary durable snapshot slot for a public request's GDN/Mamba state;
- restore that snapshot into a new hidden branch without routing through normal request scheduling;
- run exactly one target decode step from that restored state in a guaranteed batch-1 lane;
- return logits and the resulting recurrent state while leaving public state untouched.

The relevant forward path is not exposed as an independent callable. It is assembled inside `GPUModelRunner.execute_model` from a full `SchedulerOutput`, active `InputBatch`, scheduler-owned block tables, attention metadata, slot mappings, and sampling state:

- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:3987` derives slot mappings from the current scheduler batch.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:3998` builds attention/GDN metadata from that batch.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:4021` preprocesses model inputs from the same scheduler output.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:4063` runs `_model_forward`.
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:4099` computes logits.

The dev `/collective_rpc` endpoint is not enough for P0. It accepts method names plus string args only (`api_router.py:24-45`), and `apply_model` passes only the model module, not the live `GPUModelRunner` with request caches and scheduler state (`worker_base.py:115-117`). That makes it unsuitable for a real snapshot/restore/single-step data-plane probe.

## P0 verdict

P0(a) is only partially available: device-to-device state copy exists among scheduler-owned Mamba cache blocks, and CPU debug export exists. A complete snapshot/restore primitive for an arbitrary public checkpoint does not exist.

P0(b) is not available: there is no cheap callable route in vLLM 0.19 to run an isolated batch-1 single-step forward seeded from that checkpoint and prove independence from co-scheduled rows.

Therefore I cannot honestly mark the primitive feasible. Building the canonical-lane plus MDSP/SpecHub selector now would be speculative architecture work without the load-bearing state primitive. Per Addendum 11, implementation stops here.

## Required next step

A future attempt needs an explicit vLLM data-plane extension before selector work:

- reserve scratch/snapshot recurrent-state slots independent of normal request block scheduling;
- copy public `S_i` into those slots and restore it into a hidden branch slot;
- build a one-token `SchedulerOutput`/metadata path or worker method that runs with `num_reqs=1` and cannot be merged with public or unrelated requests;
- return full logits plus the produced recurrent state;
- prove bit reproducibility and batch independence before enabling any superset publication.

This may reuse the existing Mamba copy kernel for state movement, but the seeded isolated forward path is new vLLM scheduler/model-runner data-plane work. No production fail-closed behavior was weakened, and no speed or losslessness claim is made for the co-designed selector path.
