# FR13 B4 Batch-Invariant Gate Blocked

Date: 2026-06-09

Run root: `output/fr13_b4_batch_invariant_gate_20260609T215506Z`

## Objective

Run the B=4 corruption gate with batch-invariant mode enabled on tree, native, and native-noise arms:

- `VLLM_BATCH_INVARIANT=1`
- `LUMO_BATCH_INVARIANT_VLLM=1`
- B=4, `MAX_NUM_SEQS=4`
- tree `TREE_ATTN/tree_mtp`
- native `FLASH_ATTN/naive_mtp`

This was intended to decide whether the eager B=4 non-deterministic divergence is a batch-invariance failure.

## Prerequisite Fix

Commit `92854ab9` made the launchers capable of passing the requested flags:

- `scripts/fr13_launch_forked_fa2_tree_server.sh` now accepts `BATCH_INVARIANT`.
- `scripts/fr10_launch_speed_server.sh` now forwards `LUMO_BATCH_INVARIANT_VLLM`.

## Result

The tree arm did not reach health. vLLM rejects built-in batch-invariant mode with `TREE_ATTN` before model serving starts.

Artifact:

- `output/fr13_b4_batch_invariant_gate_20260609T215506Z/tree/docker_failed_batch_invariant.log`

Relevant error:

```text
RuntimeError: VLLM batch_invariant mode requires an attention backend in ['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'TREE_ATTN'.
```

The container environment confirmed the requested flags were live before failure:

```text
VLLM_BATCH_INVARIANT=1
LUMO_BATCH_INVARIANT_VLLM=1
FR10_DECODE_MODE_DEFAULT=tree_mtp
FR10_METRICS=0
```

## Bound Verdict

The requested three-arm batch-invariant corruption gate is blocked by backend compatibility. It cannot currently answer YES/NO for TREE_ATTN because vLLM's built-in batch-invariant path refuses the tree attention backend.

Native-only arms were intentionally not run because they would not answer whether the tree verifier's B=4 losses drop under batch-invariant TREE_ATTN.

The same-config repeat check was also not rerun under batch-invariant mode because the tree batch-invariant server cannot boot. The prior `e263a45b` evidence still shows same-seed B=4 non-determinism across capture runs, but this bind does not add a new repeat measurement.

Host memory was recovered after stopping the failed tree arm; no `fr13-bi-*` containers remained running.

## Next Scope

To run this test, the tree path needs one of:

- a TREE_ATTN-compatible batch-invariant mode,
- a FLASH_ATTN-compatible tree verify path for this gate, or
- a narrower custom batch-invariance toggle for the suspected GDN/tree kernels that does not trip vLLM's global attention-backend guard.

