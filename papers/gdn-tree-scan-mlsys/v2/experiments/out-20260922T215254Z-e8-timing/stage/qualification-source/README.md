# E8 single-logits reuse — staged, no GPU execution

This isolated experiment tests one implementation change inside the fresh E1 tree route: reuse each drafter head result versus the existing legacy branch that evaluates the same head again for its spine token. It does not test fixed32 GDN, batched TAW, FA2 grouping, native baselines, or another model. No frozen E1/E7 or production source is modified. The stock image is patched inside a new disposable container using byte-pinned E1 source copies mounted read-only.

The experiment is **B1 only**: Cat10 means root + nine drafts, maximum depth five, stock TREE_ATTN, policy B remap/reorder/syncfree=1/0/1, eager, synchronous, cache off, temperature zero. Engine and API seed are both explicitly 20260921. This corrects the seed ambiguity prospectively; it does not change or relabel E1's recorded T1 deviation.

## Status and commands

Preparation only, safe on the local workstation:

```bash
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/e8_prepare.py
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/test_e8_cpu.py
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/e8_verify.py --stage papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits
```

After the parent independently reviews the staged hashes and the directory is copied intact to the DGX, this command **only prints the plan**:

```bash
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/e8_qualify.py \
  --repo /home/mark/lumo-paper-v2-20260921 \
  --run /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-YYYYMMDDTHHMMSSZ-e8-single-logits
```

Adding `--execute` is the explicit qualification execution command. It admits exactly two sequential untimed boots, ON then OFF, eight frozen prefixes per arm, max32 tokens each, and stops at the first failed gate. It refuses an existing output root/container, checks idle GPU/container state, snapshots its stage, and leaves a terminal `FAILED.json` or `QUALIFICATION_PASS.json`. There is no correctness retry, threshold repair, or timing launch. The launcher itself admits qualification only (`E8_QUALIFY=1`). Deadlines are fixed at300s for launch,1500s for health,600s for the complete eight-request workload per arm, and180s for other subprocesses. A deadline failure preserves its output and stops the owned container; no correctness retry follows. No GPU command has been run in preparation.

## What the gate establishes

The original `FR13_FIX1_SELFCHECK` checks only the selected draft token against a second `_greedy_sample` computation. E8 retains that ON check and adds missing same-input evidence in both arms:

- Each primary head is paired with a legacy head on bitwise-equal hidden input. Qualification runs **two head evaluations in both arms**; it is never timing data. Strict full-logit byte equality is required, then argmax/ordered-top-two equality. At temperature zero this verifies the greedy point-mass distribution, not stochastic probability calibration.
- A complete proposal must contain one root and four loop pairs. Both independently reconstructed ordered nine-draft candidate arrays must match the actual packed proposal. All primary/legacy/root/loop counts must reconcile to the number of completed proposals. A missing or unfinished pair cannot qualify.
- Hidden input and CPU/CUDA RNG state must remain unchanged around each head evaluation. A Torch dispatcher guard rejects writes to storage belonging to hidden input/model parameters/buffers; pointer/version census also detects model tensor replacement and tracked in-place changes. This gate does not clone all model weights or claim full-model cross-boot state identity.
- The actual loaded attention, runner, sampler and GDN modules must retain the frozen E1 hashes. Thus no state publication, KV layout, convolution, or accept-path code is changed by this ablation. Together with equal ordered candidate inputs and non-mutating head computation, this is the scoped state-preservation obligation. Existing E2 state/helper qualification is reused by exact source identity; no new universal/all-layer numerical claim is made.
- Eight direct-ID API responses must cover the exact frozen prefixes. The copied E1 recorder must close cleanly and reconcile each **complete** API stream, including prefill/mixed output and terminal clipping. All qualification request IDs map to the joiner's untimed exclusion. No support/rate from this phase is retained. Each request must have actual pure-decode exposure.
- Full runtime source/arm identity, actual engine arguments, model directory identity, head census, existing FIX1 census, and a final sealed head report are required. Sink-write errors propagate. A valid prefix of a failed process is insufficient. An exclusive first-head PID owns the report; an import-only process cannot overwrite its receipt at exit.

**Stop before timing on any required mismatch or vacuous branch.** Preserve failures, do not weaken a byte gate from its observed result, and do not silently substitute another model, backend, topology, precision, sampler or seed. Independent boots can naturally diverge: cross-boot full-stream equality is recorded diagnostically, not confused with the same-input head gate.

## Clean timing plan — not enabled by this staged driver

Only after the correctness receipt and independent review, use six fresh B1 cells in three paired blocks: OFF/ON, ON/OFF, OFF/ON. Both clean rendered source variants are already hash-identified in `manifest.json`, but this launcher refuses to execute them. The subsequent timing driver must require the exact qualification manifest/source receipt and preserve terminal sealing/failure handling before enabling clean runs.

For each cell, use the established E1 event-matched API-token/unique-physical-wall estimator, two32-token warmups on pilot0, then all eight frozen prefixes at max128 tokens, N1≥8 and ≥1 retained interval per prefix. Keep slow valid intervals; report exclusions/support and outputs. Freeze balanced order, GPU/route settings, and these limits before first timing data. Three paired blocks are the replication unit; report each block and coarse paired-block uncertainty (10,000 resamples, seed20260921), with relative contrast `mean(cell_rate_on)/mean(cell_rate_off)-1` (ratio of arm means, not mean of block ratios); recompute that ratio for each paired-block resample. No additional/favorable repeats or replacing failed/insufficient cells. Do not reuse E1's old tree observations as a control arm. Output divergence must remain explicit.

## Resources and evidence limits

The first E1 tree B1 boot logged server arguments at10:33:44 and drafter engagement at10:39:00, about five minutes16 seconds before measured workload; full tree cells also included requests and shutdown. Allow roughly **15–25 minutes for the two qualification boots**, with extra diagnostic overhead possible. Six later timing boots would be roughly **40–55 minutes** on the same idle host, if qualified; these are estimates from an existing comparable boot, not a reservation or a new timing result. Memory/image/weights remain the E1 pinned FP8 route (GPU utilization0.6, max model length16384). The launcher keeps the existing host-memory recovery requirement and read-only model mount.

`manifest.json` pins every staged file and all four emitted eagle variants. `e8_prepare.py` deliberately regenerates the stage manifest; it must not be invoked to rewrite the snapshot of an attempted run. Current production repository bytes are not substituted for the preserved E1 dependency bytes.
