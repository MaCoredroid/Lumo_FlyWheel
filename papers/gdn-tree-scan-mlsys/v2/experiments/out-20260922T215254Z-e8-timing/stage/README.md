# Conditional E8 single-logits timing v1 — prospective freeze

This stage has no timing data and does not authorize GPU execution. It admits exactly six fresh B1 cells after both qualification arms pass the unchanged source/whole-logit/ordered-candidate/state-obligation/API gate in `../e8-single-logits-qualification-v2`, manifest `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`. The first qualification attempt and its pre-container launcher failure remain preserved separately. This stage copies all qualified source bytes without modifying the attempted directories.

The only arm change is the source-controlled single-logits ON/OFF branch. ON executes five head evaluations per depth-five proposal; OFF executes ten. The unchanged E8 helper records small in-memory head counters and one final receipt. Both arms disable the existing FIX1 selfcheck, paired-logit cloning/comparison, mutation dispatcher and heavy captures. Their actual loaded `eagle.py` hashes must match the qualified clean variants. The other four loaded modules and six mounted repository dependencies remain pinned. The engine and API seed are both 20260921. Cat10 is root plus nine drafts; TREE_ATTN, flat-map policyB (1/0/1), synchronous scheduling, eager, prefix cache off, temperature0, model/image and utilization remain fixed.

Each cell boots a fresh server, executes two shape-matched 32-token warmup requests on pilot prefix0, then the eight frozen prefixes sequentially at 128 tokens. These ten complete direct API-ID streams must reconcile with all emitted recorder rows, including prefill/mixed rows and declared terminal clipping. Primary throughput counts only API-bound tokens on matched unique pure-decode physical intervals with actual B1 occupancy and consecutive same-request support. Every valid slow interval remains in primary support; the1.5s cutoff is diagnostic. Coverage requires at least8 intervals overall and at least1 for each of the8 timed prefixes. Insufficient support is retained with no rate and no replacement.

Cell order is OFF/ON, ON/OFF, OFF/ON across three paired blocks: counterbalanced, not perfectly balanced. **Dated pre-data supersession, 2026-09-22:** the primary is the mean of the three paired relative differences `(rate_on_b-rate_off_b)/rate_off_b`. This supersedes only the disabled provisional ratio-of-means primary carried unchanged into the qualification manifest. The ratio of arm means minus1 is diagnostic. The paired block is the resampling unit;10,000 draws use `random.Random(20260921)`, three `randrange(3)` indices per draw, and the2.5/97.5 percentiles with linear interpolation at `(n-1)*p`. Three blocks yield coarse uncertainty. All per-block rates, support and exclusions are retained. Any missing, failed or insufficient cell makes the aggregate estimate/interval unavailable; no subset estimate, precision-driven extension, replacement or favorable retry is allowed. Wide, negative and null results remain reportable.

GPU ownership is sampled before, every5seconds during, and after the complete warmup+timed workload. Each sample preserves exact query outputs, labeled container ID and host PID sets from `docker top`; every observed GPU PID must belong to that single container. A query failure, absent owner/GPU PID or foreign process/container invalidates the cell and aborts the campaign without replacement. This is occasional telemetry, not proof against short-lived contention between samples. The existing launcher retains its memory checks; this added telemetry does not run inference. Each workload has a900s caller deadline; launch and health deadlines are300s and1500s. Ownership samples are taken identically in both arms.

The driver validates actual mounted/loaded sources and configuration before issuing requests, runs only its copied immutable snapshot, refuses an existing run root, and seals a cell only after cleanup, source revalidation, full API join, clean head census and ownership validation. All raw evidence hashes and the exact qualification receipt hash are included in each terminal seal. The aggregate rejects preliminary results or changed evidence. Output-stream comparisons are descriptive only:24 within-pair and48 within-arm across-block prefix comparisons when all cells exist. Matched prefixes do not establish matched continuations; divergence never changes retention or correctness thresholds.

Reproduce source staging and CPU checks without a GPU:

```sh
python3 -B prepare.py
python3 -B test_cpu.py
# Linux only: actual outer and inner shell, strict Docker/Python/vLLM stubs.
CUDA_VISIBLE_DEVICES='' python3 -B test_launcher_execution.py
```

The shell smoke explicitly stubs the absent live qualification receipt **only for command plumbing**, after real source verification. It cannot authorize timing. Separate CPU controls exercise the actual missing/failed/wrong-source qualification refusal. The original qualification gates and their real receipts remain mandatory.

Plan only (default):

```sh
python3 -B timing_driver.py --repo /home/mark/lumo-paper-v2-20260921 --run /absolute/new/run
```

Only after independent harness review and the parent's separate execution handoff:

```sh
python3 -B timing_driver.py --repo /home/mark/lumo-paper-v2-20260921 --run /absolute/new/run --qualification /absolute/completed/qualification-v2-run --execute
```

No automatic execution or retry is provided. Six comparable fresh B1 tree boots are expected to take roughly45–70minutes from prior E1 boot and workload durations; this is a planning estimate, not a measurement or a deadline extension.
