# CFWD all-parent paired B1 timing source

Status: **ready to run; no GPU measurement in this artifact**.

This source bundle adds a stock-first paired timing diagnostic for
`fixed32_all_parent_commit_v2` on the canonical real SWE-Verified task
`astropy__astropy-12907`. Both arms use Hydra27 fixed32, B1/concurrency 1,
full vocabulary (`K=0`, `root=0`), one physical 31-draft tree plus root, the
same pinned stock FA2 binary, and SFWD/DFWD/CFWD phase timers. The runner reduces
the canonical deploy-speed brackets to CFWD GPU ms/event, measured full-step
wall ms/event/TPS, and accepted drafts/event.

The only production delta is
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0` to `1`. Before Docker is
queried, the runner verifies that commit `f19e90053` is an ancestor, that the
qualified CFWD source files remain byte-identical, and that the curated PASS is
the exact source-bound B1 credential. After each arm it requires a resolved
SWE-Verified evaluator verdict. The final reducer requires identical container
environments after normalizing arm paths and the CFWD selector, absence of
production state in stock, an exact copied credential in candidate, and every
completed candidate work-census event on the one-launch production route.

Run from a full clean checkout containing the contract-pinned FA2 artifact:

```bash
FR13_RUN_CFWD_ALL_PARENT_B1_TIMING=1 \
RUNROOT=output/fr13_cfwd_all_parent_b1_timing_<UTC> \
TAG=cfwd_all_parent_b1_<UTC> \
bash scripts/fr13_run_b1_cfwd_all_parent_timing.sh
```

The output is `timing_summary.json` below `RUNROOT`. This one-task diagnostic,
the prior byte gate, and its measurements are all timing/floor-acceptance
ineligible. The candidate remains default off. No stock/candidate timing
numbers are claimed by this source-only artifact.
