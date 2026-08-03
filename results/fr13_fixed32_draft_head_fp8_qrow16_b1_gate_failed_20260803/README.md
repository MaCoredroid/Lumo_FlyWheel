# FR13 K64 FP8 head plus Qrow16 B1 gate closeout

Status: **FAILED BEFORE CANONICAL TASK WORK**

The source-bound one-real-task B1 gate ran from pushed commit
`711c08551de2e5eabf9788ad44002e5b0a2564db` with Hydra27, physical32,
K64/root1, Qrow16 production, and only the direct block-FP8 draft-head
candidate enabled.

Model boot, Qrow16 binding, FP8 weight quantization, the B1 FP8 head call,
and full four-iteration drafter-graph capture all engaged. On the first
canonical request, the fail-closed FP8 replay attestor rejected the capture
lifecycle because its selected-loop-head counter was not exactly four. The
engine exited before the task emitted speculative work:

`FR13 draft-head FP8 graph capture did not select four loop heads`

The gate returned `15`. It produced zero completed SWE-Verified tasks and no
valid timing, TPS, acceptance, or floor-ratio result. In particular, this is
not evidence that FP8 is faster or slower than the BF16 head.

The candidate's byte ledger remains informational only: mandatory bytes
`30,989,326,208`, weight-read floor `113.514015414 ms/event`, and nominal
1.15x cap `130.541117726 ms/event`. The existing valid B1 Hydra result remains
the current speed point until a corrected candidate clears this gate and an
exact4 timing pair completes.

The failed run's exited container was removed and final host state was Docker
empty with no GPU compute process. Raw prompts, responses, patches, task
workspaces, process/container identities, and raw logs are not published.
