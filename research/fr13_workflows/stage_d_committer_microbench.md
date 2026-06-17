# FR13 Stage D profile: committer is NOT the overhead target (2026-06-17)

Microbench of fr13_device_multidraft_commit (the planned tuning target), in-container,
GPU-isolated, vocab 151936:
| committer call | wall/call |
|---|---|
| cat6 (6 nodes) | 0.34 ms |
| cat9 (9 nodes) | 0.35 ms |
| E5 chain (5 nodes) | 0.35 ms |

The committer is ~0.35 ms and shape-invariant. The cat6-vs-E5 overhead delta is ~28 ms/step
(cat6 wall/step 0.260 - verify 0.138 = 0.122 vs E5 0.231 - 0.137 = 0.094). So the committer is
~1.3% of the delta => TUNING THE COMMITTER WON'T RECOVER THE OVERHEAD. Stage-D-as-scoped (tune
the committer) is REFUTED by the profile.

By elimination (verify equal 0.137~=0.138 MEASURED; committer 0.35ms MEASURED): the +28 ms/step
lives in the tree-PROPOSE / model-runner tree-metadata host path (cat6 builds a candidate tree +
tree-attention metadata each step; E5 builds a chain). Still partly ours (tree-propose/metadata),
but the propose+gap base is large (0.094 even for native E5) and includes STOCK scheduler/engine-loop/
sampling -> a chunk is not tunable.

CAVEAT: the microbench is GPU-isolated, so it captures the committer's COMPUTE but not lost-overlap
from its per-step .item() DtoH syncs (which kill run-ahead in the live pipeline). Only a boot-profile
with timers on propose vs committer separates that. NEXT: boot-profile cat6 to pin propose vs committer
vs stock-gap before any tuning. Tool: scripts/fr13_committer_microbench.py.
