# Fixed32 B4 BV64 graph gate rejection

Status: rejected before health and before any SWE-Verified task ran. This
artifact makes no byte-correctness, timing, throughput, acceptance, production,
or hardware-floor claim.

The canonical exact4 B4 graph diagnostic launched from
`065304d4785dae95ad954ee17d9d0e79cfdaa7dc` with concurrency four, stock BV8
served, BV64 shadow-selected, and pinned stock FA2
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
The runner returned `serve_rc=2`.

Descending FULL graph capture completed its B4 iteration (`1/4`). The next B3
iteration stopped on `FR13_FIXED32_CONV_PREGATHER live SSI source mapping
drift`. The run source selfchecked only the initial B4 SSI prefix, then marked
B1-B4 complete in the separate preseed set. B3 therefore reached its
in-capture guard without B3 in `live_selfchecked_batches`; this missing
readiness entry was the rejection cause.

Completion of the B4 capture iteration is not a B4 byte result. The server
never reached health, and no real-event marker, ingress ledger, byte-gate PASS,
byte record, or task result exists. Launch/end runtime manifests are identical,
as are launch/end external manifests. Automatic authenticated cleanup could not
finalize after EngineCore startup failed; the stopped container was manually
removed, with zero containers and zero GPU compute processes recorded.

Repair `c945055c7` requires the initial producer call to cover server capacity
and selfchecks every live prefix B1-B4 before publishing readiness. Current-HEAD
CPU validation recorded here passed 11 pregather/committer wiring tests, 26
FULL-preseed/graph-gate tests, and the full fixed32 suite with 648 passed and 7
skipped, plus `py_compile` and `git diff --check`.

The required next action is a new canonical exact4 B4 graph-byte diagnostic
from the repaired clean commit.
