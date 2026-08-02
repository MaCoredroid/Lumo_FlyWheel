# Fixed32 B4 BV64 graph gate rejection

Status: rejected before health and before any SWE-Verified task ran. This
artifact makes no byte-parity, B4 correctness, timing, throughput, production,
or hardware-floor claim.

The canonical SWE-Verified exact4 B4 graph diagnostic launched from
`131a592201201f4c65c4fc7221e9ef6fa7fdb8a9` with concurrency four, stock BV8
served, BV64 shadow-selected, and the pinned stock FA2 binary
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
The runner returned `serve_rc=2`.

During descending FULL graph capture, B4 completed its capture. The next B3
pre-capture lease check rejected `pregather_bank_aliases=False`. The GDN
SD-layout path computes `self.kv_cache[0].transpose(-1, -2)` on every forward,
so successive forwards create distinct Python Tensor wrappers for the same
exact persistent cache view. The preseed guard used Python object identity,
while the in-capture kernel guard already requires the actual lease properties:
data pointer, shape, stride, storage offset, dtype, and device.

No real-event marker, ingress ledger, health record, byte-gate PASS, or task
result exists. The launch/end runtime manifests are byte-identical, as are the
launch/end external manifests. Automatic authenticated cleanup could not
finalize after EngineCore startup failed; the stopped container was then
removed, with zero containers and zero GPU compute processes recorded.

Repair `f230f87f6` changes only the conv-cache comparison to exact Tensor-view
alias equivalence and retains rejection for different storage, pointer, offset,
shape, or stride. Validation after the repair: 26 focused graph/preseed tests
passed; the full fixed32 suite reported 647 passed and 7 skipped.

The required next action is a new canonical exact4 B4 graph-byte diagnostic
from the repaired clean commit.
