# Fixed32 FA2 qrow16 thin build

Status: **REJECTED before GPU use**.

The compile-thin candidate built one CUDA object plus one shared-module link
from the exact-safe FA2 source. It reduced the required compile from the
aborted 52-object generic build to about two minutes and produced:

- candidate SHA256:
  `71881915b18da1b9f144982e83adabc6ff626ab3bcc4a11cbf7f8c9eafaf0dde`;
- candidate size: `299340352` bytes;
- exact-safe base SHA256:
  `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`;
- exact-safe base size: `299183936` bytes.

The candidate was rejected by the pre-GPU ABI gate. Compared by normalized
dynamic symbol name and type, the explicit specialization:

1. changed
   `run_mha_fwd_splitkv_dispatch<bf16,256,false>` from weak (`W`) to strong
   (`T`); and
2. replaced one exported weak stock lambda symbol with another.

Defined symbol counts remained 685 in both modules, but zero name/type delta
is the contract. No server, CUDA context, synthetic request, or SWE task used
this shared object.

The required source correction is to preserve the stock explicit
instantiation and route the qrow specialization through a hidden internal
helper. A new production build and the same-process real SWE byte gate are
required.
