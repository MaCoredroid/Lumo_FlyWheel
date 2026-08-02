# Fixed32 DFWD K64 M1 pair8 runtime source

This artifact records the default-off runtime integration of the pinned
`pair8bits` BF16 K64 M1 draft-head kernel into the real SWE-Verified B1 gate.

The route is restricted to exact fixed32 B1, single-logits, root K64, and the
pinned gathered-vocabulary map. The candidate binary is identity-pinned and
mounted read-only. Runtime setup, tensor geometry, dtype, stride, device, map,
and CUDA graph warmup failures all fail closed.

The source and binary passed focused host/static verification only. No live
extension-load, CUDA graph replay, task-correctness, acceptance, throughput,
byte-equality, B4, production, or hardware-floor claim is made here. Those
claims require the pending real B1 run. Since the shared vLLM patcher changed,
the historical M32 qualification is intentionally invalid on this branch until
it is separately requalified.
