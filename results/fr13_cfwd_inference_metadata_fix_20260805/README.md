# Packed CFWD inference-metadata boot fix

The real B1 Hydra27 physical32 K64/root1 composed gate on source
`9091ddae2046f42fc5e754f976c3493a033785ac` reached FULL CUDA-graph capture
but failed before service readiness and before SWE-Verified task traffic.
`prepare_metadata_binding` attempted to read `Tensor._version` from immutable
TAW metadata allocated under `torch.inference_mode()`. PyTorch intentionally
does not provide version counters for those tensors.

The fix keeps exact-value attestation at binding time and storage-pointer
binding for every tensor. Normal tensors remain version-counter bound so
in-place mutation is rejected. Inference tensors use an explicit `-1`
no-version sentinel, avoiding the unsupported API during graph warmup and
capture. Pointer replacement remains fail-closed.

This artifact records a boot failure and a CPU/static regression fix only. It
does not claim a passed real-task gate, timing result, speedup, acceptance, or
hardware-floor result. Fresh source-exact B1 gates are required after the final
timing source is merged.
