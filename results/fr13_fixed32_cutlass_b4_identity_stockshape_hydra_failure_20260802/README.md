# B4 identity-stockshape Hydra27 failure

The real SWE-Verified exact4 B4 Hydra27 K64/root byte diagnostic reached the
first armed candidate projection. The candidate returned
`Error::kErrorInternal` from `gemm_op.run` before any stock-versus-candidate
comparison record was emitted, and the engine exited.

This rejects the candidate binary. It is not a byte-equality result, timing
result, task result, or hardware-floor measurement. The failure also removes
the divisor scheduler as the common cause because this arm used the stock tile
scheduler. Source inspection identified the missing identity-epilogue shared
storage carveout in the automatic stage-count path as the next bounded fix;
that diagnosis remains subject to the replacement kernel's real byte gate.
