# B4 identity-divisor Hydra27 failure

The real SWE-Verified exact4 B4 Hydra27 K64/root byte diagnostic reached
authenticated task ingress, then the first armed candidate CUTLASS call returned
`Error::kErrorInternal` from `gemm_op.run`. The engine exited before a stock vs
candidate comparison record could be written.

This is a candidate-kernel rejection. It is not a byte result, timing result, or
acceptance result. No task completed and no measurements are admissible. The
next isolation run uses the same identity epilogue and stock-shape tile with the
default scheduler; that distinguishes the epilogue from the divisor-balanced
scheduler path.

