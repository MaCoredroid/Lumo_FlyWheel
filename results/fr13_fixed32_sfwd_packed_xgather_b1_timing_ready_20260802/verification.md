# Verification

- Exact reduced byte gate, launcher source, and descriptorless kernel source
  passed the host binding validator.
- Candidate and kernel timing tests: 26 passed.
- Ingress, route wiring, and lifecycle tests: 112 passed.
- Python compilation, Ruff, and Bash syntax checks passed.
- The future runner requires source, runtime, and external launch/end equality.
- The future runner requires counts-only zero-resource censuses before the first
  arm, after stock, and after candidate.
- Failed routes also perform a counts-only zero-resource census during exit.
- Runtime and host verification require exactly 48 candidate launches across
  48 unique layers.
- No Docker, GPU, or real-task work was launched while preparing this artifact.
