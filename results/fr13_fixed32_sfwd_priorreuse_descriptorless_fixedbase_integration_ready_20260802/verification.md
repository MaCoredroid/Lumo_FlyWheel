# Verification

- Source commit `8ec4c5126cbb323f27143a06064629b76e142550` pushed to
  `agent/fixed32-sfwd-descriptorless-fixedbase-20260802`.
- Focused source tests: 60 passed.
- Full fixed32 ingress proxy suite: 42 passed.
- Python bytecode compilation: passed.
- Shell parsing for all touched launchers: passed.
- Ruff checks for the candidate, gate, and focused tests: passed.
- Git diff whitespace check: passed.
- Source manifest generation from committed bytes: passed.
- Runtime source binding for integration and kernel files: passed.
- Exact kernel-function AST bytes match offline source commit
  `83b8eb0f697eb1e5c98470aa214e6b31317d9e8d`.
- Exact padded-x and dense output/stage/weight contract: enforced before launch.
- Kernel source descriptor and runtime x/weight stride arguments: absent.
- Shadow output allocation: explicitly dense.
- Gate state: default off, B1-only PASS, K64/root1 pinned, 48 unique layers,
  two byte surfaces, incumbent always served.
- Runner: source launch/end manifest binding present and shell-valid.

Explicitly not run: GPU kernel execution, Docker, server/service, real task,
timing, TPS, hardware-floor acceptance, or production selection.
