# Verification

- Exact load-once peak-4 feasibility search: no schedule found after 2,257
  memoized partial-schedule states.
- Selected 32-node schedule: peak live current-row frontier `5`, live-frontier
  sum `116`, 32 global current-row loads per channel, and zero reloads.
- Natural load-once order reference: peak `11`, live-frontier sum `230`.
- AST structural checks require 128 ordered BF16-rounded products, 32 SiLU
  expressions, 32 output stores, and 32 source-stage current-row stores.
- Related fixed32 SFWD tests: `67 passed`.
- Ruff, Python compilation, shell syntax, and `git diff --check`: passed.
- Source commit and upstream were equal before artifact creation.
- Compiler, Docker, GPU, real-task, correctness-gate, and timing use: none.
- No codegen-resource, live-correctness, throughput, acceptance, or floor claim
  is made.
