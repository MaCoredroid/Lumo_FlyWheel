The billing ledger rules flow was updated to the RulePlan v2 API.

Tests now expect the repo to use `build_rule_plan()` from
`norm_app.rules_v2` instead of the removed legacy helpers. Update the
assembler, router, and CLI preview path so the v2 dataclass is used
consistently and the deprecated import path disappears from the codebase.

Keep the existing behavior the tests describe; do not remove the tests.
