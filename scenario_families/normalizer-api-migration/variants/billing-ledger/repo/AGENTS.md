The billing ledger rules flow was updated to the RulePlan v2 API.

Tests now expect the repo to use `build_rule_plan()` from
`norm_app.rules_v2` instead of the removed legacy helpers. Update the
assembler, router, and CLI preview path so the v2 dataclass is used
consistently and the deprecated import path disappears from the codebase.
Build the `RulePlan` once in the CLI preview path, then thread that
dataclass through the assembler and router helpers instead of rebuilding
it separately in each layer.

The preview contract also now exposes a canonical `dispatch_key` derived
from region, owner, and the normalized title slug. Keep the billing route
format stable, but append the dispatch key as a query suffix so the
preview output preserves both the route bucket and the canonical dispatch
identity. The billing preview contract in `docs/preview_contract.md`
documents the response shape and the canonical slug format that downstream
reconciliation expects.

Keep the existing behavior the tests describe; do not remove the tests.
