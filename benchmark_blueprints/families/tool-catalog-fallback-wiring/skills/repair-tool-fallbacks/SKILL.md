# Repair Tool Fallbacks

Use this skill when a preferred tool id has drifted or disappeared and the router must choose a correct fallback without collapsing to a blanket default.

## Do

- Check both preferred-present and preferred-missing behavior
- Treat capability and safety-policy filters as gating conditions
- Align manifest, config, and docs with the live selection policy
- Prefer general routing logic over case-by-case aliases

## Do Not

- Do not stop at renaming the preferred id
- Do not route every missing-tool case to one default fallback
- Do not rely on the first listed candidate when eligibility differs

## Working Pattern

1. Determine the preferred path when the primary tool is present.
2. Determine how eligibility is computed when it is absent.
3. Repair manifest, config, and docs around the real routing logic.
4. Re-check a missing-tool case and a preferred-present case.

## Success Signal

The selector chooses the correct live tool in both modes, and the operator guidance matches that policy.
