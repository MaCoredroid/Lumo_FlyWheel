# artifacts/

This directory is intentionally empty in the shipped workspace for
variants v1–v5. The gold ranking is NOT visible to the agent.

The grader mounts the real `gold_ranking.json` from
`verifier_data/proposal-ranking-manager-judgment/<variant_id>/gold_ranking.json`
at scoring time only.
