# Routing Audit Note

- Positive match: `"please draft a release handoff"` routes to `release_handoff` because it satisfies the `triggers` check and has no suppressors or missing inputs.
- Suppressed match: `"update the deploy policy doc"` does not route to `deploy_check` even with `env=prod` because `policy` and `doc` hit `negative_triggers`, so the router falls back to `general_helper`.
