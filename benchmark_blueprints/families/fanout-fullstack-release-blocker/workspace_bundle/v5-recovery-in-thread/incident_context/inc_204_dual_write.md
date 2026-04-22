# INC-204 dual-write rollback

The previous rollout tried to emit both `manual_review` and
`human_review_required` during the same session. The downstream release-gating
fanout treated that as conflicting state and rolled the deployment back.
The recovery rule is strict: no dual-write alias in the live path.
