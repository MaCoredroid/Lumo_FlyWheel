# Agent Instructions - responsive-checkout-visual-regression

Fix the checkout preview mobile overlap without disabling the compact summary experiment or removing sticky behavior globally.

Read `preview_artifacts/qa_repro.md`, inspect the checkout component/style files, and update tests with at least one mobile viewport assertion. Keep desktop two-column behavior intact.

Allowed outputs:
- Source/style/test changes inside this workspace.
- A short QA note at `docs/checkout_mobile_qa.md`.

Do not edit `.scenario_variant`, `AGENTS.md`, or files under `preview_artifacts/`.
