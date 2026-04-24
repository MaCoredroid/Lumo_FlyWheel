# Preview Consumer Contract

The downstream preview reviewer bot reads `approval_state` from the fallback
payload on `preview_unavailable` responses before it looks at any boolean helper
fields. `requires_human_review` is still expected, but it is secondary.

Do not restore `legacy_preview_hint`. The consumer was explicitly updated to
read `approval_state` instead, and aliases now create divergent behavior between
the live preview path and the fallback path.
