# Skill Routing

The router now evaluates skills in three stages:

1. A skill is only eligible if at least one entry in `triggers` appears in the request text.
2. Any `negative_triggers` hit suppresses that skill, even if a positive trigger matched.
3. Every item in `required_inputs` must be present before the skill can be selected.

If multiple skills remain eligible, the router prefers the most specific match by
counting positive trigger hits and then preferring the longest matched trigger. The
configured fallback is only used when no eligible skill remains after suppression and
input checks.

```toml
[router]
default_skill = "general_helper"
schema_version = 2
ranking = "most_specific"
fallback_when_no_eligible = true

[[skills]]
name = "general_helper"
triggers = ["help"]
negative_triggers = []
required_inputs = []

[[skills]]
name = "release_handoff"
triggers = ["release handoff", "handoff draft"]
negative_triggers = []
required_inputs = []

[[skills]]
name = "deploy_check"
triggers = ["deploy", "rollout"]
negative_triggers = ["policy", "doc"]
required_inputs = ["env"]
```

Legacy `trigger` values can still be normalized as a single-item trigger list for
backward compatibility, but new examples should use the multi-field schema above.
