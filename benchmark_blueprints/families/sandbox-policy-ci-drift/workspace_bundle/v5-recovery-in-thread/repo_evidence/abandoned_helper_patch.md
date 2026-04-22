
# Abandoned Helper Shortcut

A previous attempt normalized only `tests/fixtures/visible_config.toml`
before invoking pytest. That made one happy-path fixture green without
fixing parser compatibility for real configs.
