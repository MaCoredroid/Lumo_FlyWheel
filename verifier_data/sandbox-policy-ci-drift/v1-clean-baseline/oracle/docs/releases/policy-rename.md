# Policy Rename

Canonical active policy names are `workspace_write` for sandbox and
`on_request` for approval.

Compatibility:
- local config parsing still accepts the deprecated input spelling
  `workspace-write`
- emitted preview JSON and workflow examples stay canonical-only and
  always emit `workspace_write` plus `on_request`

Reviewer validation:
- run `make ci`
