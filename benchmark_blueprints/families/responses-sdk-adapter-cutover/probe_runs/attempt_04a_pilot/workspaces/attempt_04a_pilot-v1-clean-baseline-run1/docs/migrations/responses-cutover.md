# Responses Cutover

This workspace now reads Responses output items directly instead of routing
through the legacy chat-wrapper message path.

Cutover requirements:

- request the `responses` wire API
- consume ordered `output` items as the source of truth
- preserve `call_id` on both tool calls and tool results
- replay serialized events from structured event records, not rendered transcript text

Implementation notes:

- message items are normalized from ordered content blocks such as `output_text`
- tool correlation is preserved by carrying `call_id` through normalize, replay,
  and render paths
- transcript rendering remains a view over normalized events; replay does not
  parse rendered transcript lines
