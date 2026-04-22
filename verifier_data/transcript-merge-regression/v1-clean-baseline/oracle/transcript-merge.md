# Transcript Merge Incident Note

Stable event identity is the reducer boundary. The merge key must use
`event_id`, not `(role, tool_name)`, because two tool-output blocks can
share a tool name and still be distinct events.

Debug-only fragments that arrive after `response.completed` are merge noise
and should be dropped, but legitimate deferred tool output after completion
must still survive when it carries a real event identity.

Incident summary now counts merged events directly before rendering.
Filtering duplicate lines after rendering is not an acceptable fix.

Variant anchor: visible replay, same tool name.
