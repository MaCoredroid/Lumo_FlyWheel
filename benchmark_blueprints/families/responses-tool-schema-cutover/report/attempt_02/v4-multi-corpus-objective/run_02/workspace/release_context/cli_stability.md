# CLI stability note

The public summary format must remain `tool_call[...]` / `tool_result[...]` stable
during the Responses cutover so operator snapshots remain diff-friendly.
