# Responses Cutover

Use the Responses wire path and store replay inputs as ordered event records, not
legacy wrapped messages.

- Sort inbound response items by `sequence` before normalization so replay
  preserves original event ordering even when message, tool-call, and tool-result
  items arrive interleaved.
- Expand message `content` blocks in-order and emit `assistant_text` events only
  from `output_text` blocks; do not rebuild state from rendered transcript text.
- Preserve `call_id` on both `tool_call` and `tool_result` events so tool results
  stay correlated with the originating call during replay and transcript render.
- Keep replay event-sourced by serializing normalized events directly rather than
  reconstructing state from the human-readable transcript.
