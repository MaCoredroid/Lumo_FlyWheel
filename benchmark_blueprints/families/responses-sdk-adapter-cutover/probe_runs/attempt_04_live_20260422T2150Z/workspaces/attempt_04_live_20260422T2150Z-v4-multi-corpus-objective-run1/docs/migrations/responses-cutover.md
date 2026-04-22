# Responses Cutover

Use the Responses wire path and store replay inputs as ordered event records, not
legacy wrapped messages.

- Sort inbound response items by `sequence` before normalization so replay
  reflects the original event order even when delivery is interleaved.
- Expand message `content` blocks in-order and emit `assistant_text` events only
  from `output_text` blocks; ignore unknown future block types instead of
  rebuilding from rendered transcript text.
- Preserve `call_id` on both `tool_call` and `tool_result` events so tool
  outputs remain correlated with the originating call during replay and render.
- Keep replay event-sourced by serializing normalized events directly; do not
  reconstruct state from a human-readable transcript.
