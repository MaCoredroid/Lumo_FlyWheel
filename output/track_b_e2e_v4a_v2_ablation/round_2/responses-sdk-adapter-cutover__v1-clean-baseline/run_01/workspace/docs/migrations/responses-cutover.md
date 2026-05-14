# Responses Cutover

This workspace has migrated from the legacy chat-wrapper path to Responses event semantics.

## Configuration Changes

- `wire_api` is now set to `responses` instead of `chat_completions`
- `transcript_mode` is now set to `responses_events` instead of `legacy_messages`

## Event Ordering

Events in the Responses transcript mode are ordered sequentially by their occurrence in the conversation. Each event maintains its original position in the stream, preserving the causal order of:

1. User messages
2. Assistant responses
3. Tool calls
4. Tool results

The event order must be preserved during replay to ensure correct state reconstruction.

## Tool-Result Correlation

Tool calls and their corresponding results are correlated via event identifiers. Each tool call event generates a unique ID that is referenced by its matching tool result event. This correlation ensures:

- Tool results are matched to their invoking calls
- Multiple concurrent tool calls can be tracked independently
- Replay systems can reconstruct the exact execution flow

## Replay Behavior

Replay is event-sourced: state is reconstructed by processing the ordered sequence of events directly from the transcript. Do not rebuild state from rendered transcript text.
