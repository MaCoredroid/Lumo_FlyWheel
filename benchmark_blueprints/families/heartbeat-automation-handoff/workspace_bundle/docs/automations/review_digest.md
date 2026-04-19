# Review Digest Automation

The review-digest automation is a thread heartbeat, not a detached cron job.

Canonical serialized fields:
- `kind = "heartbeat"`
- `destination = "thread"`
- no persisted raw `thread_id`
- the durable `prompt` contains only the task to resume

The serializer preserves the user-authored prompt byte-for-byte and keeps `status = "PAUSED"` when a legacy heartbeat is round-tripped. The prompt template passes through a supplied `prompt` verbatim and falls back to the canonical inbox-opening instruction when no prompt text is provided. Cadence and workspace details remain in schedule metadata rather than the prompt body.
