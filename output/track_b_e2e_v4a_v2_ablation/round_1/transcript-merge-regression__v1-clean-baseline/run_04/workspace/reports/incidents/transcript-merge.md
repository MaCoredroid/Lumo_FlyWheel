# Transcript Merge Incident Note

## Resolution

The reducer bug has been fixed in `replay/merge.py`:

- Tool outputs now use `event_id` for stable identity instead of `(role, tool_name)` grouping.
- Post-completion `debug_only` fragments are now filtered at merge time, not render time.

The incident summary in `replay/incident_summary.py` now counts merged tool-output blocks directly from the merged events, with `count_source` set to `"merged_events"`.

## Old guidance

The previous mitigation of filtering duplicate lines at render time is no longer needed. The incident summary previously counted rendered tool lines instead of merged tool-output blocks.
