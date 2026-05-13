# Transcript Merge Incident Note

## Resolution:

The reducer in `replay/merge.py` has been fixed to use stable event identity via `event_id` for tool outputs, preventing false merges of distinct tool events. Debug-only fragments appearing after completion are now properly removed during the merge phase. The incident summary now counts from merged events directly.
