An integration regression in the payments on-call alert stream started after
the dedupe logic was simplified.

Read `logs/failure.log`, reproduce the problem through the tests, and fix the
dedupe key so alerts are collapsed by the correct time window and environment.
The raw evidence mixes aliases like `Production`/`prod` and second-level
timestamps, so normalize the environment and minute window before building
the dedupe key. The collapsed incidents should still be useful to the
on-call handoff: keep the canonical environment and minute window on each
merged record, and surface `occurrence_count`, `first_seen_at`, and
`last_seen_at` so the grouped incidents show how many raw alerts were
folded together, when the burst started, and when the latest one arrived.

Do not delete the failing integration coverage.
