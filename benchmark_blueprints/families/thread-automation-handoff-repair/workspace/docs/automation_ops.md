Release handoff runs as the existing hourly heartbeat and resumes work in the same thread.
The wake-up should produce the release handoff as an in-thread reply or inbox item, not as `handoff.md` or a detached job artifact.
Repair the existing `release_handoff` automation in place and do not create a second schedule.
