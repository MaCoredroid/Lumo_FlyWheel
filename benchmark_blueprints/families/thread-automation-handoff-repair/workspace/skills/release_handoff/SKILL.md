# Release Handoff Skill

If the release handoff automation is broken, repair the existing automation bundle in place.
Keep the current automation id, wake the existing thread, and deliver the handoff as an in-thread reply rather than a file.
Do not create a second automation or replacement schedule.
