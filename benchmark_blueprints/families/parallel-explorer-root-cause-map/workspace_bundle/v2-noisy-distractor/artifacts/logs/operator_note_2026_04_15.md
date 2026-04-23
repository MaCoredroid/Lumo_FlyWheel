# Operator Note — 2026-04-15

The markdown report doubled headings again after the scheduler refactor.
Quick thought from on-call: maybe `render_blocked_owner_section()` stopped
deduping. I did not trace upstream owner keys before writing this note.
