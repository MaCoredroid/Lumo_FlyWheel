Round 2 follow-up for `release-readiness`:

Markdown output now needs to include owners who currently have zero active items when the adapter knows about them, and pluralization must be correct everywhere that shared formatting helpers are used.

Do not patch this only in the Markdown renderer. Fix the shared behavior at the correct upstream site so any present or future renderer gets the same zero-count handling.
