# Reviewer Note

Kept Worker A's markdown/watchlist implementation and Worker B's docs/heading
cleanup, but rejected every hunk that drifted the JSON contract or touched the
unrelated fixture. The remaining regression risks are JSON key drift and silent
loss of the watchlist follow-up path when `--include-watchlist` is requested.

