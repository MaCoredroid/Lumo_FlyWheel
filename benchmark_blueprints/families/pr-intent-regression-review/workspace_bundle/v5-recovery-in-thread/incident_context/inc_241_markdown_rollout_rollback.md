# INC-241 Markdown Rollback

March rollback summary: changing the default export path away from JSON caused
the nightly release-readiness parser to fail. The rollback restored the JSON
default and re-ran the fanout job.
