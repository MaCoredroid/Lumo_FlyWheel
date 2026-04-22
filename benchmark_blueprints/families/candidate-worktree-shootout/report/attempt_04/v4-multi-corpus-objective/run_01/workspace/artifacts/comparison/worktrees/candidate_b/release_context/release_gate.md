# Release Gate

This week's gate is the importer shadow-readiness checklist.
A CLI-only patch is insufficient because the importer and saved
view repair job do not traverse `cli.render_filters(...)`.
