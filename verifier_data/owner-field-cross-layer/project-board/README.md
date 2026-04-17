Rich verifier bundle for `owner-field-cross-layer/project-board`.

The visible repo only fails because the owner field is not threaded through the
store, service, CLI, defaults, and docs. The hidden bundle extends that
benchmark with canonical routing coverage: project-board routes must preserve
the raw owner field while normalizing separator-heavy owner labels into a
stable routing-key prefix.
