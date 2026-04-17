Rich verifier bundle for `owner-field-cross-layer/release-gate`.

The visible repo only fails because the owner field is not threaded through the
store, service, CLI, defaults, and docs. The hidden bundle extends that
benchmark with canonical routing coverage: release gate routes must preserve
the raw release-train name while normalizing separator-heavy train labels into
a stable routing-key suffix.
