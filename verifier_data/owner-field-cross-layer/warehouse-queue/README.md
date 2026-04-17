Rich verifier bundle for `owner-field-cross-layer/warehouse-queue`.

The visible repo only fails because the owner field is not threaded through the
store, service, CLI, defaults, and docs. The hidden bundle extends that
benchmark with canonical routing coverage: warehouse queue routes must preserve
the raw item name while normalizing separator-heavy queue labels into a stable
routing-key suffix.
