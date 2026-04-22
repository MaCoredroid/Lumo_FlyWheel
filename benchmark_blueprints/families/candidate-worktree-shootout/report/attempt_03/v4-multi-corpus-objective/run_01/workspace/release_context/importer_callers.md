# Importer Caller Matrix

The release blocker for this cycle is the scheduled importer path.
It calls `service.compile_filters(...)` directly and still rejects
separator-heavy labels such as `Ops---Latency__Summary`.

Until the shared service contract is fixed, the importer can not
graduate from shadow to active rollout.
