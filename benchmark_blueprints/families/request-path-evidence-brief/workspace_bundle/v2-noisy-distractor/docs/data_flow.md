# Data Flow Note

The CLI entrypoint calls `sync_item`, which resolves the effective owner, persists the base record, and then emits the payload.

Historical notes often over-compressed the flow because the stored record already contains `owner`. That shortcut is not enough to prove where `owner_source` or `routing_key` are decided.
