# Fixed32 kernel-candidate integration

Status: **source integrated; every candidate default OFF; GPU gates and timing pending**.

This branch starts at the batch-invariant BV8 source
`b41720fccd6a4f901d0d809751df826a0c87561d`, whose parent is the shared
production source `92d705c31f375b8a8d42a911eaa73104c722b075`. It consolidates the
default-off Stream-K contract from `42250eeefff8c3b1d1ba0c92c987fcd19cd94daf`,
the CFWD committer layer-batch source and readiness evidence from
`40913fc80dd6b1226e252a8b692835ef052b5115` and
`11569a4ce2933faa1def927dd07dbbe715a449bd`, and the draft-head M32
qualification source from `3b06acebbd673466703268bf0b3647f4bf4a3070`.

The draft-head component's historical artifact remains byte-for-byte unchanged
and therefore retains its original patch-source hash. `manifest.json` binds the
consolidated patch-source hash used by future draft-head credentials.

Two source conflicts were resolved conservatively. The newer sample-quiesced
flush transaction remains authoritative, with draft-head terminal finalization
inside the same transaction. The launcher retains both the Stream-K credential
guard and the draft-head environment guard. No candidate selector or production
flag was changed from its default-off value.

No GPU or Docker command was run for this integration, and no byte-equality,
performance, or floor-acceptance result is claimed. Each candidate still
requires its own real SWE-Verified byte gate before timing or production use.
