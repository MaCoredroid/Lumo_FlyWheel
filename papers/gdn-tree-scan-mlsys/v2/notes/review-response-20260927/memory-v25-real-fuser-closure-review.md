# Memory v2.5: real fuser stream repair

**PASS for this bounded source repair and unchanged candidate-wrapper integration.** The real split-stream parser failure is reproduced and corrected. No remaining blocker was found in the changed code. This review authorizes no recovery, query container, candidate boot, retry, gate change or reuse of an old authorization.

Snapshot: `p0/monitor/review-response-20260927/memory-v25-reviewed-20260928T155149Z`, relative to paper v2. Independently authenticated **34/34** frozen files by size and SHA256. Freeze raw hash: `2b1413e6b671639c1866f0c851483a0090c1639c2b0744fc779835515c73b447`; canonical hash, excluding `frozen_utc` and `freeze_canonical_sha256`: `16dbaa17b13ad20c55ff7802744960d1b6c5be402b55a0f1e169f7a9e983d07e`.

## Actual failure and correction

The preserved v2.4 terminal receipt reports `REFUSED_PRECONDITION`, exit 4, on `/dev/nvidia-modeset: mark F.... Xorg`: write count 0, query not attempted, no durable reservation written, and `next_boot_permitted=false`. Its authority lock was released. This is an operational parser refusal, not a numerical result or completed recovery. The old receipt and sources remain preserved.

The actual stdout contains **15 PID tokens**, while stderr contains **15 PID-less accessor rows** across three devices. The corrected `parse_fuser_pairs` at `tools/memory_recovery_v2_5.py:407` pairs these positionally, retaining device, user, access and command. Five distinct desktop PIDs result: 4178, 4354, 4383, 4849 and 5132. The legacy PID-bearing form is still supported, but its complete PID sequence must equal stdout. Mixed forms, missing/surplus rows or PIDs and unrecognized diagnostic lines refuse.

`Recovery._inventory_gpu_clients:1142–1184` still requires a successful, non-timeout `fuser` inventory and consistent exit code. It authenticates every unique PID with `ps pid,lstart,comm` and the exact allowlisted `(lstart, comm)`. The added check also requires **every** table command for that PID to equal the authenticated `ps` command. PID presence in stdout alone cannot authorize a client.

## Independent CPU controls

Used only the frozen standard-library module, actual captured stream bytes, and canned command responses to the inventory method. No command runner, operation-mode entry point or external inventory command was invoked.

- Real bytes exactly reproduce the 15 expected `(device, PID, command)` tuples and five-PID set; the old v2.4 parser rejects those same bytes.
- Twelve malformed-stream controls refuse: missing row, missing PID, surplus PID, surplus row, mixed row forms, permission diagnostic, PIDs without a table, table without PIDs, malformed PID token, malformed access field, unknown stderr diagnostic, and inconsistent PID-bearing sequence.
- Positive legacy PID-bearing control passes. The real-form inventory control authenticates five clients using exactly one canned fuser response and five canned ps responses, preserving all 15 accessor records in preconditions.
- Nine inventory negatives refuse: unknown PID, omitted allowlist entry, changed start time for a known PID, table command disagreement, changed ps command, rc=1 with accessors, rc=0 without accessors, unsupported exit code, and timeout. Empty streams with rc=1 remain a valid zero-client inventory.

AST comparison after normalizing version-name strings finds **89 existing functions unchanged**. The only modified existing functions are `parse_fuser` and `_inventory_gpu_clients`; `parse_fuser_pairs` is new. The remaining diff is version/schema compatibility, documentation and diagnostic names. Locking, exclusive one-use reservation, pre-write rechecks, the single bit-2 write, owned query-container cleanup, post-write checkpoints, holds and finalization logic retain their accepted implementations. No numerical or resource threshold was relaxed.

## Candidate-wrapper integration

The candidate wrapper remains byte-identical at `b520e1a27b03354b22d364c890b4215f56dd6d777be2d046f4d4060961660d09`; the readiness consumer remains `2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9`.

Independently executed its exact runner-selection assignment and extracted gate Python against local fixtures with the previously reviewed 52 dependencies. `CAND_MEMORY_RUNNER=<campaign>/tools/memory_recovery_v2_5.py` selects v2.5. The gate accepts the new runner only with its new `reviewed_hashes.memory_runner`; the old v2.4 hash refuses v2.5, and the v2.5 hash refuses the default v2.4 runner. The unchanged consumer admits an idle disposable authority under an explicit no-reclaim disposition, rejects the old recovery-binding hash, and refuses a hold. Only read-only producer verification was used for these checks.

A fresh parent authorization must bind runner SHA256 `973e06783c222ece34df97246ab70d6f766c2222f48e3addf525472fa9d74ac8` in **both** `reviewed_hashes.memory_runner` and `recovery_binding.runner_sha256`, and invoke the unchanged wrapper with `CAND_MEMORY_RUNNER` explicitly set to v2.5. The separate operation authorization/config/source manifest must likewise bind v2.5, the new scientific gate/run, the existing authority namespace and the unchanged candidate reservation. The earlier unused authorization is not revived by this source approval. Live PID/start-time identities and readiness remain action-time checks.

The frozen author logs report **157 parser/runner tests** and **14 wrapper-integration tests** passed. Those logs were hash-checked, not rerun wholesale; the focused independent controls above avoid local pytest and Linux-shell dependencies. The source change does not alter the previously accepted scientific package or expand the first candidate run's scope.

| Artifact | SHA256 |
| --- | --- |
| `tools/memory_recovery_v2_5.py` | `973e06783c222ece34df97246ab70d6f766c2222f48e3addf525472fa9d74ac8` |
| Source diff v2.4 to v2.5 | `2e096c7bd0de005a4082082b771af90c08bb927e4ab922452e3f574bdd67bc66` |
| Real fuser stdout | `11660c9f946f55e5d09810404907376769c762d0c95c2bbe442e5148a99ba139` |
| Real fuser stderr | `c016204e188cff73c3b6f8a963045d7ea9642cac2231c50bc68ee1fe24b2bc3c` |
| Runner CPU log | `4993d6c9dd0245a885f00fbe405b8905dc21adc9468f6977b2262bd07659939e` |
| Wrapper integration CPU log | `06d9ed67a854a7be834aada10e81e8ef43f0158411193ed36c3ad1fe6e2ed6a4` |

Only this review note was added. No remote action, GPU/model/container/query launch, memory reclaim, worker-source edit or gate edit occurred.
