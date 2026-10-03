# Native joint common-O0 A: live boot and first repeat-pair review

**PASS for one healthy boot and the two selected root-only observations.** This is an incomplete live run: process A of the planned A/B cohort, one independently audited case out of 84. It is not a completed process, full native baseline eligibility, candidate qualification, lifecycle result, or timing comparison.

Run: `q1-native-joint-common-o0-20260929T155000Z-aligned_nonpacked-A`. Read-only metadata snapshot at **16:18:57 UTC**; accepted raw audit ran **16:21:06–16:21:20 UTC** on September 29, 2026, using exactly two allowed CPU affinity slots (18 and 19), CPU thread limits of two, and empty CUDA visibility. No model requests, GPU work, cache actions, remote file writes, or source changes were performed.

## Boot and source identity

Actual engine start is **16:03:31 UTC**, healthy marker **16:09:12 UTC**, and the exact owned CID is `cc36132e609db1d9bdb318e6fa181871c2e4af619840638beca68575927ffae3`. Read-only Docker inspection confirms this CID was running, with no OOM or restart. The immutable image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`.

The accepted `engine_view` check reproduced exact image-default plus arm environment, command/entrypoint, mounts, memory/swap and IPC policy. The native job was reproduced through the actual plan. Current source bytes matched all 46 declared paths plus the two external source bindings. The saved gate, launch binding, corpus, exact 84-case / R2 / 168-request process scope, worker patch output and patch detail all matched. Plan SHA is `ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61`; source manifest SHA is `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8`.

Boot reports no problems: aligned native target, speculative target configuration absent, packed flag correct, **48 GDN + 16 target attention layers** with actual patched FA2 dispatch. Attention allocation group has **17 members**, including the independently owned MTP singleton. Recurrent caches use FP32 with 1024-token logical blocks; target attention uses 64-token kernel pages. FA2 install/interface and generated worker receipts pass the accepted exact-byte checks.

This establishes **one additional healthy model boot**. Added to the parent's previously recorded eight, the running healthy-boot counter is **nine**; this review did not recount the eight historical boots. Terminal cleanup is not yet available and is not inferred from a healthy boot.

## First root-only pair

The selected case is `calibration-short_available__c0__root-only`, process A, repeats 0 and 1. Both have prefix length **13,487** and consumed chain tokens `[11352, 25559]` at positions `[13487, 13488]`. The accepted driver seal authenticator and joint raw auditor both pass independently. Recomputed joint audit outputs equal the saved driver outputs exactly, with matching job, request/observation, per-observation cache salt and actual prompt-token counts.

- Authenticated target O0 and O1 digests are identical between repeats. Imported MTP-prefix and combined source digests also match.
- Target O2 complete raw logits match byte-for-byte: SHA `3369ded0a0e55a7787df0eb97a17c2698ce70f2bff865ea2d17cbbdb55aff559`; smallest-ID winner **198**, one maximum, margin **0.25**.
- All three MTP continuation logit vectors and native ordered top-three lists match between repeats. Before-Z first: spine 314, top3 `[314,364,279]`; follow-up: spine 279, top3 `[279,411,593]`; after-Z first: spine 248069, top3 `[248069,12,9764]`. Target-hidden raw identities match where recorded. The two recorded persistent MTP KV snapshots also have equal logical digests.
- Full source authentication, destination bootstrap and continuation segment checks, union import/readback, storage separation and mapping, native input/operator binding, same-input reference checks, and raw state/logit finite checks pass. These checks retain the distinction between source history and actual destination execution.

Each driver seal authenticates 6,977 target objects / 1,195,911,168 bytes. The shared accepted snapshot reader authenticated **13,825 distinct target/bootstrap objects / 2,233,739,264 bytes** across this pair. Separate MTP segment auditors report valid source (17 records), bootstrap (14 records), and continuation (3 records, one follow-up) evidence for each repeat. Their object/byte totals overlap and are not summed as a unique archive size.

At raw-audit completion, **38 seals and 37 completed driver receipts** were visible, with the next receipt still in progress; no terminal run receipt existed. Only the selected two observations were independently raw-audited here. Full 84-case coverage and process B remain outstanding.

## Readiness and preserved evidence

The saved consumer receipt admits the exact run/gate/runner. Read-only copies of the 15:58:07 recovery terminal, current authorization/config and consumed marker agree: recovery `SUCCESS`, authorization consumed, gate SHA `e5b375d73cfae464f3b571f6736da5dfd77e1b59b8b9176a14147aa191a2033b`, operational configuration SHA `47d0dd17d46ad23c80e0e61acb33ad50b8762ffef6c0865eb62ce1af7bf22430`. This is a binding check of the prior operation, not a replay or a second full recovery audit. Consumed authority cannot be reused.

Evidence is preserved in `p0/monitor/review-response-20260927/native-joint-common-o0-A-live-first-pair-review/`: scripts/commands, remote metadata and member hashes, raw-auditor output, two-CPU affinity, exact live inspect, recovery binding copies, and local comparison. Raw tensors were read remotely in place, not copied or modified. The accepted auditor source SHA is `9f28dd94d2fc243c70e42769573b5c33e5613ce749c95113efc0e72ff9aae992`; unchanged helpers remain bound in the captured source-hash map.
