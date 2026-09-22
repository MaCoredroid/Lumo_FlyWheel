# E1 companion: final bounded artifact review

Review date: 2026-09-22. Read-only member/source/document inspection; no inference, numerical rerun, campaign change, or full rehash of the older large companions. Parent separately verifies every E1 member hash.

## Verdict

**PASS for essential E1 evidence and offline analysis completeness, read together with the existing paper and selected-route companions.** No essential missing raw measurement, reducer, active-source identity, or qualification dependency was found. This is a private review package, not a certification of an unmodified GPU rerun on another host. The two final paper-bundle metadata carry-forward items found in the initial review are now closed by the hash-bound recheck below.

Reviewed archive: `artifacts/e1-timing-20260922T1225Z.tar.gz`, SHA256 `bf50b33386eb9614cf7874039f60cb62a484ee62bc0882e32ab32731857318d6`, 12,955,447 bytes. It contains **1,482 manifest-listed payload files plus one embedded manifest**, so 1,483 tar members is correct. All members are regular files with unique, relative, non-traversing names. The embedded manifest equals the sidecar after excluding the sidecar-only archive digest/size fields. The archive digest was independently checked; this review did not duplicate the parent's full 1,482-member digest audit.

## What resolves from the snapshots

- All **18 sealed cells** have terminal results, configuration/verification, phase and join manifests, direct API-token maps, raw recorder events, Docker logs/inspect/model metadata, executed snapshot manifests, and five loaded Python modules each (**90 modules**). Every manifest-listed phase request has its corresponding capture: **230 requests** total, including the four first-native preflights, warmup, and timing.
- All four first-native variants contain live and final preflight reports and preflight manifests. The campaign contains frozen prefixes, original prompt pool, reference scores, route qualification manifest, qualification progression, exact order/settings, aggregate, and original execution traces.
- The reviewed aggregate and reducer are the exact archived bytes: `aggregate.json` SHA256 `dbd4542ee9899da3eb592fdf8a0b935010f5c8035de9d5cb75e581343c7deb1d`; `experiments/e1/e1_aggregate.py` SHA256 `98beec9efac26f560783d9d77af3e2154edbd6b24ebadfe141b0cafce6e48a68`.
- The complete independent support/continuation evidence is present: `p0/monitor/e1-remaining-cells-redteam.md` SHA256 `6e97dab8031d9c35270d59120132c341f8039232aadf984fe6ed554246e910ca`; companion JSON SHA256 `347ef2f472908b70c5e7f7cb20fe2f7e0efdb553d064a66239bb6698b6f0a8f8`. Its 18 per-cell support/exclusion records, 144 timed API-ID streams, and 288 retrospective comparisons are readable. These were inspected for payload presence, not numerically rerun.
- All **six repository-relative dependencies in the frozen campaign identity** resolve under `p0/monitor/e1-source-20260922T1150Z/`, and their archived bytes were individually checked against the identity hashes: patcher, tree kernel, decode modes, device multidraft kernel, fused tree convolution, and OOM guard. The source-map JSON gives each original repository-relative location and corresponding V2 archive location.
- Every hash named under the frozen tree qualification manifest's `gate_sources_sha256` and `reports_sha256` resolves in this E1 companion or `selected-route-20260922T0930Z.tar.gz.manifest.json`. This includes the final B4 closure report and the exact v3e gate, ledger alignment, operand gate, continuation reducer, tests, and derived-ID sidecar. No large companion payload was rehashed for this lookup.
- The earlier launch-guard review and the parent's structured reply-26 stub evidence are already present in the 10:16 paper archive and remain explicitly listed by `scripts/build_review_bundle.py` (CURRENT_REPORTS, lines 10–13). They do not have to be duplicated into E1. The structured stub JSON contains detailed checks; absence of its separately mentioned text log from E1 is not an essential analysis gap.

The archive retains T1 in `experiments/e1/E1_ASEXECUTED_DEVIATIONS.md`, `notes/e1-protocol-deviations.md`, and the independent reports. It therefore supports the actual request-seed/native-engine-seed/tree-engine-seed distinction and the separation between valid recorded support and exact compliance with every original frozen setting. The append-only note's older report digest is historical chronology; the final review digest is separately bound by the export map.

## Offline analysis and external boundaries

The token mapper's direct-ID route (`e1_api_tokens_from_capture.v2.py:31–37`), joiner, and aggregate are standard-library Python and have their required direct-ID/event/phase inputs in this archive. The mapper only loads Transformers/tokenizer files for the unused decoded-string recovery branch (`23–29,39–44`); that branch is not needed for E1's directly requested IDs. All aggregate and join inputs can be addressed using the V2-relative archive paths.

For example, after extraction into a scratch evidence root, the join can be invoked against a cell's `logs/e1_events.jsonl`, `e1_api_tokens.json`, and `e1_join_manifest.json`, with `--expect-reqs 1` or `4` and a separate scratch `--json` output. Reconstructing the aggregate should use a writable scratch copy of the campaign because the reducer writes `aggregate.json`. These are source-verified invocation paths; this artifact review did not execute them.

Re-executing the native first-token decoding gate additionally needs the checkpoint tokenizer and Transformers (`e1_native_preflight.py:53–64`). Fresh serving needs the pinned image/model/runtime environment and explicit restoration of repository/DGX paths. These are external replay dependencies, not missing prerequisites for inspecting the included gate verdicts or reproducing token/wall accounting. The paper checkpoint's REVIEW-CHECKPOINT instructions and artifacts README already state that original absolute paths are preserved and a full unmodified cross-host GPU rerun is not certified.

## Final metadata carry-forward (initial review; subsequently closed)

At this review, two coordinator-owned metadata locations still describe the earlier incomplete checkpoint:

1. `scripts/build_review_bundle.py:23` defaults `--scope` to “native preflight and E1 timing remain incomplete.”
2. `artifacts/README.md:3` leads with the 10:15 active-campaign state; its later dated entries remain historical.

The final paper bundle should override/update that default and add the completed E1 companion/current state while preserving dated history. The final manifest should bind this E1 archive and its sidecar alongside the three existing evidence companions. Parent was notified; this is part of the already-in-progress final metadata refresh, **not a reason to mutate/re-export the immutable E1 companion or run another experiment**.

The exporter's initial relative-`--sources` failure is consistent with source-path validation before tar creation (`scripts/export_e1.py:58–64` precedes archive creation at line 73). The corrected source map is itself archived with SHA256 `11901c21db744b0dbca33908cab24f1e6cfe9175d1adef2408e1d7f0e4beafc7`. No experiment-evidence effect is implied by that packaging-only failure.

## Final metadata recheck — CLOSED, 2026-09-22

Narrow read-only recheck of the four corrected files; no companion rehash, experiment, numerical rerun, or source edit. Both prior carry-forward items are closed. No remaining material metadata inconsistency was found in this surface.

| Checked file | SHA256 | Closure evidence |
| --- | --- | --- |
| `scripts/build_review_bundle.py` | `ad86fc4788aec6c624468219592a7fe887e5d467ae8d5050519c2e1a1665e0bd` | Line 25 now defaults to a completed bounded review checkpoint while preserving frozen failures, T1, continuation divergence, and no public/full-model-equivalence claim. Line 11 binds all four expected companions; lines 36–39 bind archive and sidecar identities. The generated checkpoint text at lines 53–55 describes completed E1 and the explicit path/runtime limitations. |
| `artifacts/README.md` | `1a9ee55a09d4b5b5bfbc6e89422a777df38d335f201e20516d1e61d22226cded` | Opening line 3 states completion and explicitly labels later dated entries as superseded history. The final E1 section (42–48) binds the reviewed E1 digest/count/size, states the 18 cells/230 captures/90 loaded modules, retains T1 and continuation divergence, and gives scratch-copy/path-mapping/external-runtime instructions. |
| `README.md` | `6a5f7ce2edbc2776681af3d5715e9ebe678b0c4358a9e5b8ce1c36a3a603c8ed` | Lines 3–5 describe the completed private bounded review and as-executed rate comparison; they do not claim equal outputs or quality-preserving causal speedup. Lines 21–30 distinguish historical archive verification, fresh execution chronology, and no-inference accounting/build commands. |
| `notes/claim-evidence-ledger.md` | `461aa6aca92a926301ae8ea756ecbf8cc1a20ebfd3cd842a9b1e677084533dfd` | Final timing row 21 records all 18 cells, the verified aggregate and coarse three-block precision results, with T1, 14/144 equal cross-arm streams, and the retained early stop. Lines 29–31 bind E1/private-package limitations; final disposition at 37–39 explicitly supersedes the 10:15 pending note and preserves deferred full-model, quality, causal, author-system and composed-stack claims. |

The older pending-state passages remain dated chronology rather than current scope. The archive itself remains unchanged at `bf50b33386eb9614cf7874039f60cb62a484ee62bc0882e32ab32731857318d6`. This closes the metadata items without requiring an E1 re-export or further experiment. Delivery-archive generation and its receipt are coordinator-owned packaging work, not an unresolved scientific or evidence-completeness finding.
