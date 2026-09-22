# Private review artifact red-team, round 1

Bounded read-only dependency review of the two private checkpoint archives. This is not a public-release audit or a request to rerun inference. No canonical source, snapshot, remote process, or GPU state was changed.

## Reviewed identity and verdict

| Artifact | SHA-256 |
|---|---|
| `artifacts/paper-review-20260922T0702Z.tar.gz` | `d12a5e80a447705ebd83aa25bcbbc9e995c1c63e84c88c5e5b4ad4737aa4b4c2` |
| `artifacts/e7a-evidence-20260922T0700Z.tar.gz` | `a1e4a4c833bff7f5c3ed6ba19d3afc2e6da893c17ad2063e8124dcb6243a9fcb` |
| Companion `.tar.gz.manifest.json` | `988ba930c737794390293c9c0ad4fcaf82465c2fe92c03ed488d987e7eba33a6` |

The main archive has 51 manifest-listed files plus `MANIFEST.json` and `REVIEW-CHECKPOINT.md`; the companion has 2,627 listed files. The package accurately declares private, incomplete status and explicitly excludes an unchanged portable full-GPU rerun. The main archive's companion digest matches the supplied companion. The coordinator separately verified all file hashes and extracted audit/build behavior; this review did not repeat those full checks.

**Useful for source/PDF review, H3 accounting reproduction, and E7a JSON-level numerical review, but not yet complete for the claimed captured-input/provenance review.** Three concrete dependency groups are omitted. All identified historical tensor and old-source omissions are recoverable from existing bytes, without reconstructing results or running inference.

## 1. E7a input and selection dependencies are missing

`scripts/export_completed_e7a.py` selects run roots manually. Its `ROOTS` omits the pilot continuation directory even though the frozen pilot and its result JSON reference that directory. Only 5 of 8 pilot tensors are present; all 32 confirmation tensors, including failed-provenance p087, are present.

Add this existing remote directory under the original relative path:

`experiments/out-20260922T010711Z-e7a-step2-captures-v4-pilot6to8/`

Its missing referenced inputs are:

| Capture / `logs/tree_gdn_capture_payload.pt` | Recorded SHA-256 |
|---|---|
| `capture_06_p095` | `e0b635931358f7738d03c895aab44816be1025270cbf10c35b9378cce8a59d81` |
| `capture_07_p058` | `3b411271f4e3e1064f99c959df302d37dfd9e292fd83b7bcb048e56c1dd3bc38` |
| `capture_08_p083` | `f2672729ea958f0b1f65190a8bbe875737f5199ad1f6d2324737712552675d74` |

Their original `capture_provenance.json`, self-tests, scheduler traces, and source snapshots are omitted with the same root. `experiments/e7a/PILOT_FREEZE.json` explicitly binds these captures and provenance hashes. The corresponding numerical results remain present in `out-20260922T013313Z-e7a-fresh-pilot-v2/result_06*`, `result_07*`, and `result_08*`, so their reported extrema can be read, but their underlying payload/provenance cannot be independently inspected from this checkpoint.

The archive includes frozen IDs, selected margins, and source names, but omits the prefix pool and native-scoring input to selection. `capture_expected.json` requires the pool hash; `frozen_prefixes.json` requires both pool and score hashes. Neither hash occurs in the companion manifest. Existing remote files were located and hashed directly:

| Relative path under remote v2 | Verified SHA-256 |
|---|---|
| `experiments/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json` | `05e731fd02b937970d05695ef2ee836a76efc463481a3a0ce4eeaa237f877d09` |
| `experiments/out-20260921T231853Z-e7a-step2-native-score/prefix_scores.json` | `2940202fa60523823340e3ea89809800bd9b9b1f75226b3306d3f5ce6b35e870` |
| Same scoring directory, `frozen_prefixes.json` | `8329672628828d35b4768732ed93b855335bc7576801890b0195957a2459973e` |

Include those two complete small run roots, preserving scoring/selection driver logs. This makes the 96-candidate native-only selection claim and capture prompt binding inspectable. Copying selected IDs alone does not supply the omitted 96-candidate data.

The four historical tensors used by the ladder-v3 run are also absent: `historical_payload_manifest.json` inventories them but does not embed them. The bundled `run_in_image.sh:10,51` establishes `/home/mark/shared/lumoFlyWheel/output` as the host source mounted at `/hist:ro`. All four files still exist remotely and were independently hash-checked:

| Path relative to that host output root | Verified SHA-256 |
|---|---|
| `fr13_replay_gpu_gates/boot1a_eager_capture_logs/tree_gdn_capture_payload.pt` | `9f1ddb6e3cfaabfba1e0b68b921d232723fe67eb8d4901537e2a0904fd846ea1` |
| `fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt` | `855b41b460c04ae978eb08868379c48e622abc5e673ccb23d968042803e383c0` |
| `fr13_l12_offline_replay_20260608T075019Z/tree/logs/fr10_tree_gdn_scan_l12.pt` | `77496f54e302a0bebc927ee835f8c95be45de042d0e874ded0bc85772203eb1c` |
| `fr13_wy_l0_localize_20260608T220307Z/tree/logs/fr10_tree_gdn_scan_l0.pt` | `c174bd740565aba4af147aed3262d26c85b938dce98b23d33369d8a1d139aa8b` |

Package those four used files with an explicit `/hist` mapping. The other historical inventory entries are not required to inspect the nine-set ladder reported in the paper.

## 2. The old ladder/tiny-gate source versions are omitted, but recoverable exactly

Both `out-20260921T230026Z-e7a-ladder-v3/manifest.json` and `out-20260921T231030Z-e7a-tinygates-v3/manifest.json` pin old `e7a_core` and `e7a_device` hashes. No file in either archive has those hashes. The packaged current files match the fresh-pilot/confirmation source hashes, so substituting them would silently change the older experiment source.

A bounded search found no matches in the remote v2 tree or remote Git history for these paths. **The local monitoring snapshot does contain both exact files** under:

`/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/p0/monitor/snapshots/20260921T2308Z/experiments/e7a/`

| File | Verified SHA-256 |
|---|---|
| `e7a_core.py` | `868d55db26eb5b83a43dc22010d639de2fe129c055a84bcd6f32a9f0e48409b7` |
| `e7a_device.py` | `70e475e3998c1d8b37761afa616ffe5c97a5ca02492366531fd13eee7012d693` |

Include these as versioned evidence sources and map them to the two old runs; do not replace the current fresh-run files. `e7a_kernels.py`, the production GDN source, and the vendored legacy WY source already have hash-matching packaged copies.

The two native reference modules are not copied into the package: `native_fused_sigmoid_gating` hash `000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f` and `native_fused_recurrent` hash `3a2a3c5245acba127af326f0b3f287b352776a8eb1e1f6e40c24a767fae3c2ef`. They are imports from the digest-pinned vLLM image, an external dependency already documented by the checkpoint. This does not invalidate JSON review or demand shipping the entire image. For self-contained source review of the gate-rounding attribution, copy those exact source modules or name their in-image locations explicitly; do not claim their bytes are already in this snapshot.

## 3. P0/source-reconstruction records and the literal frozen-audit citation are omitted

`scripts/build_review_bundle.py` includes `p0/P0-REPORT.md` and its measurement erratum but omits the report's own concrete supporting records. These paths are absent from both archives:

- `p0/historical-run-manifests.json`.
- `p0/stock-image/sampling-source.json` and the four `reconstructed-*.py` sources.
- `p0/model-weight-hashes.json`, `p0/remote-readiness.json`, and `p0/fa2-load-check.json`.
- `p0/monitor/2026-09-22-review-14-frozen-audit.json`, the literal audit path cited by the manuscript.

The first two bullets are needed to inspect the manuscript's P0 claim that three historical source reconstructions show two constraint passes and the selected current reconstruction shows one. The package currently provides only the report and a script that depends on omitted inputs. Include the original P0 records and source-reconstruction subfolder; label external image/Git requirements for actually regenerating P0 separately.

For the frozen audit, the companion already includes `out-20260922T064200Z-e7a-freeze-retrospective-audit/{pilot_v2_retrospective,confirmation_retrospective}.json`, the v2 checker, controls, and erratum. Thus the failed-rule conclusions are inspectable, but the manuscript's literal cited file is unresolved. Include that original file or add an explicit equivalence/path mapping. This is not a demand to replace immutable original verdicts.

## What is already resolvable, and the next refresh

The manuscript source/PDF, bibliography, figures, H3 raw inputs, proxy/support tables, and honest measurement erratum are present. The nine-set, tiny-gate, eight-prefix, and 31-primary-prefix E7a result JSON are present, with the original confirmation freeze and retrospective failed-check analysis. The paper correctly disclaims deployment qualification and external-system superiority. These strengths remain valid despite the input/source omissions above.

Repair the omissions by copying existing hash-matching bytes into a **new** immutable checkpoint and updating the explicit root list/path mapping; preserve these two original archives. No GPU run is needed for this repair.

After E2/E7b and E1, refresh only with the actual qualified or failed results and their frozen criteria, complete attempted-case denominators, exact runtime source/configuration manifests, raw outputs, and matched event-ID timing evidence required by E1. Update the manuscript, status/closure ledger, source hashes, archive manifest, and cross-archive digest together. The exporter currently snapshots only enumerated E7a roots, so adding final E2/E1 claims without adding their dependency roots would repeat this omission. If qualification remains incomplete, retain that explicit status. Model weights and JIT-cache bytes need not be bundled merely to support the deliberately limited private review checkpoint.

## Resolution spot-check — omission findings closed in the new pair

The coordinator produced a new immutable pair, preserving the original archives. This bounded follow-up checked the exact dependency gaps above, not a full GPU replay or a repeat of the coordinator's complete file-hash verification.

| Artifact | Verified archive SHA-256 |
|---|---|
| `artifacts/paper-review-20260922T0710Z.tar.gz` | `3676083f2e7b3c7b31ce1648ddebbf9dfcb804bf80363bb43261e8c79ed6c8ea` |
| `artifacts/e7a-evidence-20260922T0709Z.tar.gz` | `776ddcaf5d938cc665bafca20f46e29d1e9a550a34914d798e78f70636cd180b` |
| Companion `.tar.gz.manifest.json` | `50dd06ea612e00caec86e84db0b49321d3fd9878e8e4f79d68bb95dfa88c0e7d` |

**All three identified omission groups are resolved for the private review checkpoint.** The main manifest lists 80 files, and the companion lists 2,859. The main manifest names and hashes the correct new companion. The packaged manuscript remains the reviewed `main.tex` hash `5e579cbf64ed4211efafbbdd9e15acc8ba99a53574162fdc12e6069178255b76`, with abstract hash `2952d72ce3c00fa1f425b4ef63abbb8324dd2f5180261402c24576ace10f7e19`.

- All eight frozen pilot payload hashes and all eight associated provenance hashes now resolve in the companion manifest, including p095/p058/p083 under their original `pilot6to8` directory. Directly hashed the packaged p095 tensor as a byte-level spot-check; it matches the immutable freeze.
- The native selection pool and score files resolve by their recorded hashes. Both packaged files were directly rehashed and match `05e731fd...` and `2940202f...` above.
- All four historical input hashes now resolve under `historical-inputs/output/`. The companion manifest's `historical_input_map` explicitly records original `/hist` path, host source, archive path, and hash for each. Directly rehashed the packaged layer-1 historical tensor; it matches `855b41b4...` above.
- The exact old `e7a_core.py` and `e7a_device.py` versions are included in the paper archive under `p0/monitor/snapshots/20260921T2308Z/experiments/e7a/`. Both packaged files were directly rehashed and match the old run manifests. The matching production GDN source also appears in the companion's earlier `out-20260921T222616Z-e7a-ladder/source_snapshot/prod_fr10_gdn_tree_kernel.py` with hash `d9dd0c69...`. The README explains this version mapping.
- The paper archive now includes `p0/historical-run-manifests.json`, `p0/stock-image/sampling-source.json`, all four reconstructed sampler sources, model-weight hashes, remote readiness, FA2 load record, and the manuscript's literal `p0/monitor/2026-09-22-review-14-frozen-audit.json` reference.

No new material dependency omission was found in this targeted recheck. The external digest-pinned native runtime, explicit local path mapping, excluded weights/JIT caches, and lack of an unchanged cross-host GPU rerun remain declared scope limits; this closure does not upgrade the package beyond a private incomplete review checkpoint. Final E2/E7b and E1 evidence and the corresponding manuscript/status refresh remain separate open work.
