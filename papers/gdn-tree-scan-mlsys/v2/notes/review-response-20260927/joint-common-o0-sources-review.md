# Joint common-O0 source inventory: independent CPU review

2026-09-29. **One bounded run-authentication/census gap remains.** The deterministic plan, selected source raw-audit call chain and resolver identity checks pass this source review. The builder alone does not authenticate the declared completed three-source run. No source collection, model call, network, GPU, launcher, or reduction of real source tensors was performed. No implementation was edited.

## Reviewed identities

| File | SHA-256 |
|---|---|
| `q1_joint_common_o0_sources_v1.py` | `ae3094912e8edff4604f783c433240f0ecf27e7375fe924785a42f045826c2cb` |
| `test_q1_joint_common_o0_sources_v1.py` | `9fc47af944751564da599aaf1dc57af02cfb4fe8ee2789c2a63a52309df9ab6a` |
| `q1_joint_source_plan_v1.py` | `039809d05865eb38756a595728b1bc285f453bc5b46b4e283e2d52836794aa4b` |
| `q1_reference_driver_joint_source_v1.py` | `3ea559dd482a09862be15afcf611b3853adf8dca83ce48abd3821e30179a9c8f` |
| `q1_joint_source_v1.py` | `2fb11fd09b50c421fc192146790ca5aa0edbf78e4b57ed0b73c48a292f1b151e` |
| `run_q1_native_joint_source_v1.sh` | `dd41001579d994348d09ea3e2c72e307cd7764661efeadfad66c04cad42618b9` |
| `SOURCE-PLAN.json` | `c399fee087851a95ca3d0d99ab42457174b2cd46f05ad84ae7aa3b21d10f25b6` |

## F1 — terminal success strings do not authenticate the exact completed source run

`q1_joint_common_o0_sources_v1.py:18` checks only `status == COMPLETED_driver_rc=0_cleanup_rc=0` and `driver.exit == exit=0`. Those strings are correct for the actual launcher, but the builder never checks the actual terminal member inventory, cleanup-state field, driver verdict/individual request receipts, or exact `q1_ref/cases/*.json` population. It records the terminal receipt hash at line 32 without validating those contents.

The existing launcher emits a richer receipt (`run_q1_native_joint_source_v1.sh:125–144`), and the existing driver emits both per-request receipts and a final census (`q1_reference_driver_joint_source_v1.py:133–161`). The unchanged native corpus reducer already demonstrates the corresponding run/member/census checks (`q1_native_corpus_v1.py:305–379`). Raw target/MTP seal authentication is valuable, but it does not prove those outer run facts.

A narrow independent CPU control retained the real current Plan/job equality checks and injected successful raw-audit/snapshot functions to isolate this outer boundary. It produced a manifest with three sources in all three cases:

1. Terminal contains only the two success strings: no member map, driver verdict, or driver request receipts exist.
2. Terminal additionally reports `engine.cleanup_state = query_failed` and a zero-length/wrong-hash job member.
3. The case directory contains a fourth invalid observation JSON beyond the three predetermined cases.

These are receipt/census counterexamples, not evidence that invalid tensors pass the raw numerical auditors. No real source outputs exist for this review.

**Minimal resolution before source use:** require the exact three case-file names; the successful 3/3 driver verdict and the three request identity/parity receipts; hash/size verification and required membership in the terminal inventory; and the retained plan/launch binding naming the same run, job, source-plan hash and reviewed executable identities. Preserve the existing target/MTP raw validation unchanged. Alternatively, explicitly require a separately authenticated source-run review receipt that establishes these facts before admitting the resulting source manifest. This does not require another GPU experiment or turn source inventory into qualification.

## Checks that pass

- Actual `SOURCE-PLAN.json` SHA `c399fee087851a95ca3d0d99ab42457174b2cd46f05ad84ae7aa3b21d10f25b6` exactly equals a fresh CPU `Plan.build` using the existing frozen token fixtures and current pinned helper sources. The complete original 84 cases map 28 each to the three predetermined root-only calibration sources. No outcome-based selection is present. One A process, repeat 0, three source requests is clearly separate from the unchanged qualification denominator of 84 cases × two processes × two repeats = 336 observations.
- The builder requires the expected root namespace and exact declared job, excluding only its construction timestamp. It invokes the existing `Driver.authenticate_seal` for every selected source with run/job/observation/case/arm/process/repeat identity; that function authenticates target raw objects, native MTP raw history, and the same-request joint boundary. `NC.snapshot` then checks the selected target O0 geometry/layers, logical digests and finite raw bytes.
- Resolver checks bind the selected record file hash and internal seal, source run/job/observation identity, prefix bytes/length, root token, target/MTP/joint digests and native-prefix-history hash. Different later path suffixes are intentionally allowed because all 28 paths for a prefix reuse its same pre-root state. The supplied seven controls pass, including altered root/prefix/job/history, duplicate mapping and traversal.
- `source_inventory_is_qualification` and `launch_authorized` remain false. This source preparation neither establishes native repeatability nor candidate qualification.

## Trusted future integration boundary

The future caller must independently pin the resulting manifest, fixture/plan/job identities and source-root provenance; `source_for_case` accepts a parsed manifest rather than authenticating an approval itself. Its input case/prefix must be the frozen destination case. Actual tensor hydration must continue to authenticate source object bytes and destination import/readback through the existing importer. These are explicit future wiring requirements; no live-ready claim is made here.

## Reproduction

The seven supplied controls passed:

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_q1_joint_common_o0_sources_v1.py -v
```

The independent control also verifies exact current plan regeneration and its 28/28/28 mapping, then reproduces the three outer receipt/census acceptances with an explicitly stubbed raw-audit seam:

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/joint-common-o0-sources-independent/independent_controls.py
```

Source copies, `SNAPSHOT.json`, supplied output, independent script and `independent-results.json` are retained under `p0/monitor/review-response-20260927/joint-common-o0-sources-independent/`. Only reviewer-owned artifacts and temporary fake records were written.
