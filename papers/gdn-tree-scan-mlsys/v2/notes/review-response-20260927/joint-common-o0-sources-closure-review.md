# Joint common-O0 source inventory: F1 closure

2026-09-29. **Bounded CPU/source PASS: the prior run-authentication/census finding is closed.** This supersedes the disposition, not the preserved observations, in `joint-common-o0-sources-review.md`. No GPU run, source collection, network, container, model, or scientific reduction was performed; no implementation was edited. The source plan is still prospective and grants no qualification or launch authority.

| Reviewed item | SHA-256 |
|---|---|
| Builder `q1_joint_common_o0_sources_v1.py` | `35bcbddb1ffc7c916238bf07390cb82643fc13b3f95a1d70a6678c2b3d3bf85a` |
| Unchanged resolver tests | `9fc47af944751564da599aaf1dc57af02cfb4fe8ee2789c2a63a52309df9ab6a` |
| Unchanged `SOURCE-PLAN.json` | `c399fee087851a95ca3d0d99ab42457174b2cd46f05ad84ae7aa3b21d10f25b6` |
| Unchanged joint source launcher | `dd41001579d994348d09ea3e2c72e307cd7764661efeadfad66c04cad42618b9` |
| Unchanged joint source driver | `3ea559dd482a09862be15afcf611b3853adf8dca83ce48abd3821e30179a9c8f` |

The new `run_binding` (lines 14–62) is called at line 72 before raw object/seal consumption. It validates all terminal member hashes/sizes and safe paths, requires the relevant source/launch/driver records, binds the retained plan and exact job to the gate and reviewed source identities, and requires the exact three case and driver-request filenames. The driver must report successful 3/3 completion; all three request receipts must carry the correct case/repeat/observation and prefix-token parity. Contradictory cleanup refuses. The existing launcher's actual success strings and retained receipt field names agree with these checks.

The added actual image/container/configuration and boot checks are consistent with the reviewed renderer and launcher: exact image/CID/name and stopped state, original command/environment, zero restarts, declared mounts and memory limits, native spec-off 48-GDN/16-attention route, FA2 binary/installation, and exact patch output plus full MTP patch detail. A CPU reapplication of the pinned string patch yields `205799b54ec8bbae7641f8f5e47f88d78d43adbe8b4a512c0866bc604274db10`. This inspects source transformation only; it does not import or execute the model runner.

**Independent controls:** a full synthetic run header chain built from the actual Plan/job/config and patch-text functions passes and reaches exactly three raw-audit calls. Twelve negatives all refuse before any raw-audit call: missing terminal membership, contradictory cleanup, changed member hash, missing required member, extra case, extra driver request, missing driver verdict, rehashed incomplete driver census, false prompt parity, foreign launch source, wrong full MTP patch detail, and path escape. The expensive raw audit/snapshot functions were explicitly injected successful seams so these controls isolate the repaired outer boundary; no claim is made that synthetic headers qualify real tensors. The original seven resolver tests also pass.

AST comparison confirms `source_for_case` is unchanged. Removing only the new `run_binding` call and returned binding from the new builder gives the identical original `build` AST. Thus target/MTP raw audits, source selection, mapping and numeric behavior are unchanged. The plan still maps 84 cases to three predetermined roots, 28 per prefix, and retains the separate 336-observation qualification denominator. All reviewed source hashes matched again after the checks.

Review sources, diff, controls and output are preserved under `p0/monitor/review-response-20260927/joint-common-o0-sources-independent/repair1/`; the initial failed snapshot/results remain at its parent. Run the independent controls with:

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/joint-common-o0-sources-independent/repair1/independent_controls.py
```

Future integration must still pin the resulting manifest and source bytes in parent-owned authority and retain real raw-audit/import/readback evidence. This closure does not admit an unapproved source run or establish native repeatability, candidate correctness, timing, or workload performance.
