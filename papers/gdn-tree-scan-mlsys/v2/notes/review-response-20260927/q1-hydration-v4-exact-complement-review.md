# Hydration v4 exact-complement closure review

Disposition: **PASS for the bounded source repair and root-only successor wiring**. No remaining blocker found in this delta. This is CPU/source acceptance, not a GPU result, gate authorization, successful hydration, or full-model qualification. The original 20260929T083835Z failed run and its failure review remain preserved.

## Exact repair

`tools/q1_candidate_hydration_v4.py` SHA256 `b35f2e5d49e0f49827103db3b0c3ddecdec6ce38189883a7c4a4c13c79350fac` preserves all selected-destination disjointness checks and constructs the authorized union from those actual storage spans (lines 145–173). It then subtracts only the intersecting bytes from each previously declared guard. Guard views are recursively decomposed into contiguous **views of their original allocation**, not contiguous copies with invented addresses (lines 48–100). This handles the observed cross-family allocation sharing symmetrically.

For the recorded geometry, row 52's convolution guard is wholly covered by a legitimate attention import. Row 56 loses exactly 358,400 authorized bytes while retaining **337,920 guarded bytes**. The reverse full KV page 53 guard retains **634,880 + 352,256 = 987,136 bytes** around the authorized convolution and SSM writes. Independently executed controls confirm that changing an excluded byte leaves the guard digest unchanged, whereas changing an adjacent protected byte changes it. A noncontiguous two-piece control confirms that physical gaps are not accidentally packed into or excluded from the guard.

Selected convolution padding cannot overlap any planned write (lines 82–86), and the unchanged pre-write zero-padding test still examines its full tensor view (lines 223–227). Destination collisions are refused before writes; source objects are authenticated before the transaction; every imported object is reread after all copies; the complete native-order and per-layer KV digests still match; runner identity is compared after all guard checks. Failure after mutation remains fail-stop. The new guard exception does not waive destination readback or whole-native-digest equality. Source and guard scope explicitly retain the bounded-neighbor limitation: this does not protect every cache row, distant page, other tensor, or MTP state.

## Powered controls and limits

Independent local standard-library controls executed the actual new functions extracted through AST, with address-preserving byte-view stubs: 2,000 randomized interval comparisons against a pointwise set oracle; observed row-56 and reverse page-53 residuals; protected versus permitted mutations; strided gaps; pointer rebinding; padding overlap; full coverage; invalid intervals; and AST equality of all unchanged v3 helper functions. All passed. Source-authenticated test scripts and results are saved in `p0/monitor/review-response-20260927/hydration-v4-complement-review/`.

Separately, I inspected the parent's connected Torch CPU test and its preserved attempt-2 log: **7 methods passed**, using one shared allocation, real recorded strides/rows/pages, and sealed native source objects. It reproduces the old v3 row-52 failure, accepts v4, detects residual/authorized-destination corruption, rejects padding/collision violations, and retains registry failure detection. It intentionally reduces 48/16 layers to 3/1 and disables the full alias-count requirement for this geometry fixture; it is not a test of all 16 alias classes or GPU execution. The parent-performed test was not rerun locally because this reviewer interpreter has no Torch. Attempt 1 remains preserved; its fixture alias mutation was fixed with a list copy.

## Root successor integration

The root hooks/job/patcher v1.4 changes are limited to importing/pinning H4 and propagating the version. The wrapper v2.9 binds H4, those successors, and raw auditor v1.2. All old root-path selector literals are absent from the new wrapper. The generated diagnostic v3.2.3 is byte-for-byte its predecessor after exactly five patcher v1.3→v1.4 literal replacements; no launch settings or scientific rules change. The parent patch receipt shows six callbacks applied and a second application refused. These are source/CPU observations, not live execution evidence.

Raw auditor v1.2 requires the v1.4 root record plus the explicit guard rule, positive integer interval/protected-byte counts, and nonnegative integer overlap count. I independently executed its actual new predicate: the positive passed and **29 missing, wrong-type, nonpositive, or wrong-rule variants refused**. This receipt summary is source-bound bookkeeping, not independent raw reconstruction of untouched cache bytes; it must not be represented as a standalone whole-cache proof.

A future parent-authorized attempt may retain the existing scope: one candidate process, shortest calibration prefix, root-only path, two repeats, unchanged serving settings. Bind the reviewed bytes and fresh operational authority. This review opens no gate and does not extend the accepted root scope to cycle-0/all-path or MTP/lifecycle qualification.

## Snapshot

Full exact hashes and sizes for the 12 reviewed source/test/log files are in `FINAL-SNAPSHOT.json`. Key successor hashes:

- hooks v1.4: `386307beea80b2162dd11e4030a5ea59c0f708d3e7393cc0250284dce292f35e`
- job v1.4: `8b48a94f4df9796175419d3866c050f8eae9c2bd74ad8311de90a1fe4a964218`
- patcher v1.4: `f263479e4b2f0a9d80a79ac2010520915809cc026d38fd3b9b80f2801c1ec95d`
- wrapper v2.9: `1fa985850ca9fd78eff9dce52e98976ee8c4301b3d964eb72868215cfdf9c452`
- generated diagnostic v3.2.3: `6bcbc3147c298308165b4357942e96a969fab4a67e4d0e05e5bf6d102d5495df`
- raw auditor v1.2: `2de897c9e5127f496a8cc1994f34275fb2fa3e6c293e2c5342d58a7606f40e8d`
