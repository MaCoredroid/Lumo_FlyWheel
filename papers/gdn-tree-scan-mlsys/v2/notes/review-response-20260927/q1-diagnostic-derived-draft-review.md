# Q1 diagnostic derivation: bounded draft review

Reviewed 2026-09-28, CPU/source only. Snapshot: `p0/monitor/review-response-20260927/candidate-diagnostic-draft-20260928T0556Z`; its `PARENT-SNAPSHOT.json` SHA256 is `6e6c04a9eb7f78cb7479588b53117141a6a568441a09a6186d9996a5be59da69`. This is the preserved 13-file **unfinished worker draft**, not a freeze or launch approval. No remote operation, Docker, Torch, CUDA, GPU, source edit, or gate mutation was performed.

**Verdict:** the reviewed diagnostic credential/source lineage is coherent and retains the original scientific checks and evidence. One narrow source correction is needed for the generator's promise that its unlabelled branch behaves like production. This finding does not demonstrate a bypass in the intended exact-label diagnostic wrapper path. Missing final caller integration and freeze are expected unfinished work, not new regressions.

## Concrete correction: selector expansions are not label guarded

`tools/q1_make_diag_launcher_v3.py:264–273` replaces three production literals with `${Q1_DIAG_PATCHER_HOST/IN:-production}`. `guarded_expansion_ok` at line 350 validates the textual substitution, but not its dependency on the diagnostic label. The resulting generated launcher uses these expressions at lines 2561, 7780 and 7862 outside diagnostic conditionals. Its introductory assertion that these variables are unset whenever the label is unset is not enforced.

A bounded shell probe, executing only parameter expansion, reproduces the issue:

```sh
unset FR13_Q1_DIAGNOSTIC
Q1_DIAG_PATCHER_IN=/unreviewed/alternate.py
printf '%s\n' "${Q1_DIAG_PATCHER_IN:-/workspace/scripts/fr13_patch_fa2_tree_bias.py}"
# /unreviewed/alternate.py
```

Thus an inherited selector can change the generated script's unlabelled patch-source validation or executed patcher. The original production launcher file itself is unchanged. Repair by assigning private selector values to the production paths at entry, then assigning derived paths only inside the exact-label branch; alternatively, explicitly refuse inherited diagnostic selector variables on an unlabelled invocation. A two-case CPU control should establish that an unlabelled inherited selector cannot change the selected production paths, while the exact labelled route still selects the reviewed derived paths. No numerical change is required.

## Checks that passed

- The parent credential exists locally with its required SHA256 `37ed4ff59f6c0d6f9ae482916e0c5a5939f38f70d80756fdb43a44101d6dde19`. The actual bounds file hashes to `ee49c3a712971f81509617bbd3f7cabffa5f43c74cd9db852459a9a758c1958e`.
- Removing only `credential_sha256`, the added `derivation` block, and `identity.patch_source_sha256` makes the parent and derived JSON bodies exactly equal. Both canonical body digests validate. The arm remains `gqa_pair_splitk`; the FA2 binary SHA remains `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`. All measurements, bounds, probe, determinism, selector and other identity values are preserved. No new measurement is represented as having occurred.
- Executed the **original frozen** `validate_tierb_credential` on both real credential files, passing each actual patch-source digest and the actual bounds file: both pass bounds and probe-strength checks. AST comparison of original versus derived sidecar finds all 35 existing non-`main` functions unchanged, including the scientific validator and binary checker. Only the CLI dispatcher changes, alongside newly added lineage helpers.
- Applying the recorded patcher delta strictly to the pinned parent reproduces the derived source byte-for-byte. Its four hunks have no removed lines and fall inside the declared original-source windows 6080–6168 and 7491–7589. They add the diagnostic workload, exact label/manifest checks, and record fields. They do not change kernel, dispatch, geometry, numeric bounds, or existing workload entries.
- Executed the isolated actual workload-table/helper block with standard-library environment and file-read fixtures. The exact diagnostic label and manifest bytes yield zero task IDs and `is_swe_subset=false`. Six negatives refuse: missing label, wrong label, wrong manifest digest, altered manifest bytes, fresh SWE task IDs, and legacy SWE task IDs. All six original workload cases return exactly the same records from original and derived source.
- Rebuilt launcher v3 in memory using its generator and the hash-verified v2 generator library. All 17 edits reproduce the generated file exactly; reversing them reproduces production launcher SHA256 `11b604b5fc9e652c9a59840511aea9e0952bd56e817d5f27f0ab9bb66bae1313`.
- In the intended diagnostic path, the host checks exact label and fixed manifest digest, rejects legacy/fresh SWE pins, pins parent and derived artifacts, and checks the actual derived patcher's admission table before container creation. Generated lines 7741–7768 invoke the new lineage verifier before the original `verify-tier-b`; the latter validates the derived patch source at line 7780, and line 7862 executes that same source. Sidecar `verify_tierb_derived_credential:705–891` checks actual binary identity, both original scientific validations, both source hashes, exact additive delta reconstruction, manifest arm, parent/derived lineage, and equality of every unchanged credential field. This is a transparent derived credential, not a replacement PASS assertion.

The local checks do **not** constitute an executed full `verify-tier-b-derived` against the installed binary, nor a container/model boot. These remain runtime obligations. The later caller must explicitly bind the derived credential/source route: presenting the old parent credential intentionally takes a refusing path. This is expected integration work in this unfinished package; it should not be “fixed” by weakening the validator or substituting a SWE identity. The accepted wrapper/env and bank-view closures remain closed.

## Reviewed source identities

Paths below are relative to the campaign inside the preserved snapshot.

| File | SHA256 |
| --- | --- |
| `tools/q1_make_diag_patcher_v1.py` | `d6bf784cd02f72cc9f7ce0ec19ec53ca30461c44d1464a8904e56f5a911d2d1c` |
| `tools/q1_derive_tierb_credential_v1.py` | `bdb720b51049b6a417ecf81fdc7df57b9377037977935c6a604961969a4c7178` |
| `tools/q1_make_diag_launcher_v3.py` | `b6bfb3a0f7c2f684ee7418604b241c65c7f8fb7bbc9a193177c913ad35429d0e` |
| `tools/q1_make_diag_sidecar_v1.py` | `1a9260b1bf3556f9460d9700868003936104efea8e9e5096f31361ecbd0e3d74` |
| `tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v3.sh` | `a68219ba53891382ba613534a5a88bc54888099ecc61146367cc0c6d0aa3d2cd` |
| `tools/generated/q1diag-patcher-v1/fr13_patch_fa2_tree_bias.py` | `782d6dd81ec0a9a159f23435870f5c5d36082bc8274c0acdc6405374469a9c49` |
| `tools/generated/q1diag-sidecar-v1/fr13_qrow32_b1_pass_sidecar.py` | `1210470d728372fb5cfd62b780701ef79182c4ffe566505b2621b854eb4fe945` |
| `identity/fr14_splitk_tierb_credential.q1diag.v1.json` | `0a4f7576db4e7e35080b0865e4cdb04fd652a29466b66cd0a8cebb1843664453` |
| `identity/patcher-diagnostic.v1.diff` | `531809201ed315824e9634647cd23ee67883b0a907311b2a6ea1210e18212b36` |

Parent patcher SHA256: `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`. Parent sidecar SHA256: `affea02e8b99a79dea3fe9fc15395a34d91df57a45398f048b5ddd514a12880c`. Parent diagnostic manifest SHA256: `674aec8a042f57078e70a8aef228f07c8db4c307b571cad59386f5496b5eae6a`.
