# Q1 continuation fixtures v2: bounded independent review

Reviewed 2026-09-29 UTC. **Accepted as prospective CPU fixture preparation; no concrete blocker found in this delta.** V2 closes the v1 input-coverage gap for the declared candidate 1024-token page geometry. This is not a runtime result, a qualification denominator, or permission to launch. No GPU, model, Docker, remote, cache, implementation or gate operation was performed.

## Exact reviewed bytes

Campaign-relative paths unless otherwise noted:

| Artifact | SHA-256 |
|---|---|
| `tools/q1_continuation_fixtures_v2.py` | `cf09893afa0078f053cbc2185572c98c03aa0abac62718c532ec9899ade19879` |
| `fullmodel/fixtures/token-fixtures.sequences-v2.json` | `e9683d2f8097ecfe195331d9977f9e57e1ab8482f1d583e04c5268ad92777318` |
| `tools/tests/test_q1_continuation_fixtures_v2.py` | `dd254fd5fdc5413d5629de5e2140003df8430c6ad9aaea5950ebad30baaaa853` |
| Preserved v1 generator | `16f85ffff80008fdd396e48c9b1e0387770b7b4ee8e0611de566fa05abf8ce9e` |
| Repo-root `results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json` | `37ed4ff59f6c0d6f9ae482916e0c5a5939f38f70d80756fdb43a44101d6dde19` |
| Repo-root `scripts/fr13_patch_fa2_tree_bias.py` | `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2` |

The saved artifact's canonical seal is `c1c0202ce6db695ccd5fc6be9196e0ab99f0ece9ad8450997e4c545446e2576e`. Independent reconstruction matches this seal and the complete rebuilt artifact excluding its timestamp. Records are covered by this whole-artifact seal; the package does not claim separate runtime-record seals.

## Checked closure

- **Scope preserved:** the same 48 unique record IDs remain. All 30 F2/F3 records are identical as parsed objects, including pending-token continuity, aliases, reset restrictions and split labels. Only the 18 F4 records change; differences are confined to padding, derived positions/extents/hash, and explicit geometry fields. Their accepted path, physical row tokens, pending token, scenario IDs and offsets remain unchanged. Counts remain 24 calibration plus 24 held-out evaluation records and 108 cycle descriptions before process/repeat expansion.
- **Both physical boundaries covered by inputs:** every F4 record starts at the next 1024-token boundary minus the already-declared 1, 3 or 11 tokens. The 12-token accepted path has materialized positions on both sides of that boundary, and also crosses a 64-token native kernel block. This was checked directly from the saved positions, not merely from the generator's exclusive-end extent comparison. Coverage is 18/18 for each geometry.
- **Padding and isolation:** padding is the minimal nonnegative addition for the chosen offset, always less than 1024 tokens (observed 322–962). It is drawn deterministically from valid, non-added-token values in that same source prefix. Original prefix bytes/hash, split and held-out labels remain bound. All padded-prefix hashes reconstruct exactly; calibration/evaluation source hashes and padded-prefix hashes remain disjoint. Materialized prefix lengths span 14,325–60,415. All physical token rows, pending tokens, topology-depth positions, contiguous consumed positions and pending-token extents pass the independent artifact checks.
- **Geometry provenance remains prospective:** generator lines 20–23 require the exact v1 source, parent credential and production patcher hashes before constructing the successor. Credential lines 225–230 declare `page=1024` and `block_n=64`; the patcher's fixed B1 route requires `page_block_size == 1024` at line 1247. The accepted native record remains exact-hash bound through v1. These establish intended/source-supported geometry, not an actual candidate tensor observation. Generator lines 35–42 and the saved artifact correctly retain `candidate_page_runtime_confirmed=false` / `candidate_runtime_confirmed=false`.
- **No qualification promotion:** `qualification_denominator` remains null; launch and candidate-qualified flags remain false, including every record. Status remains `SOURCE_FIXTURES_ONLY_NOT_RUNTIME_ADMITTED`. Neither geometry crossing is reported as an executed outcome. No new workload task or numerical criterion is introduced.

## Reproduction and remaining boundary

Independently ran `python3 -B tools/tests/test_q1_continuation_fixtures_v2.py`: **2/2 CPU test methods pass**. Separate direct saved-artifact checks verified the exact population, allowed delta, both materialized boundary crossings, token/padding validity, position semantics, split separation, canonical seal, deterministic rebuild and all seven declared source hashes. Mutating each of the v1-generator, credential and production-patcher expected hashes independently refused construction. Audit: `q1-continuation-fixtures-v2-independent-audit.json`, SHA `c9e1e6b232b8f95434d60dc1b4c76bfe917fc95aa17fb426f0a3060687cfbbb9`.

The v1 artifact and its native-only scope correction remain historical. A future runtime job still needs to authenticate this exact successor and confirm both routes' actual tensor/block-table geometry; the prepared values do not supply runtime acceptance. No additional source fix is required for this bounded preparation delta.
