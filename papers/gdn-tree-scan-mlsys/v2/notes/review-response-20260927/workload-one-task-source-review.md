# One-task workload source review

Disposition: **PASS for this bounded source change**. No blocking defect found in the reviewed task binding or generated v2.2 launcher. This is not launch approval, full-model qualification, WP-gate approval, or acceptance of the evolving private boot-pin/caller implementation. The reviewed task is `scikit-learn__scikit-learn-9288`; its workload token is `review-response-20260927-sklearn9288`.

## Source and evidence

The independent reviewer copied and hashed 16 inputs before testing, and rechecked every live source hash afterward. Snapshot and controls are under `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-one-task-source-review/`. No source drift occurred during these controls. Only local stdlib, extracted Python AST/template functions, and extracted Bash statements were executed; no Torch/model import, SSH, Docker, GPU, engine/proxy invocation, credential-secret read, gate change, or workload attempt occurred.

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `LUMOTREE-WORKLOAD-OWNED-v2.2.json` | `f19367825efa463d518525579e6dbade3dae6b2d5a64277b12238c5ed3704c8e` | 1037 |
| `lumotree_workload_owned_v2_1.sh` | `3d186ac2f44781527f659ac2b4a369d0f8110c7c82f887b8386d846486060943` | 498414 |
| `lumotree_workload_owned_v2_1_to_v2_2.diff` | `371b946c03a9d20e87a0caa7baf1cd23a0ae80afc5a8e1605243f501ac925693` | 8757 |
| `lumotree_workload_owned_v2_2.sh` | `eb8143836beddc5446b132381ad39c5c4625c89b873aa4b4d7b5566f0b5a6d5d` | 503007 |
| `make_lumotree_workload_owned_v2_2.py` | `d28d71834da69602b8601c405e1a88eedf9b6ebe19d28e03e3aae4052a1f81ef` | 6094 |
| `LINEAGE.json` | `5d5cb9fa2f1263d0f533cecfd9477889a5c1671fa6b883c24867445766383a00` | 1005 |
| `SOURCE-VERIFICATION.json` | `231bdb3ceca0c9bc09def467c437f13cbe3af1244bb3a745395b7319ae9b3323` | 20327 |
| `credential.json` | `909758655c36684c2ed909600d8b75acd75a3d197012bef17a448c75c1edbf62` | 17234 |
| `fr13_patch_fa2_tree_bias.case-study-v1.py` | `5ac584f7a6726427132fdb3cfa0e57c366476669b6632a362d2a1afe07914874` | 481354 |
| `subset.json` | `7098e0015b73a66c058b5885a5b5e6d1cc0cdf86c1f0d301ff6bba71c999a4b6` | 220 |
| `task_binding.py` | `0d0af3e28eab69a14c0fe7b671a355aac4194c3ddc655ff454ba7e9276565847` | 6448 |

The five inherited inputs are also recorded in `SNAPSHOT.json`: approved SCOPE, original patcher, original credential, original bounds and unchanged sidecar validator. Their hashes exactly match `task_binding.py:PINS`.

## Findings

1. **The scientific patcher changes only one keyed workload identity.** `task_binding.py:57–80` authenticates the parent inputs and the approved four-arm/single-task scope, builds the exact one-task SWE-bench Verified subset, and inserts one 149-byte entry after the unique workload-table anchor. Removing that insertion reproduces the parent patcher byte-for-byte. Every prior table entry and every other source byte, including template functions and numerical logic, is unchanged. The new table entry is at derived patcher line 6081; the unchanged `_fr13_fa2_qrow32_b1_tier_b_workload` at line 6114 checks the declared token against exact task IDs and subset digest, including conflicting legacy-name refusal.

2. **The credential is a transparent source derivative.** Relative to the pinned parent JSON, only `identity.patch_source_sha256`, added `case_study_source_lineage`, and `credential_sha256` differ. Numerical measurements, determinism, bounds, binary identity, grants and other scientific fields remain equal. `task_binding.py:84–106` compiles the exact hash-checked original sidecar and invokes its actual `verify_tierb_credential_file` for both documents. Both pass. The independent result exactly reproduces `SOURCE-VERIFICATION.json`. Lineage is enforced by the helper's exact recipe check; the unchanged numerical validator does not independently interpret the added lineage field. The launcher invokes both checks, so this division does not omit lineage validation.

3. **Host and container admission agree on the selected task.** Generated launcher lines 82–93 require exact task, workload, ingress IDs, tier-B IDs, subset/source/credential hashes, empty legacy aliases, and the two expected credential/private-pin paths before the helper verification. The new host-table row at 1234–1238 hashes the actual subset file through the unchanged function ending at 1313. The count branch at 5488–5490 requires exactly one selected ID; the prior format/uniqueness checks remain in place. The container table independently checks exact identity. Prior production table entries remain intact, while this dedicated launcher preamble intentionally admits only the case-study task.

4. **The selected patcher reaches both actual container commands.** Launcher lines 102–103 assign the fixed derived host/container paths. Host source hashing uses the selected host path at 2673. The container helper runs at 7896 under the existing `set -euo pipefail` command (7652); the unchanged scientific `verify-tier-b` at 7907 passes the derived `--patch-source` at 7914 and its expected SHA at 7916. The patcher invocation at 7996 uses the same derived path and the tier-B serve flag. Independent two-stage shell expansion and inert `python3` shell-function argv capture verified the exact path in both commands; no actual Python patcher or verifier command was executed by those shell controls. The inserted paths/tokens contain no shell metacharacters, and the new comparisons quote variable expansions.

5. **No middleware or scientific relaxation is introduced.** The entire generated launcher/diff/manifest was reproduced byte-for-byte into an isolated temporary directory using the actual generator functions. The diff is confined to source pins, task identity/count/table entries, fixed source selection and the extra lineage check. `FR13_FIXED32_MIDDLEWARE_FLAGS` remains empty at both assignments, lines 5416 and 5579. This delta adds no proxy-authenticated ingress or raw-capture behavior. Private boot-pin creation, main-caller wiring and request-observer behavior are outside this review; their presence or success is not inferred from this source PASS.

## Executed controls and limits

`review_controls.py` completed **64 bounded controls**. They include valid parent/derived credentials; exact saved verification; exact insertion and unchanged prior tables; temporary missing/symlink/wrong-task subset and altered patcher/credential refusals; actual validator rejection of stale file/self digests, resealed wrong source/binary, a nonfinite measurement and missing determinism; selected/prior container identities; wrong/extra/unknown/conflicting identity refusals; actual host missing/wrong subset refusal; exact/wrong/extra/duplicate/empty count cases; all 11 new preamble bindings; isolated exact generation; and actual command argument propagation. No broader serving/runtime behavior is claimed.

Local macOS Bash 3.2 cannot parse the pre-existing `[[ -v ... ]]` construct in either predecessor or successor. This is an unchanged local parser limitation, not a new defect; the new extracted shell paths were tested locally. The separately parent-performed target `bash -n` receipt, copied as `PARENT-SYNTAX-RECEIPT.json` (SHA `f22ae473fccfd1ffa328226e19ff8d92ab051d0bcad9a4663edfa6eeb7002805`), binds the exact reviewed launcher `eb8143836beddc5446b132381ad39c5c4625c89b873aa4b4d7b5566f0b5a6d5d`, return code 0 and empty stderr. It records syntax-only input, not launcher execution.

Reviewer control development initially encountered an extraction mistake (the container workload function is embedded in a source template), the expected Bash-3.2 limitation, and a whitespace-sensitive middleware regex; only the reviewer harness was corrected. No implementation file changed. The final reproducible control file and complete results are sealed below.

- `SNAPSHOT.json`: `0e42ada1d5fc604b50b4fd261ef5ad87a04659b030976dd8a6d038f3ad7caa39`
- `review_controls.py`: `a82a2c4ea801160bcff5f0d5999128464b658c7758fecd11c58b197639041e77`
- `CONTROLS.json`: `d0e64c6b5a84aea4d9a92f3b1d3e60c2a6388f5ce8b74ffe0f4826dbbe85669f`

Any eventual parent approval must bind the reviewed source bytes and separately reviewed runtime wiring. This note authorizes no operational action and advances no experiment or qualification counter.
