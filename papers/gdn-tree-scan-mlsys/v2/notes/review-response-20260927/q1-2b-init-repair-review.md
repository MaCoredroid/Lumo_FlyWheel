# Q1.2b initialization repair review — 2026-09-28

**Verdict: one concrete initialization launch blocker; lifecycle source repair otherwise acceptable within this bounded static review.** The separate INIT gate is incompatible with the unconditional numerical-policy authorization path. Correct that branch without changing the evaluator or disguising INIT approval as calibration approval. Final tests, frozen identities and actual initialization remain separate parent gates.

## Exact reviewed snapshot

`p0/monitor/review-response-20260927/init-repair-review-snapshot-20260928T004249Z/` — `SNAPSHOT.json` SHA-256 `2189eed457d41e4c7c51a92b3cccda26aca88cc8d11f9b80cac0279a2a00a218`. Independently rehashed **8/8 members**, all match.

| File | SHA-256 |
|---|---|
| `tools/q1_component_runner_v2_1.py` | `8d8cf68f4d93de7e6948586b0f5f8036814d98c9457b8e702255f3b37595c670` |
| `tools/run_q1_2b_init_smoke_v2_1.sh` | `b012380f54db6ba6363e2ab31a48c7ba22090a879ef29f37f1b56fe1cb37f5a2` |
| `tools/run_q1_2b_component_v2_1.sh` | `84a377973c3cf68504b964fae7082e8db4d4daba6335bb9878f02bc3cb2456e2` |
| `tools/q1_2b_reduce_v2_1.py` | `1d58c83d4cf96d755d12822ed23f4557e5d037f6f9aeadbb8a135dfe5d59449c` |
| `tools/q1_policy_evaluator_v2_1.py` | `2d25e4c9bb669a4c66a5a3caa5093b614e5aba136eb424efb0e410f0b0b61061` |
| `tools/tests/test_q1_2b_v2_1_setup.py` | `2e7ad5258b4a9c62dac6f520ad2bde2d2f375a9bece97fb7ac9d792d132c8284` |
| `tools/tests/test_q1_2b_v2_1_pipeline.py` | `1657a144a87296eda9c16544a394c53a077d249353d3730d1f40240609139b93` |

Line numbers below refer to this snapshot. The unchanged gate-bound production kernel was consulted at SHA `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`.

## Blocker I1 — a correct INIT approval always fails numerical-policy authorization

The init launcher requires `gate == "Q1.2b-INIT"`, `approved is True`, exact approved run ID and an init-only single-container scope, and snapshots that gate. Runner `gate_approval` **469–474** passes the gate name through as `approval.stage`. Main **733–736** then unconditionally calls `load_policy_verified_v2_1` and treats `_approved_ok == False` as a refusal. The unchanged evaluator has `STAGE = "Q1.2b"` and rejects other approval stages in `load_policy_verified_v2_1` (**31,84–85** in the previous layout; locate the same function in this snapshot). The intended INIT gate therefore reaches the refusal exit at runner **763–766**, before `Backend` construction.

This is a deterministic cross-file mismatch, not missing GPU qualification or a failure of the repaired page layout. Parent independently reports a matching negative/positive reproduction in `init-gate-policy-stage-reproduction-20260928T004400Z.json`; this reviewer did not rerun that parent receipt.

**Minimal correction:** branch explicitly on `args.init_only`. Validate the dedicated INIT gate, approved run identity, immutable source bindings and init-only scope as initialization authorization; load the unchanged numerical policy for integrity/domain checks only. Leave its calibration authorization false/unused in the init path. Full calibration must still require the ordinary `Q1.2b` approval path. Do not rewrite the INIT stage to `Q1.2b`, loosen the evaluator's accepted stage, or let `--init-only` waive source checks. Add a CPU control exercising a correct INIT gate through the actual pre-Backend authorization branch, with wrong-stage/unapproved controls and a guard proving `run_process` cannot be reached.

## Prior lifecycle findings: source closure

- **Shared typed pages:** `build_alias_pages` **164–183** creates sixteen BF16 raw allocations, one channel-contiguous conv view and one FP32 `(rows,48,128,128)` SSM view per allocation. The 48-entry tuples reuse the identical view objects at layers `a,a+16,a+32`. Per-row byte arithmetic independently checks: page 4,194,304 bytes; conv `[0,696320)`; SSM `[696320,3842048)`; remaining gap 352,256 bytes. Conv and SSM regions are nonoverlapping and aligned; dtype and row stride are preserved.
- **Rows and controls:** rank-specific run rows **4,5,6**, scratch rows **7,8,9**, controls **10–15** are disjoint across each alias class, avoid null/warm rows and preseed safe root 22. `restore_O0` **367–373**, `published_state` **435–436** and `control_rows_intact` **438–440** consistently use the new row functions. Clearing shared row 0 is idempotent; the unique per-rank control rows avoid conflicting per-layer sentinel writes. I executed the exact row functions/validator via AST using standard-library Python: **16/16 classes passed**, and insufficient `rows=8` was refused.
- **Production metadata:** `Backend` **334–353** uses contiguous int32 `(48,1,32)` commit SSI, `(1,16)` accepted paths and `(1,)` lengths; three retained SSI-group tensors; 48 distinct BF16 `(36,10240)` source stagings; and the public topology-derived int64 source map. The new `fr13_tree_conv_fused.py` dependency is included in both launchers' expected/source maps and the reducer helper map.
- **Public lifecycle:** `boot_lifecycle` **130–161** calls production subtree/group registration, recurrent graph preseed, conv preseed, SSI selfcheck, lease audit and warm on the same retained tuple/tensor objects. The repeated identical group registrations in `Backend` and `boot_lifecycle` are allowed before pregather freezes the registry; this is redundant but not a blocker. No private conv-state dictionary is manufactured and the production guard is unchanged.
- **Warm accounting:** the wrapper requires production `fixed32_committer_warmup_counters().ready`, classification `unmeasured_boot`, all state/input/lease restoration flags, and zero recurrent measured counters. Production warm itself requires initially zero conv/recurrent counters and restores both (kernel **15720–15795,15888–16057**). Its `ready` computation additionally checks the expected warm replay/direct-conv counts and alias contract (kernel **13535 onward**). No extra wrapper copy of those numerical criteria is needed for this narrow init check.
- **Scientific boundary:** the old `no_conv=true` statement was replaced by a clear conv-setup-executed/no-conv-qualification boundary (**699–704**). This correctly distinguishes the public warm's activity from the GDN fixture campaign.

## Init-only boundary and launcher inspection

Runner **787–795** writes a receipt with `fixtures_done=0`, empty fixtures, `characterization_complete=False`, `integrity_ok=None`, and exits. It precedes `run_process` (**796**) and therefore precedes `TensorStore`, `Inventory`, fixture `torch.load`, reference computation and candidate fixture scans. Reading/hash-binding the JSON fixture manifests earlier does not run their cases. Warm and preseed remain actual GPU initialization work and must not be reported as “no GPU work” or numerical qualification.

The init launcher binds every listed runner/helper/kernel/native/conv source, image, manifest, policy/base/contract and its own launcher hash before Docker; refuses missing/unapproved/wrong-stage/wrong-run/mismatched-source inputs; requires a fresh output directory; records actual command/image/CID, exit and terminal receipt; and uses exactly one `--init-only` container with no reducer invocation. `bash -n` passed. I did not execute the shell or claim an execution-level launcher pass. The source-level authorization mismatch above is the only concrete blocker found in this init path.

The final frozen receipt must bind the exact adopted initialization design and complete dependency set; that final freeze is intentionally pending. This snapshot's CPU setup tests exercise alias/ownership/row faults and public-call ordering, but their kernel is a stub and does not establish actual device warm success.

## Later calibration: separate disposition

The same repair changes physical row strides and storage aliasing while preserving fixture input tensors, FP32 recurrent arithmetic, distinct per-instance state and reference policy. The checked restore/readback/control-row code is consistent; no additional concrete later-calibration defect was found in this bounded pass. This is **not** acceptance of a full calibration launch or its numerical results. Later full-source tests/freeze, actual initialization receipt, exact reducer linkage and a new parent calibration decision remain required. Preserve the original failed run and all prior numerical/negative controls.

No remote command, GPU/container/model execution, Torch suite, source change or gate change was performed by this reviewer. Executed checks were hash verification, source/diff inspection, standard-library AST row arithmetic, and shell syntax validation only.
