# Candidate wrapper v2: early L2–L4 closure review

**HOLD for final integration. L3 is closed; L2 and L4 are partially closed, with the narrow residuals below.** This review does not cover the still-unbuilt generated diagnostic launcher v2, its L1 operational exclusions or Tier-B identity decision. Those remain separately pending, not newly discovered defects. No launch or operational authorization follows.

Snapshot: `p0/monitor/review-response-20260927/candidate-v2-early-reviewed-20260928T0500Z`, relative to paper v2. The snapshot initially listed seven files and was supplemented with the two environment artifacts during this read; the reviewed wrapper/test bytes stayed unchanged. Independently verified the resulting **9/9** listed sizes/hashes, zero mismatches; no environment values were printed.

| Item | SHA-256 |
|---|---|
| `PARENT-SNAPSHOT.json` after the two-artifact supplement | `cdb53c027433645efa7315a9d4bc53cf1bc5bccc209fc19ea87e8dd015d2c40c` |
| `tools/run_q1_candidate_stage1_v2.sh` | `dd07245e31082c244747be9336ef6c0792cbc3f9d405de49feecc46be8c33f04` |
| `tools/tests/test_q1_candidate_wrapper_v2.py` | `fe2588b266c5f69e3b3d00480f730e96530ade8e0363f69749af453814bce260` |
| Author test log, attempt 1 | `1260e7a9b2b2527df1354399969df1aedcced66811c6288af068fd9a8c97b6d8` |

## Confirmed repairs

- **L2 stopped-name refusal:** lines **121–122** inspect all container names, including stopped containers, and refuse before launcher execution or cleanup arming. Lines **188–196** require a full creation CID and exact CID/name/image inspection before sourcing the accepted cleanup library. A pre-creation failure seals without cleanup. The generated launcher's actual creation/CID publication and final caller integration must still be checked when available; this early wrapper is not proof of that missing caller.
- **L2 atomicity at one selected path:** line **81** uses exclusive `mkdir`, so two entrants at the same `$OUT` cannot both claim it. It no longer uses check-then-`mkdir -p`.
- **L3:** the required-flags helper, OOM guard and old reference hook v2 are all now in `REVIEWED` (**34–40**), checked against the gate (**64–71**) before launch. Staging explicitly checks copy/chmod and the reviewed hash (**159–165**). The old omitted-dependency defect is closed, contingent on the final gate carrying these exact keys/bytes.
- **L4:** ordinary contention/readiness/job/staging failures after line **119** now have the no-owned-container EXIT sealer. `FINALIZED` is initialized locally (**84**). The host launcher is supervised by `timeout` (**186**) and the wrapper then resolves ownership before routing failure to prelaunch sealing or the existing accepted finalizer (**188–200**). Its default startup timeout is 1800 seconds, distinct from the later health timeout. The unchanged cleanup library was not reopened.

## Remaining corrections

### W1 / L4: readiness consumes a synthetic schema and obsolete lock namespace

Wrapper **135–145** does not match the actual reviewed recovery v2.1 producer (`memory-v21-reviewed-20260928T0450Z/tools/memory_recovery_v2.py`, SHA `17d9c4453bbea247f0e3c60cb122859aee9e529806a5196c893066e04dac89e2`):

| Wrapper expects | Actual producer |
|---|---|
| `.lock-v2` | `<authority_dir>/.campaign-operation.lock`, producer **749, 775–786** |
| `ATTEMPT-v2-*.json` | authorization-wide consumed reservation, producer **886–900**; finalization hold repair remains pending |
| only `holds/HOLD-*.json` | **any file** in the authority's `holds/` blocks, producer **799–809** |
| `boot_hold.reservation_durable` | `boot_hold.conditions.reservation_durable`, producer **1348–1368** |
| `authorization.scientific_gate_sha256` | `authorization.fields.scientific_gate_sha256`, producer **1420–1423** |

The author positive test fabricates the wrapper's expected shape at test **64–67**, rather than serializing the actual producer receipt. Its passing result therefore does not close this integration.

Executed the wrapper's **unchanged readiness Python heredoc** in memory, with `open` and glob results stubbed and no real output/operations. A positive no-reclaim control was included. Results:

| Modeled input | Observed disposition |
|---|---|
| Empty authority, explicit parent no-reclaim disposition | accepted as skipped |
| Actual `.campaign-operation.lock`, same disposition | **accepted as skipped** |
| `holds/pending-finalization.json`, same disposition | **accepted as skipped** |
| Actual v2.1-shaped SUCCESS receipt, nested conditions/authorization fields, matching gate | **refused**, `reservation_durable=null`, `gate_bound=false` |
| Author test's synthetic flattened receipt | accepted as performed |

**Minimal fix:** consume the final settled memory-recovery contract, using its real field layout and authority-wide lock/pending-finalization/hold rules. Bind the authority/recovery namespace to the parent-issued contract instead of relying on freely selected `CAND_RECOVERY_ROOT` (**24**) and a search under that directory. An empty alternate root must not bypass an unresolved authority. Preserve the distinct explicit parent no-reclaim branch, but it must still refuse active locks or holds. Use an actual producer-generated receipt projection in focused CPU tests. The separately reported memory F5 repair remains open; do not guess its final schema or treat the current receipt as qualified.

### W2 / L2: the single-use claim is still limited to a freely selected output root

`OUT_ROOT=${CAND_OUT_ROOT:-...}` at **24** is absent from the gate's checked scope at **61–73**, and the sole attempt claim is `$OUT_ROOT/$RUN_ID` at **81**. Thus the same approved run/gate can claim a second output root; the later absence check can pass after the first engine is removed. The author second-entrant test (**88–93**) only exercises the same root.

**Minimal fix:** either bind/check one canonical output/attempt path in the gate, or place the immutable run-ID claim in a canonical parent authority independent of output location. Retain exclusive `mkdir` or equivalent. Add one alternate-output-root refusal control; no broader scheduler is needed.

### W3 / L4: the first two post-claim evidence writes precede the trap and their errors are ignored

After successful namespace claim (**81**), the claim JSON redirect (**82**) and gate snapshot copy (**83**) occur before the failure sealer is armed (**119**). With `set -uo pipefail` and no explicit checks, either write can fail and execution still proceeds. This leaves the exact earliest portion of original L4 unclosed.

**Minimal fix:** initialize/define the no-owned-container sealer first, install its trap immediately after the atomic claim, and explicitly fail on claim-record or gate-snapshot persistence errors before inventory or launch. Preserve primary failure if receipt persistence also fails. The later staging/job/binding checks already have the right bounded structure.

For the existing bounded-startup repair, also validate the declared supervisor interval as a positive finite bounded value before invocation: line **22** accepts arbitrary `CAND_LAUNCH_TIMEOUT_S`, including GNU timeout's disabling value `0`, without gate validation. This is a small closure of the requested bounded-startup contract, not another operational axis.

## Verification boundary

Local `bash -n tools/run_q1_candidate_stage1_v2.sh` passed. The independent readiness controls used only extracted Python source, standard-library in-memory stubs and source/hash reads. The supplied 61-test Linux log, including focused wrapper controls, was inspected but not rerun here. No SSH, Docker/container operation, GPU/model/API call, process manipulation, reclaim or source/gate edit occurred. This note is the only file written.

A final disposition still requires the repaired readiness/claim/earliest-sealing path, exact final freeze/gate equality, and the completed diagnostic launcher/caller with its separately pending L1 exclusions. No expanded cleanup-library review or new scientific experiment is requested.
