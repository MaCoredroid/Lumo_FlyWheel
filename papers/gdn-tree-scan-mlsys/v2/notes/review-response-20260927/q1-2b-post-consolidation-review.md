# Q1.2b post-consolidation static review — 2026-09-28

**Verdict: one remaining launch blocker.** The five original repair groups are substantially addressed in this immutable snapshot, but N1/N3/N4 negative-power enforcement is unreachable. This review is source closure and bounded standard-library execution, not a full-pipeline pass, runtime-parity result, final freeze, or permission to launch.

## Reviewed bytes

All paths below are relative to the immutable parent snapshot:
`p0/monitor/review-response-20260927/repair-source-snapshot-20260928T000400Z/`.

| File | SHA-256 |
|---|---|
| `SNAPSHOT.json` | `4bd9555636b0691c8af861a410bea20cc7bc932411c99a4c53b418f1419a722c` |
| `tools/q1_2b_reduce_v2.py` | `a22a6b6b71bb95000766a39651f3ce3cffbb14b176469e050508a67e29060090` |
| `tools/run_q1_2b_component_v2.sh` | `f8ba293608c72411c8786f01ab9645f3a35f27d678fdf7f2597e9f0ea1cb4514` |
| `tools/q1_component_runner_v2.py` | `cbb754e8d306758f5d11ab65fca4d533614839980d32ff16f82d76fc7a885544` |
| `tools/q1_policy_evaluator_v2_1.py` | `2bdf5b6bda6e470820d7e779fe0217915f67fdd048fa8fef221ff186810190a5` |
| `tools/q1_2b_c2_preflight.py` | `75a6802812eb9a577aed80a0c59a9f0a263e5c6cf05bbe7795b34bc58e2b1c76` |

Independently rehashed **60/60 snapshot members: all match**. Compared the repaired files to the preserved `reviewed-v2-snapshot-20260927T235500Z`. Parsed policies v2 and v2.1: `constants` and `rules` are exactly equal. No remote reads/writes, GPU, model, container, or serving execution occurred in this review.

## Remaining blocker: negative detection no longer controls the verdict

At reducer **lines 566–570**, the loop requiring N1/N3/N4 to yield `FAIL` is indented under `if F["malformed"]`, after its unconditional return:

```python
if F["malformed"]:
    summary["rows"] = rows; return summary, 2
    for tag in ("N1_sibling_substitution", "N3_ring_swap", "N4_stale_metadata"):
        if not neg_power[t].get(tag) or any(v != "FAIL" for v in neg_power[t][tag]):
            F["structural"].append(...)
```

The loop cannot execute with either an empty or nonempty malformed list. Its current position would also leave `t` as B if merely dedented once; enforcement must explicitly cover **both A and B**. Raw negative tensors and metric recomputation are now present, but honest recomputation yielding `PASS`/`UNCOVERED`/refusal for a required corruption witness does not itself block the run. Ordinary candidate aggregates can therefore pass despite absent negative power.

**Minimal repair:** after collecting all negative results and handling malformed input, iterate both process tags and the three required negative tags; append a structural failure unless every required result is `FAIL`. Preserve N5 as a causal clean-C0 versus poisoned-C0 equality check, with its paired verdict diagnostic only. Add a full-pipeline control for each no-op N1/N3/N4 with correctly recomputed arrays/tensors; also exercise an A-only failure so an accidental B-only check cannot pass. No policy constants or model experiment need change.

I independently executed an AST extraction of the **exact final reducer block, lines 566–582**, with ordinary A/B aggregates `PASS`, empty preexisting findings, and N1/N3/N4 all `PASS` in both processes. It returned **rc=0 with no structural finding**. This is a direct control-flow reproduction, not a claim that a complete Torch fixture campaign was executed.

## Closure of the original five groups

| Original group | Status in this snapshot | Evidence / remaining gate |
|---|---|---|
| V2-1: PID namespace falsely rejects fresh containers | Source and isolated controls closed | Reducer 93–114 uses distinct container hostname/name, process tags and UTC; PID is informational. Launcher 31 provides separate `--cidfile` and names. Three local controls: two distinct containers each PID 1 accepted; duplicate container identity and wrong process tag refused. |
| V2-2: helper map/gate/source identities | Static closure | Launcher 53–64, 81–83 emits top-level `helpers` and complete `expect`; reducer 191–220 reads that same location, maps helpers to expected hashes, compares every source key and scope with the unchanged gate, and binds base-policy/contract snapshots. Reducer 297–319 checks actual helper/runtime identities and A/B agreement. No remaining cross-file key mismatch found. |
| V2-3: negative tensors/arrays not independently verified | Retention/recompute repaired; negative-power verdict still blocked above | Reducer 134–170 requires tensor bytes/hash/dtype/shape, 377 applies it to ordinary and negative records in both processes, 556–564 regenerates C2 and recomputes negative metric arrays before evaluation. The unreachable power check prevents full closure. |
| V2-4: exact negative product, chronology, replay types and causal N5 | Static closure, subject to power check above | Reducer 358–377 checks exact negative IDs and binds internal/inventory case IDs; 421–430 checks reference-before-candidate phase/inventory ordering and case census; 441 rejects boolean replay counts; 497–519 binds negative identity/reference/path and poisoned-versus-clean C0 equality. Runner 352–377 seals native repeats and reference eligibility before first scan (383–384); N5 477–487 uses same unpoisoned C0. |
| V2-5: host Torch recomputation differs from image | Launcher/source closure; actual CPU preflight remains pending | Launcher 150–159 now runs reduction in the same immutable image, no `--gpus`, empty CUDA visibility, OMP/MKL=1, repository read-only and `/runs/$RUN_ID` preserving the approved basename. Reducer 221–231 checks actual Torch/threads/CPU-only state. New preflight 38–59 regenerates stored and expected C2 hashes through the same fixture helper. This report has not executed or accepted that preflight. |

## Cross-file compatibility and limits

The repaired launcher passes existing runner/reducer arguments, including `--recompute all`; it does not substitute a different block or numerical policy. The reducer mount preserves the actual run ID required by line 236. Helper keys emitted by the launcher match the runner attestation/reducer map. Reference-only eligibility calls use the evaluator's supported `bound_cases` and `fixture_sha256` arguments. The evaluator's new reference-only recheck strips candidate fields and compares the same reference projection; the normal and negative paths retain the correct ordinary native/C2 references. I found no additional launch-path false refusal within these checked interfaces.

Final closure still requires: the negative-power repair; fresh hashes/freeze binding for changed source; the complete CPU pipeline controls against those exact bytes; and the pinned-image C2 runtime-parity receipt with expected nonempty calibration population and runtime identity. The parent retains all authorization decisions. No broader experiment or policy change is requested.

## Reproducible local control

Executed with the bundled Python using `-B`, with working directory set to the immutable snapshot. The relevant exact-tail reproduction is:

```python
import ast
from pathlib import Path
p = Path("tools/q1_2b_reduce_v2.py")
tree = ast.parse(p.read_text())
r = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reduce_run")
tail = [n for n in r.body if n.lineno >= 566]
fn = ast.FunctionDef(name="tail", args=ast.arguments(
    posonlyargs=[], args=[ast.arg(arg="neg_power")], kwonlyargs=[],
    kw_defaults=[], defaults=[]), body=tail, decorator_list=[])
ast.fix_missing_locations(fn)
ns = {"F": {k: [] for k in ("malformed", "structural", "numerical", "uncovered")},
      "summary": {}, "rows": [], "cases": {"x": {}},
      "case_results": {"A": [{}], "B": [{}]}, "verdicts": {},
      "agg": {t: {"aggregate": "PASS"} for t in ("A", "B")},
      "recomputed": 1, "a": {"attestation": {}}, "b": {"attestation": {}},
      "seen_tensor": {}, "t": "B"}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), "exec"), ns)
neg = {t: {n: ["PASS"] for n in ("N1_sibling_substitution", "N3_ring_swap", "N4_stale_metadata")}
       for t in ("A", "B")}
summary, rc = ns["tail"](neg)
assert rc == 0 and ns["F"]["structural"] == []  # reproduces this snapshot's defect
```
