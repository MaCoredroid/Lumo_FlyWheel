# Fixed32 full-vocabulary exact4 B1 stock baseline

Status: `VALIDATED_RECOVERED_STOCK_ARM`

This package records the stock arm from a real SWE-Verified exact4 B1
full-vocabulary timing campaign. All four tasks completed with clean serving
and fixed32 boundaries:

- 6,095 pure-decode events
- 126 authenticated logical model requests
- zero aborted logical requests
- zero failed serving attempts
- two resolved and two failed task evaluations

The original outer launcher exited with `serve_rc=16` after all task, ingress,
terminal-flush, and boundary evidence was complete. The teardown-only failure
was:

```text
NameError: name 'inspect' is not defined
```

The audit import had been placed in the wrong embedded Python block. Recovery
commit `f1878c3cd` corrected that placement. The canonical traffic audit was
then rebuilt from the immutable exact4 task boundaries, ingress ledgers, work
census, traces, and evaluation records. It reconciled exactly at 6,095 events.

## Measurement

The timing contract's retained full-wall result is:

- Step wall: `294.69449927663806 ms`
- Full-wall TPS: `19.94807850778614`
- Accepted drafts/event: `4.878589007383101`
- Mandatory-weight lower bound: `153.938384645 ms`
- Ratio to lower bound: `1.9143665821636244x`
- Acceptance cap: `177.0291423413919 ms`
- Distance above cap: `117.66535693524616 ms`

The measured component split is:

- SFWD: `159.32640444698416 ms/step`
- DFWD: `98.79363861084005 ms/step`
- CFWD: `20.69117135821651 ms/step`
- Remaining wall: `15.88328486059736 ms/step`

The reducer also reports aggregate `s_per_fwd=288.5993202076537 ms`; that is
not the retained full-wall `step_wall_ms` used by the timing contract.

## Scope

This is a recovered stock baseline, not a clean stock/candidate pair and not a
formal floor-acceptance result. The candidate arm did not start because the
stock wrapper failed during teardown. A separately source-bound candidate-only
continuation is required before drawing a kernel delta.

Measured source commit: `722e5bdb4ebb517d1f74cb44b5e2cbab78e363fe`.
