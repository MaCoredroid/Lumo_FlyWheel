# #51508 / #48475 / #50021 zero-accept-row kernel probes (GB10, 2026-09-17)

`variant_*.py` are the fused recurrent gated-delta-rule kernel modules extracted
from main `80447d2765`, #48475 `f70b0ffe66`, #50021 `9a198c0f84` and #51508
`54b69f5dd5` with only the Triton import rewritten (Codex-verified against the
pinned sources). `probe_zero_accept.py` plants a live block id immediately
before the index-tensor view (inside the backing allocation) and reports which
blocks each variant writes for a zero-accept (Z), count-1 control (C) and
null-row (N) case. `probe_wrongstate.py` compares block 5 after count 0 vs
count 1 and reports untouched / same-as-count-1 / different.

`logs/probe_run_*.log` is the full run (both probes; verdict string of the
wrong-state probe was two-way in that run — "DIFFERENT" for pr50021 there
means untouched-vs-advanced, see the v2 log). `logs/probe_wrongstate_v2_*.log`
is the re-run with the three-way verdict. No other process was on the GPU
(`procs on GPU` is empty in both logs). Paths in the probes are hardcoded to
the machine they ran on.
