# Fixed32 SFWD topology-specialized prior masks

Status: **OFFLINE_SOURCE_READY_CODEGEN_REQUIRED**.

Source commit `1a1a2459870b5eb9c4358c415a4cde4a6c7d67db` specializes
the packed x-gather kernel's prior-state selection to the exact fixed 32-row
tree. For convolution taps 0, 1, and 2, only rows `0-8`, `0-3`, and row `0`
respectively can read the three historical state rows. The kernel uses those
row masks directly instead of repeating generic `source_row < 3` and
three-way historical-row selection on every tap.

The source tests derive all tap sources from the fixed parent vector and prove
the three masks for every physical row. The exact historical values are also
checked: tap 0 selects `(0,1,1,1,2,2,2,2,2)`, tap 1 selects `(1,2,2,2)`, and
tap 2 selects `2` for row 0. Tap and accumulation order are unchanged.

This is source evidence only. No Triton codegen, GPU kernel, Docker service,
real task, timing, acceptance, or hardware-floor measurement was run. The
candidate requires isolated B1/B4 codegen and hard resource comparison before
any real byte gate.

This reduced package excludes compiler output, binaries, IR, raw logs,
task/model/request/response/patch content, task identifiers, environment
values, credentials, process identifiers, container identifiers, and secrets.
