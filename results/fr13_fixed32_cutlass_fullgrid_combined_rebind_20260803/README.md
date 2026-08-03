# Fixed32 B1 fullgrid combined-binary rebind

The first exact4 production attempt on the older fullgrid-only source completed
the stock arm, then failed before candidate task execution. The production
installer resolved `scripts/fr13_dvk_subset_blocks.json` relative to the image
working directory instead of the repository mounted at `/workspace`.

No candidate task or candidate timing sample was produced. The completed stock
arm is therefore not a paired result and is not acceptance evidence.

Commit `9b0be32d2a77e7f132002a53b7918b2e2112ecbd` fixes production verification by
resolving the K64 block map beside the already source-bound patch file. It also
rebinds the fullgrid selector to the audited combined CUTLASS binary, which
contains both the fullgrid B1 schedulers and the later MTP M1/M4 scheduler.

The combined binary is not committed because it exceeds the repository file
limit. Its exact identity is recorded in `manifest.json`. Production remains
default-off and requires a new real SWE-Verified B1 byte gate on the final
frozen source, followed by exact4 timing.
