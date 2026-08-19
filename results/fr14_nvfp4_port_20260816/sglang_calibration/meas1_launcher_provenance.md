# MEASUREMENT 1 — LAUNCHER PROVENANCE (campaign runner, 2026-08-19)

"A staged copy is a fork with a birthday."

The run dir held a launcher copy staged at 18:18Z, BEFORE F1/F2 landed at 18:52Z.
Booting it would have silently reproduced the pass-106 exit 2 and measured nothing.

    PRE-F1 staged copy   sha256 67fa87172c47ea41e95b8e84701769fb8b0840250798c91429e668c0be9a22c5
                         size   441032
                         F1 mint block: ABSENT     F2 arbitration: ABSENT
                         retained at $OUT/launch_nomiddleware.PRE_F1.stale.sh

    POST-F1 tracked      scripts/fr14_leg3_launch_nomiddleware.sh
                         sha256 2e4cb94f81a8e38a53581805911044c2629c45d2a1fe87d4d621cf5551a16e2a
                         size   430819
                         F1 mint block: PRESENT    F2 arbitration: PRESENT

Copying the tracked launcher INTO the run dir breaks its sibling resolution --
it does `source "$SCRIPT_DIR/fr13_required_tree_flags.sh"` and only scripts/
holds that file. So $OUT/launch_nomiddleware.sh is now a SHIM that execs the
tracked launcher IN PLACE, which is how ablation_a_leg3_boot.sh has always
invoked it. The vehicle is therefore the tracked file at boot HEAD, never a copy.

Also required in env: TAG (any value). The staged boot sources
fr13_fixed32_floor_timers_seq.sh under `set -u`; run_variant is stubbed to a
no-op first, so TAG's value is irrelevant -- it only has to exist. Passed as
TAG=oursrandom rather than editing the staged script.
