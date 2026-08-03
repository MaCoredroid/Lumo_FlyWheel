# CFWD Hydra27 TAW prerequisite audit

This is an offline readiness audit for the fixed32 CFWD logit-direct B1 gate
and exact4 timing runner. No Docker container or GPU workload was launched.

No Hydra27 source-v7 TAW production bundle or its credential chain was found
in the available workspace artifacts. The only located Hydra27 TAW artifact is
the archived B1-only `fr13-fixed32-taw-exact-commit-v3` diagnostic. It is not a
source-v7 production bundle and explicitly lacks B4 coverage.

The required production path is:

1. Run `scripts/fr13_run_b1_k64_taw_source_v7_gate.sh` with
   `MODE=hydra27_fixed32` to issue the fresh B1 live bundle and credential.
2. Run `scripts/fr13_run_b4_tail23_all_parent_live_gate.sh` with
   `FR13_FIXED32_ALL_PARENT_MODE=hydra27_fixed32` on the same source file to
   issue the reviewed exact4 B4 production bundle and verdict.
3. Run `scripts/fr13_taw_b1_credential.py merge` to replace the B4 bundle's B1
   record with the fresh source-bound B1 record and create the merge binding.
4. Supply that complete chain to the CFWD live gate. Only after the CFWD gate
   issues its credential can the exact4 CFWD timing runner execute.

The CFWD runners now call `fr13_taw_b1_credential.py validate-production`
before their Docker preflight. They consume the explicit merged credential
chain; they do not silently compose or modify qualification artifacts.
