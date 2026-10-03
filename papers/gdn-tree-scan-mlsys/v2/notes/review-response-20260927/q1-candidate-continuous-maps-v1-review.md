# Candidate continuous maps: bounded independent review

**Required fix remains: different logical GDN layers can select the same persistent state bytes after a permitted column transition.** This is source/CPU review only; no runtime, GPU, gate or qualification action.

Initial source SHA `32c9ffb7ea42ce4c1011101b7c1df8d3b1151c01ea1cf2d9841770a77301a5f6` is preserved. The reviewed bounds repair is SHA `10045da349ecb279d542d832c10c6d4e13a4c6400a09c6d7acb2105e412dab01`.

## F1 — bounds: repaired

The initial mapper and extension admitted actual running row 99 or spare row 99 in conv/SSM banks with six rows. The CPU/GPU table, request block IDs and actual SSI were coherently changed at 992→993; the old importer is absent from this read-only path. The repair checks current model bank geometry/dtype and both selected/spare rows against both banks before returning. Both retained false admissions now refuse. This closure does not modify the accepted initial mapper or import engine.

## F2 — selected persistent state overlap: remains open

Using the same real CPU fixture and production bank identity methods, group 0 column 1 can coherently change to in-range row 4 during 992→993. Group 2 column 1 already selects row 4. For the layers of each three-layer shared bank, both logical layers therefore point to the exact same running conv/SSM bytes. Their CPU/GPU tables, request block IDs and SSI agree, and the maps/extension return `storage_owner_preserved=True` and `materialized_maps_preserved=True`. Neither the initial nor bounds-repaired mapper checks disjoint selected persistent state spans across logical GDN layers (current row admission around lines 53–69; `extension` preserves each prior consumed column, but a newly selected column can introduce this collision).

Minimal closure: derive the currently selected running conv and SSM occupied byte spans from the actual live tensors and row strides; reject overlap for distinct logical states/layers. Keep legitimate sharing of a backing bank at disjoint rows. This check concerns physical map validity, not numerical state equivalence; deferring state-byte comparison cannot make two distinct persistent states sharing the same live destination valid. Preserve all current ownership, selected/spare bounds and prefix retention checks. A focused positive must retain the existing three layers sharing a bank on different rows; the retained row4/group0/group2 case must refuse.

## Powered CPU checks

The actual mapper ran against retained full-geometry candidate bank tensors and AST-extracted unchanged production bank identity methods. No importer was called. The repair pass contains 13 checks: real 992→993 changes SSI column 0→1; real 993→1024 retains column 1; both out-of-range running/spare rows refuse; changed already-selected GDN column and materialized attention page refuse; request, drafter and bank tuple replacements refuse; bad metadata census and CPU/GPU table mismatch refuse; all cache tensor version counters remain unchanged and the import latch stays zero. The thirteenth finding is the still-accepted selected-state collision described above. These tests preserve actual initial failures and do not claim raw-state, model or qualification evidence.

Evidence: `p0/monitor/review-response-20260927/candidate-continuous-maps-independent/` contains initial/bounds source snapshots, `controls.py`, `controls_collision.py`, `controls_bounds_repair.py`, JSON results and logs. The first collision-script attempt intentionally refused the changed source SHA before any fixture work; the rerun explicitly used the preserved initial snapshot. Re-review should be limited to F2 and unchanged positive transitions after the implementer repairs it.
