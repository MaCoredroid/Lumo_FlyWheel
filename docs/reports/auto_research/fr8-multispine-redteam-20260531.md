# FR8 Multi-Spine Mechanism Red-Team

Date: 2026-05-31

## Result

`LUMO_MULTI_SPINE=1` was found to route through the retired kernel-row/internal-row machinery:

- `_prelaunch_for()` treated multi-spine as `kernel_rows_requested`.
- The multi-spine environment forced `LUMO_FB_INTERNAL_ROWS=1` and `LUMO_FB_KERNEL_ROWS=1`.
- The runner materialized sibling rows with `input_batch.add_request(...)`.
- The sample path pruned those rows with `remove_request(...)`, `condense()`, and parent promotion/collapse logic.

This is the forbidden condense/sibling-collapse route, not the requested persistent static-shape row construction.

## Gate Applied

The launcher now fails closed when `LUMO_MULTI_SPINE=1` is set. This prevents B=4 validation runs from producing misleading numbers through the retired mechanism.

## Validation Status

No B=4 run was performed after this red-team result. `path0 == 3.150 / 13.3% acc=0` is therefore not validated for FR8 multi-spine, and no winner number should be trusted from the invalid route.

## Required Rebuild

The next valid implementation must keep alternate spines as persistent rows in a static batch shape, copy recurrent GDN state with the `LUMO_MULTI_SPINE_COPY_RECURRENT_STATE` device-side `index_copy_` path, select the winner at accept time, and avoid per-step `remove_request`, `condense`, and `req_to_blocks.pop` collapse.
