
# Request Path Evidence Brief

You are tracing how `--owner`, `owner_source`, and `routing_key` move through this repo.

Rules:
- Do not edit repo behavior.
- The only allowed writes are:
  - `artifacts/request_path_brief.md`
  - `artifacts/path_map.json`
  - `artifacts/docs_correction.md`
- Use repo-local evidence only.
- In markdown outputs, cite concrete evidence with backticked `path::symbol` tokens, for example `sync_app/service.py::_resolve_owner`.
- `artifacts/path_map.json` must use schema `cnb55.request_path_map.v1` and include:
  - `variant_id`
  - `live_path[]` with `file`, `symbol`, `role`, `caller_symbol`, `callee_symbol`
  - `field_derivations.owner_source`
  - `field_derivations.routing_key`
  - `field_derivations.emission`
  - `test_observations[]`
  - `rejected_decoys[]`
