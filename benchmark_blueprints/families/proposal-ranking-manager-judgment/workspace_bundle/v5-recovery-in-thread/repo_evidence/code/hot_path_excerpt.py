# Excerpt from src/respproxy/hot_path.py (current v0.19.0)
# Lines 120-168; trimmed for brevity. Read-only evidence — do not execute.

def handle_tool_call(request, skill_pack):
    # 120: entry
    normalized = normalize_request(request)                        # 52 ms p95

    # 124: re-derive validator every call
    compiled = compile_tool_schema(                                # 128 ms p95 (per-call waste)
        skill_pack.schema_for(normalized.tool_name)
    )

    # 144: structural validation on hot path
    validate_tool_call_args(normalized.args, compiled)             # 84 ms p95

    # 152: dispatch
    result = dispatch_to_skill_pack(skill_pack, normalized)        # 61 ms p95

    # 156: finalize stream
    return finalize(result, request)                                # ~95 ms p95 aggregate


def compile_tool_schema(schema_dict):
    # NO CACHE — recompiles every request.
    # Priya's note (P4) proposes caching; normalizer today copies the dict
    # to guard against in-place mutation by downstream code paths.
    from jsonschema import Draft7Validator
    return Draft7Validator(schema_dict)
