# Excerpt from src/respproxy/normalizer.py (current v0.19.0)
# Lines 100-148. Read-only evidence.

class Normalizer:
    def __init__(self, skill_pack):
        self._skill_pack = skill_pack

    def normalize(self, request):
        schema = self._skill_pack.schema_for(request.tool_name)
        # 112: defensive copy because upstream call sites historically mutated
        # the schema dict in place. This copy is the reason compile_tool_schema
        # cannot cache on object identity.
        schema = dict(schema)
        args = self._coerce(request.args, schema)
        return NormalizedRequest(
            tool_name=request.tool_name,
            args=args,
            meta=request.meta,
            compiled=None,  # 140: validator is (today) built in hot_path.py
        )

    def _coerce(self, args, schema):
        # trim: type coercion, enum aliasing, timezone normalization
        ...
