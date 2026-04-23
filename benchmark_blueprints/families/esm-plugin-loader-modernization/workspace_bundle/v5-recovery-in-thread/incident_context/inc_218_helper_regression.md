
# INC-218 helper regression

A prior rollback happened because the ESM loader patch stopped rejecting malformed plugin modules and broke the `.cjs` helper path used by a release-only plugin. Do not repeat that recovery mistake.
