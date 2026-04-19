# Request Path Trace

Use this skill when the task asks for an evidence-backed explanation of how one field or decision flows through a repo.

## Objective

Reconstruct the live path, not the most grep-visible path.

## Procedure

1. Start from the ingress surface named in the task and identify the first concrete symbol that receives the input.
2. Walk forward through the live callers and callees. Record only symbols you can justify.
3. Separate derivation steps from emission steps. A field being serialized is not the same as the field being decided.
4. Identify at least one plausible decoy path and explain why it is not live.
5. Produce a machine-readable path artifact with adjacency, not just a prose summary.

## Guardrails

- Do not trust docs over code without direct confirmation.
- Do not infer liveness from names alone.
- Do not “improve” the architecture unless the task explicitly asks for a code change.
- If evidence is missing, lower confidence and score yourself accordingly rather than filling gaps.

