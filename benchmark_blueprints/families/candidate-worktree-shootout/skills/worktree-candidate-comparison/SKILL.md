# Worktree Candidate Comparison

Use this skill when two plausible implementations should be explored in isolation before one is landed.

## Required workflow

1. Evaluate Candidate A in its own worktree.
2. Evaluate Candidate B in its own worktree.
3. Record exact commands, touched files, and observed outcomes for each.
4. Reject the weaker candidate with evidence, not preference language.
5. Land one clean final strategy without cross-candidate contamination.

## Deliverable checklist

- both candidate evaluation notes exist
- service-level regression test covers the non-CLI path
- final diff reflects one coherent strategy

