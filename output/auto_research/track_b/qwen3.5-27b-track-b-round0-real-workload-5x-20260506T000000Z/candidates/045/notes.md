No blocker. Candidate 045 intentionally uses a stricter ngram lookup window than the prior fast-but-risky speculative-decode shapes to reduce B-1 empty/truncated output risk while keeping a nonlocal speculative-decode speed lever.

Measurement caveat: candidate 043 is listed as unknown with `num_speculative_tokens=4`, `prompt_lookup_min=5`, and `prompt_lookup_max=8`; this candidate is measured-distinct by raising the minimum prompt lookup to 6.
