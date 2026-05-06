# B-1 Strong-Equivalence Note

Candidate `001` is a serving-shape mutation: it uses concurrent warm cache-hit requests to amortize one dense FP8 weight stream across multiple active sequences. It does not change target weights, sampling parameters, kernel math, or KV contents.

The B-1 artifact `b1_result.json` is therefore a deterministic serial-vs-batched equivalence gate rather than a full teacher-forced KL fixture:

- source gate: `scripts/run_track_b_batch_equivalence.py`
- prompt count: 4
- concurrent requests: 4
- match rate: 1.0
- pass: true

This is enough to admit the candidate for speed exploration. It is not a substitute for B-2/B-3 promotion gates.
