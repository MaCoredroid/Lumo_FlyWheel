# Notes

- No live benchmarks or gates were run by this worker.
- Candidate 004 is intended as the lower-concurrency safety point after candidate 002 at concurrency 8 and candidate 003 at concurrency 6 both failed B-3 workload equivalence with 0.75 match rate.
- The controller should measure whether concurrency 3 still clears the 15 tok/s decode target while improving B-3 serial-vs-batched agreement.
