# Pooled decode speed-claim audit — 23 September 2026

PASS. The complete loaded manuscript (abstract, body, three TikZ figures, tables and captions) and current claim documents use one decode estimator:

`D_pool = (sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`.

| Claim | Verified value | Disposition |
| --- | --- | --- |
| Tree pooled decode | 248077 / 9677.79721069336 = 25.6336224659 tokens/s | Abstract, results, Table V, conclusion: 25.63 |
| SGLang EAGLE pooled decode | 47809 / 1777.685661306139 = 26.8939560242 tokens/s | Same estimator and sites: 26.89 |
| Tree relative difference | -4.68630779794% | 4.69% lower; descriptive, not causal or task-completion speedup |
| Workload | Ten tree tasks / 265 requests; two SGLang tasks / 45 requests | Different SWE-bench Verified Astropy subsets and deployed settings disclosed |
| Agent runtimes | Original task metadata in minutes | Kept separate from decode speed and evaluation time |
| Optimization mechanisms | Work avoided, resource/lifetime tradeoffs | No unsupported isolated or multiplied speed gain |
| Local numerical studies | Error magnitudes, counts and bounded continuation checks | Not performance results; kernel timing values remain audit-only |
| Historical and local-document rates | Superseded estimators / non-agent workloads | Absent from loaded paper and PDF; original evidence preserved |

The current abstract, results equation/table, conclusion, README, claim ledger, baseline inventory and proposed experiment definition agree. The common CPU reducer reproduced its saved JSON byte-for-byte from 48 hash-bound inputs. It includes all completed engine requests in the selected task brackets, including internal traffic; request time excludes TTFT and between-request tool work and is not machine-wall throughput. The original trace audit supports no detected degeneration in the ten tree tasks only.

The accompanying JSON binds every audited source and the rendered PDF and lists all keyword hits for review. Dated calculation-only notes and their original manifests were preserved rather than retroactively changing their scope. Independent manuscript and artifact reviews are recorded in `paper-pooled-speed-redteam.md` and `../../notes/pooled-decode-paper-artifact-redteam-2026-09-23.md`. No inference was launched.
