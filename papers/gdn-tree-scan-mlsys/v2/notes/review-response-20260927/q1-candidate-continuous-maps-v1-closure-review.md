# Candidate continuous maps — F1/F2 repair closure

**PASS for the bounded source/CPU scope.** Exact reviewed mapper SHA-256: `c169b0b990dddc872e196057fc45b61644996b7b55d1334b9afa36041a05771e`. Both previously reproduced false admissions are closed; no remaining blocker found in this closure. This is not launch or qualification approval.

All **13 retained actual-mapper CPU controls pass (3 positives, 10 refusals)**. The original failing source, results and review remain unchanged; the initial seal is preserved as `REVIEW-SEAL.json.initial`.

- F1: actual running and spare SSI rows must be positive and fit both live conv/SSM banks of the specified geometry/dtype. Both out-of-range row99 cases now refuse.
- F2: the mapper derives each selected running conv/SSM view's actual pointer and contiguous byte extent, checks backing-storage bounds, sorts spans per device, and rejects any overlap (source lines 68–80). The retained coherent group0/group2 row4 collision now refuses. Shared physical banks with disjoint selected rows remain accepted.
- Real **992→993** SSI column0→1 and **993→1024** column1→1 transitions remain accepted. Actual consumed-column remapping, prior attention-page remapping, owner/bank tuple replacement, metadata census corruption and CPU/GPU table disagreement still refuse. All cache tensor version counters remain unchanged and the import latch remains zero.

The `extension` function is AST-identical to the initial reviewed source: the fixes add live structural admission checks, without changing prior-map retention, numerical criteria, declared populations or state-byte comparison scope. Its output continues to state `state_bytes_compared=False` and `qualification=False`. The source uses views/metadata reads only; it does not invoke the importer or model.

Evidence is under `p0/monitor/review-response-20260927/candidate-continuous-maps-independent/`: `controls_final_repair.py`, `final-repair-results.json`, `final-repair-controls.log`, exact final mapper/test snapshots, and final `REVIEW-SEAL.json`. The CPU fixture uses real tensors with the retained candidate geometry and unchanged production bank identity methods. No remote command, GPU, runtime, gate or qualification-count change occurred. Future frozen callers must bind these reviewed bytes and retain separately required raw-state/phase/launch evidence.
