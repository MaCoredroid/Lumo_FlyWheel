# Final bounded review — 22 September 2026

The v2 paper has completed its authorized experiment and independent review loop. No unresolved material finding or indispensable additional experiment remains for its stated claims. This is a private review draft, not a public submission.

- [Paper PDF](main.pdf): 11 pages, compiled and visually inspected.
- [Final manuscript review](p0/monitor/paper-final-e1-redteam.md): all material findings closed.
- [Final artifact review](p0/monitor/e1-artifact-final-redteam.md): essential evidence, active sources, reducers and qualification dependencies complete.
- [Delivery receipt](artifacts/FINAL-DELIVERY.json): exact archive/companion hashes, extracted rebuild and analysis checks, and remote-copy verification.

| Completed scope | Outcome |
| --- | --- |
| P0 provenance and measurement audit | Complete; historical unknowns retained |
| E7a | Two kernel campaigns, 8 pilot inputs, 31 eligible confirmation inputs out of 32 originals; original frozen criteria failed and no compact candidate promoted |
| E7b | Six corrected diagnostic boots; narrow stage-isolated and offline continuation observations |
| E2 | Selected synchronous flat-map B1/B4 routes pass bounded helper/live checks; incompatible policy A retained as a failure witness |
| E1 | All 18 original timing cells independently verified; no replacement or extension; all six prespecified precision targets met |

The measured rates for native chain-5 / native chain-11 / tree are 13.67 / 11.44 / 12.69 tokens/s at B1 and 55.38 / 42.15 / 47.28 at B4. Tree is 7.14% and 14.61% below chain-5, respectively. These are as-executed instrumented configuration measurements on fixed prefixes. Engine seeds differ, output streams differ, and three-block intervals are coarse. The paper makes no matched-output, quality-preservation, full-model equivalence or isolated algorithm-speedup claim. Bole/TreeWY mechanism families are compared through local realizations, not their authors' serving systems.

The current paper archive and member/companion verification are identified by `artifacts/FINAL-DELIVERY.json`. The previous 15-page package and its extracted accounting/E1 replay checks remain preserved as dated audit evidence. The current 11-page source/PDF passed the independent historical-number removal review and canonical build/visual checks in `p0/monitor/2026-09-22-historical-removal-build.json`. Historical numerical results and remeasurements on archived operands are excluded from the manuscript; fresh findings and failures remain. External pinned model/image/runtime dependencies and explicit DGX path mapping remain necessary for fresh GPU execution.

At the earlier 12:33 UTC closure, the DGX was idle. Guarded cleanup restored approximately 102.5 GiB available RAM and zero swap used while leaving swap enabled. The ten-minute monitor is paused; its confirmation is recorded in the delivery receipt.
