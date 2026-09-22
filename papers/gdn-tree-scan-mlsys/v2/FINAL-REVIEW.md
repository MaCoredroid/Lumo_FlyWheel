# System-design review — 22 September 2026

The author-approved reframing is complete. The paper now develops the integrated tree verifier, GPU scan/replay execution, and implemented optimization mechanisms. The accounting audit supports the methods and appendix. No material manuscript finding remains at the reviewed source hashes.

- [Paper PDF](main.pdf): 14 pages, compiled and all pages visually inspected.
- [Design manuscript review](p0/monitor/design-reframe-redteam.md): PASS at main `0d7bab38` and abstract `887f36b6`.
- [Build and preservation checks](p0/monitor/2026-09-22-design-reframe-build.json): clean build, all citations resolved, historical numbers excluded, fresh numerical results unchanged.
- [Implementation/evidence map](notes/design/optimization-evidence-map.md): implemented mechanisms, actual measured routes, and missing attribution experiments.
- [Independent route audit](notes/optimization-route-audit-2026-09-22.md): exact frozen-source and live-route boundaries.

| Completed core scope | Outcome |
| --- | --- |
| P0 | Provenance and measurement audit complete; historical unknowns retained |
| E7a | Eight pilot inputs and 31 eligible confirmation inputs out of 32 originals; original frozen criteria failed and no compact candidate promoted |
| E7b | Six corrected diagnostic boots; narrow stage-isolated and offline continuation observations |
| E2 | Selected synchronous flat-map B1/B4 routes pass bounded helper/live checks; incompatible policy A retained as a failure witness |
| E1 | All 18 original timing cells independently verified; no replacement or extension; all six prespecified precision targets met |

The fresh measured rates for native chain-5 / native chain-11 / tree remain 13.67 / 11.44 / 12.69 tokens/s at B1 and 55.38 / 42.15 / 47.28 at B4. These are as-executed instrumented configuration measurements on fixed prefixes. Engine seeds and output streams differ, and three-block intervals are coarse. No matched-output, quality-preservation, full-model equivalence or isolated algorithm-speedup claim follows. Local Bole/TreeWY mechanism realizations are distinct from those authors' serving systems.

E8 qualification is independently PASS on both actual boots, with515paired head checks per arm and all eight32-ID API streams equal across arms. The exact six-cell B1 timing harness also passes independent review and has been handed to the existing worker. There is no E8 timing result yet, and qualification supplies no performance estimate. It does not establish universal full-model state equivalence. The initial pre-container launcher failure remains preserved; the corrected qualification changed launcher plumbing only.

The preceding dated packages and receipts remain preserved. The current delivery receipt identifies the most recently packaged revision; compare its source hashes to the build proof above. Historical quantitative results remain excluded from the paper, while raw historical audit files are retained. External pinned model/image/runtime dependencies and explicit DGX path mapping remain necessary for fresh execution. This is a private review draft; no publication or submission was performed.
