# Latest design and completed E8 review — 22 September 2026

The manuscript describes the exact qualified serving route and its architecture, sequential scan/replay, state publication and active optimizations. Superseded FA2 implementation narratives and pilot-only numerical summaries have been removed; original failures and immutable data remain in the evidence archive. No historical quantitative result has been restored.

E8 is complete: two qualification boots and all six fixed B1 timing boots independently PASS. ON/OFF mean rates are 12.69119/9.69541 tokens/s; the frozen mean of three paired relative changes is +30.89888%, with coarse 95% paired-block bootstrap interval [29.80504%, 32.90320%]. All original cells remain included. The last ON boot differs on all eight continuations, including p021 EOS at 101 tokens. This is an instrumented rate comparison, not equal-output acceleration or quality evidence. E1 already enabled reuse; E8 does not increase its Cat10 row again.

The E1 means remain native MTP-5 / MTP-11 / Cat10 = 13.67 / 11.44 / 12.69 tokens/s at B1 and 55.38 / 42.15 / 47.28 at B4. These are fixed-configuration means, not best-case tuning results. B4 is aggregate throughput. E1 retains the engine-seed deviation and full-stream divergence. E7a reports the final 31/32 eligible confirmation result and failed frozen criteria; neither compact candidate is promoted. Local Bole/TreeWY realizations remain distinct from their authors' systems.

Review and verification records:

- `notes/latest-design-supersession-audit-2026-09-22.md`: repository/source freshness and route applicability.
- `p0/monitor/e8-timing-final-redteam.md`: complete independent raw six-cell reconstruction.
- `p0/monitor/paper-latest-design-e8-redteam.md` and `paper-e8-results-redteam.md`: final manuscript reviews.
- `p0/monitor/2026-09-22-latest-design-e8-build.json`: canonical build and visual QA.
- `artifacts/FINAL-DELIVERY.json`: final source, PDF, archive and extraction identities.

E8 has no B4 or composed-optimization result, and no full-model equivalence or quality result. No further GPU experiment is needed for these bounded claims. All owned GPU work is stopped; the final recovery receipt records 102.29 GiB available RAM and zero swap. No publication, submission, external message or push was performed.
