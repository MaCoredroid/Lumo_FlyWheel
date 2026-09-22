# E8 extracted companion: final independent review

2026-09-22. Reviewer: paper_redteam_round1. **PASS: no unresolved material completeness, relocation, source-generation, or result-binding finding.** Review was local and CPU-only. Original archive, extracted evidence and attempted sources were not modified. Source regeneration used a separate disposable temporary copy; no Docker, model import, inference or GPU operation occurred.

## Exact artifact

- Archive: `artifacts/e8-single-logits-20260922T2255Z.tar.gz`.
- SHA-256: `b4c0a5cbcd2592ce5ddfdc823243f0e99dc15ef412d238a096068d6bae11ebf4`; 16,237,576 bytes.
- 702 evidence payload files **plus** `E8-EVIDENCE-MANIFEST.json`, hence 703 regular archive members. No link or unsafe relative member.
- Embedded manifest SHA-256: `d23ebb793916ae77ebf6051638412463f2150d4fe7e09c4e45bc07252d9e95e0`.
- Extracted root: `/tmp/lumo-e8-final-extracted-20260922T2255Z`.
- Exporter SHA-256: `e50c3ce4438a7e148180616e0b3bad5227ba840f5ef817deb60db7d37cda291f`; plan SHA-256: `e2a727cca4dd54fd89065dd6defea64138339c1471e497720327058e1f59562f`.
- Independent review JSON: `p0/monitor/e8-artifact-final-independent-review.json`, SHA-256 `927c20fb68576748452c3e321315b2548dfb8444e3d9ddab7f32047710c0fbdf`.

## Independently verified closure

Every one of the 702 manifest payloads matches both its current original file and its extracted file byte for byte; every archive member matches extraction. Comparing directory file sets independently confirms complete preservation of the six required roots, apart from named caches: original source (23 files), original failed attempt (25), qualification-v2 source (27), actual qualification run (108), timing source (36), actual timing campaign (421). The original `FAILED.json` and attempted stage retain the pre-container comma-command failure; it is not relabeled as data, silently fixed, or merged into the scientific count.

The archive pins original/qualification/timing manifests `45da9147…`, `bc2feab3…`, `8bbb5b48…` and actual qualification pass `815b348b…`. It contains both qualification receipts and raw API/ledger/head/loaded-source evidence, all six exact timing terminal seals and their complete raw API/support/configuration/ownership records, start/completion markers, and frozen aggregate `74b0988b191bb68fad43d41f76f520d49d529e86753ac0f18535721f6c3757fb`. The final actual-data review `8f246e66…`, independent reconstruction JSON `274566f0…`, checker `1f22d65b…`, final worker status and wrapper receipt are present and hash-bound. The six terminal hashes match the independent final timing review; no seventh cell, missing arm, substitution or selective result omission appears.

Executed the **extracted** exporter with `--verify-extracted`. All member hashes passed, both qualification raw joins reproduced the exact stored JSON with rc3, all six timing raw joins reproduced exactly with rc0, and the complete frozen aggregate reproduced. The sole comparison normalization is the relocated absolute `qualification.root` string; archived original bytes and all numerical/source/status fields remain unchanged. The reproduced aggregate retains all three pairs, stream divergence, EOS and the frozen mean-paired-relative estimator. Independent replay output is preserved in the review JSON.

## Source-generation dependency check

The exporter correction at `scripts/export_e8.py:183–189` includes and pins all four unchanged original loaded modules in addition to the original drafter. The companion also supplies all six repository source overlays, seven E1 data/helper inputs, the hash-pinned base v7 launcher, and the specific E1 CPU regression fixture at their required relative paths. They are explicitly labeled source-generation/regression inputs, not E8 measurements.

To test completeness beyond file existence, copied these dependencies and the two stages into a disposable temporary tree, then executed only qualification `e8_prepare.py` followed by timing `prepare.py`. The qualification manifest regenerated **exactly** as `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`, with all 23 pinned payloads matching. Timing regenerated **exactly** as `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`, with all 33 payloads matching. This closes the prior missing-loaded-module packaging finding with an execution-level check of the actual extracted dependency set. The delivered source freezes were untouched.

The package accurately excludes model/image bytes while retaining their identities and weight-preflight evidence. Python standard library suffices for the tested raw CPU reproduction; fresh serving requires the declared external runtime. Full-vocabulary equality was checked in memory during qualification, not captured as raw tensors, and the package does not promise offline full-tensor replay or full-model equivalence. No further experiment or artifact repair is necessary for this companion's bounded evidence claims.
