# E8 companion exporter: bounded independent review

2026-09-22. **PASS for the corrected exporter source and packaging plan; no unresolved material source finding.** No archive, experiment, GPU, Docker or network operation was performed. Actual final packaging/relocated replay remains a completion-time acceptance step after all six cells and the final review are available.

Reviewed settled hashes:

- `scripts/export_e8.py`: `e50c3ce4438a7e148180616e0b3bad5227ba840f5ef817deb60db7d37cda291f`.
- `notes/e8-packaging-plan.md`: `e2a727cca4dd54fd89065dd6defea64138339c1471e497720327058e1f59562f`.

One concrete finding was raised and closed. The initial exporter included only the original E1 `loaded_backend/eagle.py`; both shipped `e8_prepare.py` generators derive their four unchanged-module pins by globbing the same directory. The corrected exporter lines 183–189 now includes all four other modules and verifies each against the qualification manifest. I independently verified the original input hashes:

| Original loaded module | SHA-256 |
|---|---|
| tree_attn.py | a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97 |
| gdn_linear_attn.py | 723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28 |
| rejection_sampler.py | 5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f |
| gpu_model_runner.py | b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40 |

The corrected source covers all six original/failed/qualification/timing source-and-run roots; exact attempted manifests; source overlays, workload/recorder/joiner, frozen prefixes and raw API/event/head/ownership evidence; all six terminal timing seals; qualification receipts; the final source-bound aggregate; and the necessary original generator/CPU-regression inputs. Those prior E1 files remain explicitly labeled regression/source dependencies and are not substituted for fresh E8 measurements.

The original infrastructure failure is preserved with its original source snapshot. Its current `FAILED.json` still hashes to `60a01a22ed29afa94e0100f43bdd417746df1fa80905145eafeac10bab235096`. The helper checks the known pre-container failure condition and absence of scientific requests, and archives its raw bytes. All seven pinned review/provenance files match their expected hashes. A final actual-timing review must be explicitly selected and hash-bound through the additional-sources map.

Closure checks require the completion marker, exactly the six planned cell directories, terminal seals, unchanged raw evidence, the exact source/qualification pins and unchanged final aggregate. Missing, additional, failed or preliminary cells refuse complete-campaign export. Insufficient-support cells are preserved; they cannot be silently dropped for a favorable aggregate.

Replay lines 113–141 run all eight frozen CPU joins (two untimed qualification, six timing), compare the complete JSON and original exit codes, then rerun the frozen aggregate. Relocation comparison changes only the newly computed `qualification.root` metadata string after checking both the archived expected suffix and actual extracted path; original bytes and every numeric/result field remain untouched. The extracted-verification path checks all member hashes and invokes the same closure/replay checks. This is sufficient for the documented relocated CPU replay; it does not claim an offline rerun of full-vocabulary tensors that were checked only in memory during qualification.

I re-executed all 14 exporter selfchecks: PASS. They cover missing terminal data, unsafe/symlink paths, wrong/missing source payloads, six sealed cells including insufficient support, missing/preliminary cell seals, changed raw data, campaign failure, extra replacement cells and altered aggregate. No archive was created. Final parent execution should retain the prescribed sequence: completed local transfer and final source map, check-only/replay, export, fresh extraction and `--verify-extracted`. No new scientific experiment is required by this packaging review.
