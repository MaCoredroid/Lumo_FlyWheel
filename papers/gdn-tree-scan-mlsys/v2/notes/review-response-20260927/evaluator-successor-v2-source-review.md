# Evaluator runtime successor v2 — bounded provisioning review

**PASS for the saved CPU provisioning/source/import evidence, with the limits below. No runtime gate, evaluator task, agent attempt, or GPU work is approved by this note.**

The review used the local mirror under `workload-plan/runtime-deps/evaluator-successor-v2/`. It did not run the provisioning scripts, import the installed harness again, connect remotely, or execute tasks. The reviewer artifacts are `p0/monitor/review-response-20260927/evaluator-successor-v2-source-review/{controls.py,SNAPSHOT.json,RESULT.json}`: 319 checks, including metadata/hash checks, across 24 retained source/evidence files.

## Verified

- **82 exact wheel locks.** `requirements.in`, the pip resolve report, `requirements.lock`, and the installed `pip-freeze.txt` agree on exactly 82 unique normalized names and versions. Every report URL is an HTTPS `files.pythonhosted.org` `.whl` with a valid SHA-256; every hash matches its generated lock entry. These are 82 wheel distributions, including pure-Python wheels, not 82 native compiled binaries. The resolve report targets Linux aarch64, CPython 3.12.3. `install_remote.py:7–16` validates the reported versions/origin/hash shape, downloads with `--only-binary=:all: --require-hashes`, and installs offline with `--no-index --find-links ... --require-hashes`. Saved download, install, and `pip check` all return zero, with `No broken requirements found.` All 13 mirrored artifacts in `REMOTE-FETCH-VERIFIED.json` match their listed hashes.
- **Separate environment.** `resolve_remote.py:5–10` selects `.workload-runtime-venv-v2`, refuses an already-existing directory under normal Python, and invokes plain `python3 -m venv` without `--system-site-packages`. All recorded pip commands and the observed import interpreter use the new v2 path. No provisioning source selects the old `.workload-runtime-venv`; the original dependency receipt remains intact in the local evidence tree. This supports non-replacement by these commands; it is not a fresh byte-for-byte audit of the old remote environment.
- **Official harness source/import evidence.** The embedded known identities in `verify_remote.py` equal the current accepted `known-locks-v2.json`. Its verification loop (`:5–9`) imports `swebench.harness.run_evaluation` and checks the pinned SHA and byte length of every declared source member. The saved receipt lists all 73 expected members, exactly matching the known source list, with actual paths under the new v2 `site-packages/swebench`. The official module resolves there; SWE-bench is 4.1.0. The saved SSH stdout and verification receipt agree. The receipt records `torch_imported: false`; the reviewed script imports/checks only and contains no task/evaluation call.
- **Public-task projection.** `verify_remote.py:10–16` first checks the pinned dataset hash, then calls the Parquet reader with exactly `instance_id`, `repo`, `base_commit`, `problem_statement`, and `version`. It selects exactly one `scikit-learn__scikit-learn-9288` row and checks base commit `3eacf948e0f95ef957862568d87ce082f378e186`. The saved JSON has exactly those five keys and the receipt-bound hash. Actual extracted projection code, run with an in-memory reader stub containing gold/test sentinels in additional fields, produces only the public projection; absent/duplicate task and wrong base commit refuse. Gold/test columns are not decoded into that object or disclosed in the saved public JSON. The separate full-file checksum reads raw Parquet bytes; this is not a claim that dataset bytes containing gold were never read.

The actual extracted installer validation also rejects wrong version, non-wheel URL, wrong download host, malformed SHA, and missing package controls. These controls exercised only selected AST statements with in-memory inputs, not subprocesses or installation.

## Scope limits and next integration boundary

1. The mirror does not include the wheel payloads or installed v2 package tree. This review independently verifies exact report/lock equality and the saved successful hash-enforcing pip commands, not a second live checksum of all wheels/installed files. The 73 installed source checks are supported by the source-bound verifier and its saved execution receipt.
2. There is no post-provision old-venv byte inventory in this package. Preserve the narrower statement that the reviewed commands target a separate new path and do not modify the old venv; do not upgrade that to independently observed current byte identity.
3. **The new installed dependency identity differs from the historical evaluator gate.** New `pip-freeze.txt` SHA is `3bc94b8315f33341923a28f5d7a0d2db164adf3fc9c85a094be486d34700a57e`; current `e3_preflight_v5.py:382` still requires `3cbf393abe3825735a48b47e871267e3397d8e5556d57493645637f24ef2eb6c`. Source/import success cannot satisfy that mismatch. Any later runtime admission must explicitly and prospectively bind the intended host/interpreter/dependency identity; this note changes neither the known lock nor a gate and makes no claim about the separate x86 evaluator host.
4. The scripts are one-shot provisioning/verifier programs using Python assertions, not hardened admission gates. Their checks assume normal, non-optimized Python. `PLAN.json` remains a planning artifact (`RESOLVE_ONLY`); completion is evidenced by later installation/import receipts, not by that plan status. No task-quality or performance inference follows.

## Exact reviewed identities

- `resolve_remote.py`: `aee92eb0dea812215b68c487995744982de15ac32f3dfba944374ce3dd084c5f`
- `install_remote.py`: `8c87c85bfea3c4a0041a407eb3743b8c0c8dfde350b59e83909e21d510b5b830`
- `verify_remote.py`: `f106b06d1bf43912974c4eb6a862e4ae4fa8686673be3e8fddeedef333a982d4`
- `requirements.lock`: `82216e16c9a0a99a7d8d9e22670e0a8f1594e0c69ae31962824cc17fe92d7934`
- `SOURCE-IMPORT-VERIFICATION.json`: `7f33d13fd9dc66200fe993f02f283110b9135e9691bdba7bc54fb8fb21299e1e`
- `public-task.json`: `8bc86253c3e09fbf938c95cd3ed90a3fffb1a10d635c8af3f458d6aca3362cd9`
- `REMOTE-FETCH-VERIFIED.json`: `2271a610164b798a7cff487c2a2b0344725d8737862df7857c4f1cc04a9036cf`
