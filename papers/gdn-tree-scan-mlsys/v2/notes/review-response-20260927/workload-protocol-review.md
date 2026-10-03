# Independent E3 workload-protocol draft review

Reviewed September 27, 2026. Scope: `experiments/review-response-20260927/workload-plan/` protocol, selection and provenance scripts, source bindings, candidate schedules, and validation artifacts. No GPU calls, remote mutations, runtime changes, or paper edits. The only file written by this review is this report.

**Disposition: approve the preparation draft. No blocker to the candidate selection or draft accounting was found. This is not approval to launch a frozen confirmation campaign.** In particular, 192 confirmation attempts remain a candidate budget, and the proposed root seeds are not attested sampling controls.

## Reproduction and selection checks

- All **13 unit tests passed** under the configured Python with bytecode writing disabled.
- `verify_artifacts.py --include-current-sources` verified **20 payload files and 12 inspected local source files**, with no mismatches.
- Independently reran the selector's output generation **entirely in memory**, intercepting JSON/CSV writes. All **nine outputs** are byte-identical to both the saved files and the validation hashes; no candidate artifact was overwritten.
- The saved projection has 500 unique official IDs and only the six allowed neutral metadata fields. The projection helper hashes the complete Parquet, then reads only those columns. The fixed pin is SHA-256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`, 2,096,679 bytes, snapshot `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. This review verified the saved projection/source binding and code, not a new remote download of the Parquet.
- Selection is independent of result bodies: exposure collection reads Git/current filenames; task selection reads metadata and excluded IDs. Repository/task ordering uses a fixed SHA-256 rule. The scripts do not use success, patch size, latency, token count, or problem/test contents to admit candidates. This verifies the prepared algorithm; it does not prove that no unavailable external archive contains earlier exposure.
- The **2 pilot, 4 tuning, and 16 confirmation candidates** are mutually disjoint and disjoint from the **61 exposure exclusions**. All **22 official Astropy tasks** are excluded. The selected four repositories are scikit-learn, Requests, SymPy, and Pylint, with four confirmation candidates each; 98 unused deterministic reserves remain. Equal repository quotas are correctly described as a scoped study, not a representative benchmark-wide estimate.

## Schedules, budget, and attempt accounting

The 8/12/16-task confirmation cohorts are nested and maintain all four repositories. Their schedules contain respectively **96/144/192** attempts, with all four arms present in every task/seed block. Across each seed block, every arm appears equally in each ordinal position and all twelve directed within-block predecessor pairs occur equally often. Pilot ordering is explicitly only partially balanced. Reboot/cache-reset rules and any change to the ordering must be frozen before confirmation.

The maximum candidate arithmetic is correct: **8 pilot + 48 tuning + 192 confirmation = 248** task attempts before separately recorded infrastructure retries and component/qualification work. The draft distinguishes a resource ceiling from a requirement to spend duplicate tuning slots, and retains all tested configurations and all starts.

The pilot rule uses operational duration and infrastructure information rather than confirmation outcomes. `1.5 × 3 × N × sum(mean per-arm device occupation) / 3600` correctly budgets three repeats and four arms for N tasks. The factor is explicitly a planning margin, not a statistical bound. The draft requires H, time limits and tuning/qualification reservations before selecting the largest feasible N in {16,12,8}; it forbids outcome-based truncation after confirmation starts. No saved status or schedule silently freezes 192 attempts.

## Conditions still required before workload launch/freeze

These are already identified as gaps in the draft and must remain open until verified:

1. **Dataset/evaluator/harness binding:** implement a separately manifest-bound review mode without bypassing existing provenance checks. All arms and the evaluator must consume the same task bytes/revision. Pin the official evaluator and task images, then preflight native-x86_64 environments. The cached Astropy images do not qualify the newly selected repositories. Any objective feasibility replacement must follow the predetermined same-repository reserves before confirmation outcomes exist.
2. **Actual sampling seeds:** deliver and record root-to-logical-request seeds, including retries/compaction, and confirm effective engine values. Scheduling labels alone do not create seeded replicates. If an engine cannot support the intended rule, revise the repeat interpretation before starting.
3. **Qualified serving arms and identity:** current AR/chain launch routes, immutable image/patch manifests, checkpoint-content hashes, tokenizer/config hashes, and precision/physical concurrency policies remain unresolved. Historical FP8 or unpatched native defaults are not current NVFP4 same-stack controls. The draft states this correctly.
4. **Measurement closure:** qualify abort/retry accounting, additive SGLang acceptance, effective sampling receipts, phase/event alignment, warmup, and telemetry overhead. All starts, empty patches, timeouts, mechanism failures, infrastructure failures and missing-rate counts must remain visible.
5. **Final budget and estimands:** declare H, calendar window, positive common limits, retry rules, tuning configurations/objective, final N and exact run order. Specify the exact time assignment for crashes, timeouts and infrastructure retries when freezing “budget-capped agent time”; a fast failure must not be silently interpreted as faster successful work. Include evaluator scheduling when assessing calendar feasibility, even though the device-hour formula measures GPU occupation.

The current text leaves these fields proposed/unverified and correctly separates pilot/tuning/confirmation outcomes. The draft contains no fresh GPU result and does not qualify the live worker's pending Q1 execution. A later launch review must assess the completed manifests and effective runtime controls, rather than treating this preparation approval as their substitute.
