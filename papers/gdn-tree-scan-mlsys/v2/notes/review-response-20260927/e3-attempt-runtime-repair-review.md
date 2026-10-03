# E3 attempt-runtime F1/F2 repair: independent bounded review

**Disposition: approve the F1/F2 CPU preparation repair.** Both original blockers are resolved in the reviewed source. No remaining blocker was found within these two changed paths and their seal/test consistency. This is not live-runtime qualification, evaluator/task execution evidence, workload evidence, or launch approval. No gates, accepted dependencies, or implementation files were changed by this review.

## Reviewed identity and reproduction

Package: `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/` (paths below are relative to paper `v2/`).

| Artifact | SHA-256 |
| --- | --- |
| `MANIFEST.json` | `be15b0fcefcf9bda4f70168607829a8be8d849cc358935dfdfe0af903b3eea6d` |
| `runtime.py` | `09f35b0faa72839c9a654249f29e20d39b0adae9011082bdb5c0e2cc8778e2e9` |
| `evaluator.py` | `9b52383d2b5f0e0794cb9dee1194fdf8fe74c846cfc6a4b92a7387f220ac2e71` |
| `test_runtime.py` | `ea63dae84af4a9c164133ac5bf1798ae050a0a16142b5ac09c611c569e2ddb15` |
| Author's `unit-tests-repair-v2.txt` | `173cffd9d60c08e0ae103b9b779a97154aec700c3887bd70f68542503596168d` |
| Independent `e3-attempt-runtime-repair-audit/reproduce.py` | `b82c5d901bc5d0aa5985129006219a0df7d988a33ad39d8969185bd895b3eeff` |
| Independent `e3-attempt-runtime-repair-audit/results.json` | `ea1807c96e1107ec6fd42eab2d439e8c4b17ba2191445335d3f4d47a94b4006b` |
| Independent `e3-attempt-runtime-repair-audit/focused-tests.txt` | `6bbf15c662b31cc9189b3810ab2798789bf184be3876455df598e1f3daef2fb2` |

Independent commands, run from `v2/` with `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B`:

```text
notes/review-response-20260927/e3-attempt-runtime-repair-audit/reproduce.py
experiments/review-response-20260927/workload-plan/tools/attempt-runtime/verify_bundle.py
```

The focused run passed all **18 added F1/F2 tests** (1.240 s), plus the independent controls described below. The sealed source-bound author log reports **91 tests passed**; I verified its hash and retained 73 original test names plus 18 added names, but did not rerun all 91 or the unchanged accepted dependency suites. The seal verifier passed **63 payload files / 11 source bindings**, preserved the 28-file submitted snapshot and its 13-file predecessor, and verified the accepted adapter **29**, collectors **25**, and closure **30** files unchanged. Its output is retained separately as `e3-attempt-runtime-repair-audit/seal-verification.json`.

## F1 — observed boot transition and real collector: resolved

`runtime.py:130–155` now enters the agent stage using retained launch identity, actual injected Docker server identity/ownership/image/StartedAt, and a clock observation. The start must fall between the launch gate and the observation. The transition retains the launch-packet hash and raw Docker/clock evidence before updating stage and boot time. The unchanged collector then reads the same boot independently and supplies the gate-close time.

I reproduced the original launch → `run_agent` sequence with the actual `GateHarness` and `Collector`, without the removed `PrecollectedAgentGates` shim. The collector ran once, raw boot identity matched the retained transition, launch-packet bytes remained identical, one injected agent start occurred, and closure was `CLOSABLE`. The eight focused F1 controls also passed, including refusal of wrong ownership/boot, absent or out-of-interval boot time, a restart between reads, missing probe, and changed launch timeline. This verifies the formerly broken CPU integration path; the start and Docker observations were injected fixtures, not live operations.

## F2 — internal pinned-row TestSpec construction: resolved

`evaluator.py:145–147` refuses **every caller-provided TestSpec** before retaining an invocation or calling the evaluator body. The original same-instance-ID/wrong-repo/wrong-version/arbitrary-script substitution was refused with zero evaluator calls and no invocation artifact.

The positive path executes `selected_record` internally (`evaluator.py:27–35,55–75`): it checks dataset bytes before and after reading, requires exactly one selected row, and matches all five metadata fields (`instance_id`, `repo`, `base_commit`, `version`, `environment_setup_commit`). It resolves the constructor at the bound module/file/code path, verifies its source hash, and constructs from that row with the bound `x86_64` architecture. The exact retained official dataclass/factory AST is exercised in the fixture; script generators and evaluator body are synthetic, and installed-harness verification is mocked only for this injected body.

I independently recomputed the receipt projection for all **15 dataclass fields** and all **three generated scripts** (`eval_script`, `install_repo_script`, `setup_env_script`), as well as full-row, dataset, constructor, and harness-manifest hashes. The fields cover repository/version, all script lists, FAIL_TO_PASS/PASS_TO_PASS, architecture, language, Docker specs, namespace, and the three recipe tags. Generator spies confirmed the base commit and test patch flow into the official evaluation-script factory, the repository script receives the base commit, and the environment factory receives the complete selected row, including its environment-setup metadata. The whole-row hash also binds fields that the official constructor does not consume.

Official default namespace and recipe tags remain `None`/`latest`; these are hashed recipe fields. The actual image lookup still uses the accepted immutable task digest through `PinnedSpec`, not a mutable tag. Constructor source verified by this path: `f9ab368091b5770d2ab6e842e7d56161f87e70a20519576731c122e310820f4a`.

The invocation receipt contains the new hashes, not row/test/gold/script contents (`evaluator.py:70–74,156–159`). Synthetic test/gold and test-name sentinels were absent from both this receipt and `safe_agent_input`. This is a check of the changed receipt and public-input boundary, not a new assertion about an executed evaluator's logs. Additional independent wrong-instance, base-commit, version, and environment-setup controls all refused before evaluator body execution; the focused suite also covers wrong repository, duplicate rows, changed dataset before/during reads, and wrong constructor import/function/file identity.

## Retention and remaining boundary

The original failure report/reproducer/results and submitted source remain sealed under `review-v1/` and `superseded/submitted-v1-2f15cfc427e0/`. New observed refusals and positive-control output are in the separate review audit directory; historical failure evidence was not overwritten.

Remaining qualification is the already documented real transport/clock behavior, installed dependencies and official harness, actual runtime source/model/probe/seed/accounting collection, and the real caller smoke under the parent's gates. This review performed no Docker/SSH/model/evaluator/task/workload operations and grants none of those approvals. The repair is ready for that existing next qualification stage when separately authorized; it adds no experiment result or scientific claim.
