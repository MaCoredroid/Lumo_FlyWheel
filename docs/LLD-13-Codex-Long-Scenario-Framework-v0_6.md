# LLD-13 · Codex-Long Scenario Framework

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S0 proto-scenario + S0b → Implement S0b  
> Status: SIGNED OFF v0.6

---

## Changelog

| Version | Change |
|---|---|
| v0.6 | **Two P1 cleanups, two polish fixes — targeting freeze.** P1-1: Milestone helpers colocated under the verifier tree — physical location changed from `milestones/<family_id>/` to `verifiers/<family_id>/milestones/`. Phase 3 `docker run` already bind-mounts `verifiers/<family_id>/` to `/verifier/`, so milestone helpers are now at `/verifier/milestones/` with no additional mount. §3.1, §9.3, §10.1 updated to match. P1-2: Phase 3 agent-filesystem extraction changed from `docker inspect --format={{.GraphDriver.Data.MergedDir}}` (overlay2-specific) to `docker cp` (driver-agnostic). Added note that rootful Docker with overlay2 on the DGX Spark is the tested configuration. Polish: Tier 2 audit log example now includes `spoofed_functional_success` row; freeze checklist `benchmark_manifest.lock` sentence updated to reference the full v0.5+ field set including `family_spec_hash` and `grader_image_digest`. |
| v0.5 | **Two P0 patches, two P1 cleanups.** P0-1: `family_spec_hash` added to `benchmark_manifest.lock` per variant — covers `functional_checks` commands/timeouts and all other family-spec content. Enforcement contract Phase 1 now verifies this hash before agent launch. Post-freeze silent changes to Phase 2 grading semantics are now structurally impossible. P0-2: Trusted Phase 3 invariant requirement added — every family must have ≥ 1 `grading_invariant.expected_final_state` check that can independently reject a spoofed Phase 2 result using trusted tools only (no reliance on Phase 2 exit codes). Added to §3.2 checklist and §6.1. Tier 2 adversarial audit (§11.2) expanded: "spoofed functional success" (shadow `npm`/`pytest`/`cargo` to exit 0 without running real tests) is now a required exploit class — verifier must still return `pass: false` when Phase 3 integrity checks detect the spoof. P1-1: `grader_image_digest` verification added to enforcement contract Phase 2 — LLD-03 must confirm the `codex-long-grader` image digest matches the manifest before launching any Phase 3 container. P1-2: Milestone execution authority resolved — `verify.sh` is the single source of truth for all milestone results. Milestone scripts (§9) are helper functions sourced or called by `verify.sh`, not independently executed by LLD-03 or LLD-05. §9 rewritten to remove the ambiguity. |
| v0.4 | **Two P0 patches, two P1 cleanups.** P0-1: Verifier trust model fixed — grading split into three phases: (1) `docker commit` snapshot, (2) functional checks (test suites) run sandboxed inside the agent's own runtime with `--network none`, (3) integrity checks (checksums, greps, file inspection) run in a trusted grading container with a pinned base image. The verifier interpreter and all integrity-check tools (bash, jq, sha256sum, grep) now come from the trusted image, not the agent's rootfs. Agent filesystem is mounted read-only at `/agent/` for inspection. An agent that tampers with `/bin/bash` or shadows `sha256sum` cannot subvert integrity checks. P0-2: Snapshot mechanism changed from `docker export`/`docker import` to `docker commit`, which preserves ENV, WORKDIR, USER, and PATH in the committed image config. Functional checks (Phase 2) launch FROM the committed image and inherit the agent's full runtime config natively. Prose narrowed: export-import config-loss caveat removed. P1-1: `--network none` added to both Phase 2 (functional checks) and Phase 3 (integrity verification) grading containers — enforced in the example launch commands, not just described in §6.5. P1-2: §4 type distribution preamble now cross-references the §12.1 Public-Dev carve-out instead of stating the 5-type-per-split rule as a universal invariant. |
| v0.3 | **Two P0 patches, two P1 cleanups.** P0-1: Reverted to HLD v2.3 behavior — 35-family geometry restored to `~20 / ~7 / ~6 / ~2`, Rule 1 floor restored to 8 families, B1 killed on 35-family path as pre-registered. The 5-type-per-split constraint now has an explicit carve-out for Public-Dev when family count is too low to satisfy it; the inconsistency is flagged as a recommended HLD errata but this LLD no longer overrides the HLD. P0-2: Grading execution semantics fixed — the exported agent filesystem is rehydrated as the grading container's rootfs via `docker import`, so verifier commands (`npm test`, `gcc`, etc.) execute inside the agent's own toolchain and PATH, not the grading image's. Verifier and milestone scripts are bind-mounted into the rehydrated container at `/verifier/`. §6.1, §6.4, §7.1, and all code examples updated to match. P1-1: Manifest enforcement contract (§12.6) extended — LLD-03 now checks milestone script hashes and verifier_data hash before grading, not just image digest and verifier hash before run launch. P1-2: Explicit runtime-state boundary added (§6.5) — Codex-Long verifiers may depend on persisted filesystem state only; scenarios whose success depends on live processes, listening sockets, or external/network state are out of scope. |
| v0.2 | **Three P0 patches, two P1 cleanups.** P0-1: 35-family split geometry fixed — Public-Dev raised from ~2 to 5 families (one per scenario type) to honour the 5-type-per-split rule; Train-Long, Val-Long, Test-Long reduced accordingly; Rule 1 Test-Long floor lowered from 8 to 5 on the 35-family path and consequence updated (B1 remains viable at 5 families but power caveat strengthened). P0-2: Grading surface widened from workspace-only to full writable-layer snapshot; verifier execution uses a writable scratch clone, not a read-only mount; §6 contract, §7 injection protocol, and §10 integrity table rewritten to match. P0-3: `benchmark_manifest.lock` added (§12.6) — per-variant immutable record of image digest, verifier hash, milestone hash, AGENTS.md hash, split, and scenario type; any post-freeze change requires a version bump, change note, and re-gate of affected runs. P1-1: Verifier release timing made explicit — only Public-Dev verifiers are public pre-eval; full verifier suite released after Sprint 3 evaluation completes (§10.4). P1-2: Derived-repo provenance fields added to family spec schema — `source_repo`, `license`, `redistribution_ok`, `modification_notice` required for any variant with `repo_source: derived:*` (§3.1, §3.2). |
| v0.1 | Initial draft derived from HLD v2.3. Covers scenario family spec format, Docker environment factory, state-based verifier design, post-run injection protocol, milestone check scripts, oracle solution isolation, integrity protocol, two-tier audit protocol, family-to-split assignment and freeze procedure, smaller-v1 escape hatch with Rules 1 and 2, B2-only proceed rule for the 35-family path, and Gate 3 proto-scenario exception. |

---

## 1. Purpose & Scope

This document specifies the Codex-Long scenario framework — the foundational benchmark infrastructure that Sprint 0b produces. Every Codex-Long collection run (Sprint 2) and evaluation run (Sprint 3) depends on the artifacts defined here. No full Codex-Long execution can begin until this framework is complete and frozen.

**Responsibilities:**

- Define the scenario family spec format: repo pattern, injected breakage class, grading invariant, milestone definitions, and shortcut-resistance notes
- Implement the Docker environment factory that produces per-variant container images pinned to base image digests
- Implement the state-based verifier and the post-run injection protocol that enforces verifier/oracle isolation
- Implement milestone check scripts (injected post-run, never visible to the agent)
- Enforce oracle solution isolation (used for solvability proof during authoring, never distributed)
- Codify the integrity protocol (verifier isolation, test resource injection timing, shortcut prevention)
- Define and execute the two-tier benchmark audit protocol (Tier 1 type coverage + Tier 2 adversarial audit)
- Manage the family-to-split assignment freeze procedure and enforce the smaller-v1 escape hatch rules

**Out of scope:** Task orchestration and execution loop (LLD-03), trajectory parsing and SFT data formatting (LLD-06), SWE-bench patch-based grading (LLD-05 SWE-bench path), model serving (LLD-01), and the Gate 4 pilot campaign management (LLD-07). This LLD produces the scenario artifacts and verifier contracts that those LLDs consume.

**Proto-scenario exception (Gate 3, Sprint 0):** One Codex-Long-style pilot scenario is authored in Sprint 0 before Sprint 0b begins. It does not require the full LLD-13 framework. See §14.

---

## 2. Scenario Family Model

A **family** is a scenario design template — a reusable pattern that defines a class of agentic coding tasks. A family is not a single task; it is a structural blueprint from which multiple **variants** are instantiated. Variants within a family share structural DNA: the same repo pattern, the same grading invariant shape, and the same milestone structure — but differ in the specific repository state, injected breakage, and expected resolution.

### 2.1 Family vs Variant Distinction

This distinction is load-bearing for the entire benchmark design. The HLD's split is **family-disjoint**, not environment-disjoint: Val-Long and Test-Long must contain families not seen in Train-Long, not merely unseen variants of Train-Long families. Conflating the two invalidates the B1 harness-specificity claim.

| Concept | Definition | Example |
|---|---|---|
| **Family** | A scenario template: repo pattern + breakage class + grading invariant + milestones | `dependency-migration-npm` — migrate a Node.js project from one major dep version to another |
| **Variant** | A concrete instantiation of a family with a specific repo state and injected breakage | `dependency-migration-npm/lodash-3-to-4` — migrate lodash 3.x → 4.x in a specific project |
| **Environment** | A runnable Docker container built from a variant — one env = one task attempt | The container image built for `lodash-3-to-4` with pinned base image digest |

### 2.2 Target Counts

| Split | Families | Envs/Family | Total Envs | Role |
|---|---|---|---|---|
| **Train-Long** | ~30 | ~5–8 | ~150–240 | Primary SFT/RL trajectory collection |
| **Val-Long** | ~10 | ~3–5 | ~30–50 | RL early stopping and hyperparameter selection only — no gradient updates |
| **Test-Long** | ~10 | ~4–6 | ~40–60 | Reported secondary benchmark (sealed until Sprint 3) |
| **Public-Dev** | ~5 | ~3–4 | ~15–20 | Published dev set for reproducibility |
| **Total** | **~55** | | **~235–370** | |

Pre-declared reduced-split geometry for the 35-family path is specified in §12.

---

## 3. Scenario Family Spec Format

Each family is described by a YAML spec file. This is the authoring contract — every field must be populated and reviewed before the family enters the frozen split.

### 3.1 Schema

```yaml
# scenario_families/<family_id>/family.yaml

family_id: dependency-migration-npm
scenario_type: migration_refactor          # one of the 5 named types (§4)
description: >
  Migrate a Node.js project from one major dependency version to another.
  Agent must update imports, adapt API usage, fix breaking tests, and verify
  the full test suite passes under the new version.

repo_pattern:
  language: javascript
  framework: node/npm
  structure: >
    Standard Node.js project with package.json, src/ directory, and test/
    directory. Dependency under migration is imported across 3–8 source files.
  base_image: node:20-bookworm@sha256:<pinned_digest>

breakage_class:
  injection_method: version_bump
  description: >
    package.json is updated to the new major version. npm install succeeds
    but tests fail due to breaking API changes in the dependency.
  surfaces:
    - renamed_exports
    - changed_function_signatures
    - removed_utility_methods

grading_invariant:
  type: state_based
  description: >
    Final container state must have: (1) updated dependency at target version
    in package-lock.json, (2) all source files using new API, (3) full test
    suite passing with exit code 0.
  verifier_script: verifiers/dependency-migration-npm/verify.sh
  functional_checks:
    - id: npm_test
      command: "cd /workspace && npm test"
      description: "Run full test suite under migrated dependency"
      timeout_seconds: 120
  expected_final_state:
    - dependency_version_check: "package-lock.json contains target version"
    - api_usage_check: "no imports of removed/renamed exports"
    - test_suite_check: "npm test exits 0"

milestones:
  - id: m1_dep_updated
    description: "Dependency version updated in package.json"
    check_script: verifiers/dependency-migration-npm/milestones/m1_dep_updated.sh
    partial_credit: 0.2
  - id: m2_imports_fixed
    description: "All source imports updated to new API"
    check_script: verifiers/dependency-migration-npm/milestones/m2_imports_fixed.sh
    partial_credit: 0.3
  - id: m3_tests_passing
    description: "Full test suite passes"
    check_script: verifiers/dependency-migration-npm/milestones/m3_tests_passing.sh
    partial_credit: 0.5

shortcut_resistance:
  notes: >
    Verifier explicitly checks that the original test suite is present and
    unmodified (checksum comparison against known-good test files injected
    post-run). Deleting or modifying test files to make them pass is caught.
    Mocking the dependency to avoid actual migration is caught by the
    api_usage_check which greps for removed/renamed exports.
  known_exploits_tested:
    - "Delete failing tests"
    - "Mock dependency entirely"
    - "Downgrade back to original version"
    - "Disable test runner exit-code propagation"

difficulty_estimate:
  target_solve_rate: "30–50%"
  rationale: >
    Requires multi-file understanding, API documentation lookup via tools,
    and iterative test-fix cycles. Moderate difficulty for 27B models.

variants:
  - variant_id: lodash-3-to-4
    repo_source: authored        # "authored" or "derived:<source_repo>"
    injected_breakage: "lodash 3.10.1 → 4.17.21; 5 call sites affected"
    env_dockerfile: variants/lodash-3-to-4/Dockerfile
    base_image_digest: "sha256:abc123..."
  - variant_id: moment-to-dayjs
    repo_source: authored
    injected_breakage: "moment 2.29 → dayjs 1.11; 8 call sites, plugin system change"
    env_dockerfile: variants/moment-to-dayjs/Dockerfile
    base_image_digest: "sha256:def456..."
  - variant_id: webpack-config-drift
    repo_source: "derived:github.com/example/webpack-starter"
    # ── Required provenance fields for derived variants ──
    provenance:
      source_repo: "https://github.com/example/webpack-starter"
      source_commit: "a1b2c3d"               # pinned commit used as derivation base
      license: "MIT"                          # SPDX identifier of the source repo's license
      redistribution_ok: true                 # legal review confirms redistribution is permitted
      modification_notice: >                  # required by many open-source licenses
        Derived from webpack-starter (MIT). Modified: injected dependency
        version drift in webpack.config.js and removed CI workflow files
        to simulate config skew. Original test suite preserved unmodified.
    injected_breakage: "webpack 4.x config with webpack 5.x binary; loader API mismatch"
    env_dockerfile: variants/webpack-config-drift/Dockerfile
    base_image_digest: "sha256:ggg888..."
  # ... 2–5 more variants
```

### 3.2 Required Fields — Authoring Checklist

Every family spec must satisfy all of the following before it is eligible for the frozen split:

- [ ] `family_id` is unique, kebab-case, descriptive
- [ ] `scenario_type` is one of the five named types (§4)
- [ ] `repo_pattern.base_image` includes a pinned digest (`@sha256:...`), not a floating tag
- [ ] `breakage_class` describes the injection clearly enough for independent reproduction
- [ ] `grading_invariant.verifier_script` exists and is executable
- [ ] `grading_invariant.functional_checks` specifies at least one Phase 2 command (test suite or equivalent) with a timeout
- [ ] At least one milestone with a check script and partial credit weight
- [ ] Milestone partial credits sum to ≤ 1.0 (full solve = 1.0 from the verifier, milestones provide partial credit for RL reward shaping only)
- [ ] `shortcut_resistance.known_exploits_tested` lists at least 3 adversarial approaches and documents why the verifier catches each
- [ ] **Trusted invariant requirement:** At least one `grading_invariant.expected_final_state` check can independently reject a spoofed Phase 2 result using trusted Phase 3 tools only — i.e., the family cannot pass grading solely by making Phase 2 functional checks exit 0. See §6.1.
- [ ] `difficulty_estimate.target_solve_rate` is in the 20–80% range
- [ ] At least 3 variants defined, each with a distinct `variant_id` and `env_dockerfile`
- [ ] Each variant's `base_image_digest` is pinned (not a floating tag)
- [ ] **Derived-repo provenance (required for any variant with `repo_source: derived:*`):** `provenance.source_repo`, `provenance.license` (SPDX identifier), `provenance.redistribution_ok` (boolean, confirmed by legal review or license text inspection), and `provenance.modification_notice` (human-readable description of changes) are all populated. Variants with `redistribution_ok: false` must not appear in Public-Dev or any published data release — restrict to Train-Long or Val-Long only, and note the restriction in the split assignment.
- [ ] Oracle solution exists and passes the verifier (§8)

---

## 4. Scenario Types

Five named scenario types span the Codex-Long benchmark. All five types must be represented in Train-Long, Val-Long, and Test-Long. Public-Dev coverage is subject to the carve-out in §12.1 (at the 35-family path, ~2 Public-Dev families cannot span all five types). Gate 4 requires one pilot family per type.

### 4.1 Type Definitions

| Type ID | Name | Description | Characteristic Agent Behavior |
|---|---|---|---|
| `feature_evolution` | Feature evolution | Implement a spec delta across code, tests, and docs | Multi-file creation and modification; test authoring; documentation updates |
| `migration_refactor` | Migration / refactor | Update an API or dependency across multiple call sites | Codebase-wide search-and-replace with semantic understanding; regression testing |
| `build_ci_breakage` | Build / CI breakage | Dependency drift, toolchain mismatch, config skew | Build system debugging; config file editing; environment troubleshooting |
| `investigate_then_fix` | Investigate-then-fix | Start from logs or a failing integration test | Log analysis; hypothesis generation; targeted fix; verification |
| `cross_layer_changes` | Cross-layer changes | Backend + CLI + config + tests in one session | Multi-component coordination; interface consistency; end-to-end testing |

### 4.2 Type Distribution Target

At the full plan (~55 families), each split should contain at least one family from each type. At the 35-family path, this holds for Train-Long, Val-Long, and Test-Long but not Public-Dev (see §12.1). For the full plan:

| Type | Target Families | Min Families |
|---|---|---|
| `feature_evolution` | ~12 | 8 |
| `migration_refactor` | ~12 | 8 |
| `build_ci_breakage` | ~10 | 6 |
| `investigate_then_fix` | ~11 | 6 |
| `cross_layer_changes` | ~10 | 6 |

Exact distribution is flexible. The hard constraint is structural: Gate 4 requires ≥ 4 distinct scenario types contributing matched families. A benchmark dominated by one or two types is structurally too narrow for a harness-specificity claim, regardless of raw numbers.

---

## 5. Docker Environment Factory

Each variant is a self-contained Docker environment. The factory builds and registers container images from variant Dockerfiles, pinned to base image digests for reproducibility.

### 5.1 Dockerfile Contract

Every variant Dockerfile must:

1. Start from the family's `base_image` with a pinned digest (`FROM node:20-bookworm@sha256:<digest>`)
2. Copy the variant's repository state into `/workspace`
3. Install all dependencies needed for the task (the agent should not need to install missing tools)
4. Apply the injected breakage (version bump, config change, broken import, etc.)
5. Set `WORKDIR /workspace`
6. Not include any verifier scripts, milestone checks, oracle solutions, or test resources used by the verifier — these are injected post-run (§7)

```dockerfile
# variants/lodash-3-to-4/Dockerfile
FROM node:20-bookworm@sha256:abc123def456...

COPY repo/ /workspace/
WORKDIR /workspace

# Inject breakage: bump lodash to v4 in package.json
RUN sed -i 's/"lodash": "^3.10.1"/"lodash": "^4.17.21"/' package.json
RUN npm install

# Verify breakage is real: tests should fail
RUN npm test; test $? -ne 0 || (echo "ERROR: tests pass before agent — breakage not injected" && exit 1)
```

> **The final `RUN` is a build-time smoke test.** If the injected breakage does not actually cause test failures, the Dockerfile build fails. This catches authoring errors where the breakage is cosmetic or the test suite does not cover the affected code paths. Remove this line only after confirming the breakage is validated by a separate mechanism.

### 5.2 Image Registry

Built images are tagged with a deterministic scheme and stored locally on the Spark:

```
codex-long/<family_id>/<variant_id>:<build_hash>
```

The `build_hash` is derived from the Dockerfile content and repo state (not a timestamp). Rebuilding from the same inputs produces the same hash. LLD-03 references images by this tag when launching agent containers.

### 5.3 Build Script

```bash
#!/bin/bash
# build_envs.sh — Build all variant images for a family
set -euo pipefail

FAMILY_DIR="$1"
FAMILY_ID=$(basename "$FAMILY_DIR")

for VARIANT_DIR in "$FAMILY_DIR"/variants/*/; do
    VARIANT_ID=$(basename "$VARIANT_DIR")
    CONTEXT_HASH=$(tar -cf - "$VARIANT_DIR" | sha256sum | cut -c1-12)
    TAG="codex-long/${FAMILY_ID}/${VARIANT_ID}:${CONTEXT_HASH}"

    echo "[BUILD] ${TAG}"
    docker build -t "$TAG" -f "${VARIANT_DIR}/Dockerfile" "$VARIANT_DIR"
    echo "[BUILD] ${TAG} — OK"
done
```

### 5.4 Build Validation

After building all variant images for a family, the factory runs a validation pass:

- [ ] Image starts and `/workspace` contains the expected repository state
- [ ] The injected breakage is present (tests fail or build breaks, as expected)
- [ ] No verifier scripts, milestone checks, or oracle solutions are present inside the container
- [ ] Image size is logged (flag images > 5 GB for review — bloated images slow container spin-up)

---

## 6. State-Based Verifier

Grading is state-based over the **final container state** — the full filesystem as the agent left it — not patch-based. The verifier inspects what the agent left behind, not what it changed.

### 6.1 Verifier Contract — Three-Phase Grading

Grading is split into three phases to separate **functional checks** (which must run inside the agent's own runtime) from **integrity checks** (which must run on a trusted substrate the agent cannot tamper with). LLD-03 orchestrates all three phases; this LLD defines the contracts.

**Phase 1 — Snapshot:** After the agent session terminates, LLD-03 commits the agent container to a snapshot image via `docker commit`. Unlike `docker export`/`docker import`, `docker commit` preserves the container's runtime config — ENV, WORKDIR, USER, PATH, and other Dockerfile-level metadata — alongside its filesystem changes.

**Phase 2 — Functional checks (agent-controlled runtime, sandboxed):** LLD-03 launches a container FROM the committed snapshot image to run the test commands specified in the family spec's `grading_invariant.functional_checks` list. This container inherits the agent's full runtime config natively — `npm`, `gcc`, PATH, installed packages, and everything the agent modified are available at their original paths. The container runs with `--network none` and has no access to verifier scripts, milestone checks, or oracle solutions. Exit codes and stdout/stderr are captured to a shared volume. The container is removed after completion.

**Phase 3 — Integrity verification (trusted runtime):** LLD-03 launches a container FROM a **trusted grading base image** (`codex-long-grader:<pinned_digest>`, audited and pinned at Sprint 0b freeze). This image provides the verifier's interpreter and all integrity-check tools — `bash`, `jq`, `sha256sum`, `grep`, `coreutils` — from a source the agent cannot control. The agent's filesystem is extracted from the snapshot and mounted read-only at `/agent/`. Phase 2 results are mounted read-only at `/functional/`. Verifier scripts, verifier data, and milestone scripts are bind-mounted at `/verifier/` and `/verifier_data/`. The verifier script runs under the trusted image's `bash` and writes `verify_result.json` to `/results/`.

**Why this separation matters:** If the verifier ran inside the agent's rootfs (as v0.3 proposed), a buggy or adversarial agent could alter `/bin/bash`, shadow `sha256sum` to always return success, replace `grep` to hide stale imports, or tamper with `npm` to always exit 0. The shortcut-detection checksums would be useless because the checksum tool itself is compromised. By running integrity checks on a trusted substrate, the verifier's own execution cannot be subverted by the thing it is grading. Functional checks (Phase 2) deliberately use the agent's runtime — that is inherent to what we are testing — but the results are interpreted by the trusted Phase 3, not self-reported.

**Trusted invariant requirement (per family):** The three-phase split only works if Phase 3 has enough independent evidence to reject a spoofed Phase 2. A family where the verdict depends entirely on Phase 2 exit codes — with only weak or cosmetic Phase 3 checks — is vulnerable to an agent that shadows the test runner to always report success. Every family must have **at least one** `grading_invariant.expected_final_state` check that can independently reject a false Phase 2 pass using trusted tools only. Examples of trusted invariants: checksum comparison of test files (detects deletion/modification), grep for specific code patterns (detects fake migration), file-existence checks for expected build outputs, or version-string checks in lock files. A family that can only be graded by "did `npm test` exit 0?" with no independent filesystem check is not shortcut-resistant and must be redesigned. This is enforced in the §3.2 authoring checklist and tested in the Tier 2 adversarial audit (§11.2).

```bash
#!/bin/bash
# verifiers/dependency-migration-npm/verify.sh
#
# Runs in Phase 3: TRUSTED grading container (codex-long-grader image).
# bash, jq, sha256sum, grep are from the trusted image, NOT the agent's filesystem.
# Agent's filesystem is mounted read-only at /agent/.
# Phase 2 functional check results are at /functional/.

AGENT_WS="/agent/workspace"
RESULT_FILE="/results/verify_result.json"

# Initialize result
echo '{"pass": false, "milestones": {}, "errors": []}' > "$RESULT_FILE"

# ── Check 1: Dependency version (integrity check — trusted grep) ──
if grep -q '"lodash": "^4\.' "$AGENT_WS/package-lock.json" 2>/dev/null; then
    jq '.milestones.m1_dep_updated = true' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
else
    jq '.errors += ["Dependency not updated to target version"]' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
fi

# ── Check 2: API usage updated (integrity check — trusted grep) ──
STALE_IMPORTS=$(grep -rl "require('lodash/string')" "$AGENT_WS/src/" 2>/dev/null | wc -l)
if [ "$STALE_IMPORTS" -eq 0 ]; then
    jq '.milestones.m2_imports_fixed = true' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
else
    jq ".errors += [\"${STALE_IMPORTS} files still using removed API\"]" "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
fi

# ── Check 3: Test suite passes (read Phase 2 functional result) ──
if [ -f "/functional/npm_test_exit_code" ]; then
    EXIT_CODE=$(cat /functional/npm_test_exit_code)
    if [ "$EXIT_CODE" = "0" ]; then
        jq '.milestones.m3_tests_passing = true' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
    fi
else
    jq '.errors += ["Phase 2 functional check (npm test) did not produce results"]' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
fi

# ── Check 4: Shortcut detection — test files unmodified (trusted sha256sum) ──
if [ -f "/verifier_data/test_checksums.sha256" ]; then
    # Rewrite checksum file paths to point into /agent/ mount
    sed 's|  \./|  /agent/workspace/|' /verifier_data/test_checksums.sha256 > /tmp/checksums_remapped.sha256
    if ! sha256sum -c /tmp/checksums_remapped.sha256 --quiet 2>/dev/null; then
        jq '.errors += ["Test files were modified — shortcut detected"]' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
        jq '.pass = false | .milestones.m3_tests_passing = false' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
        cat "$RESULT_FILE"
        exit 0
    fi
fi

# ── Final verdict ──
ALL_MILESTONES=$(jq '[.milestones[]] | all' "$RESULT_FILE")
if [ "$ALL_MILESTONES" = "true" ]; then
    jq '.pass = true' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
fi

cat "$RESULT_FILE"
```

### 6.2 Verifier Output Schema

```json
{
  "pass": true,
  "milestones": {
    "m1_dep_updated": true,
    "m2_imports_fixed": true,
    "m3_tests_passing": true
  },
  "errors": [],
  "shortcut_detected": false,
  "wall_clock_seconds": 3.2
}
```

LLD-05 (Evaluator, Codex-Long path) consumes this schema. LLD-12 (Results) aggregates milestone-level partial credit for RL reward signals and binary pass/fail for headline solve rates.

### 6.3 Verifier Isolation — No Exceptions

The verifier script and all supporting data (reference checksums, expected outputs, test resource files) are stored in the `verifiers/` tree, which is:

- Never mounted into the agent container during a run
- Never included in any file visible to the agent (no AGENTS.md references, no repo-level README mentions)
- Injected only into the Phase 3 trusted grading container by LLD-03, after the agent session terminates

This is architecturally enforced, not policy-enforced. LLD-03's container launch must not bind-mount `verifiers/` into the agent container. If a verifier is found inside an agent container during an audit, the family is disqualified.

### 6.4 Three-Phase Grading — Execution Details

#### Phase 1 — Snapshot via `docker commit`

```bash
# After agent session terminates (LLD-03)
docker commit <agent_container> codex-long-snapshot/<run_id>
docker rm <agent_container>
```

`docker commit` creates a new image that includes the container's filesystem changes AND its runtime config (ENV, WORKDIR, USER, PATH). This is distinct from `docker export`/`docker import`, which exports filesystem only and discards runtime config unless manually reconstructed via `--change` flags. Phase 2 requires the committed config to be faithful so that `npm`, `gcc`, and other tools resolve correctly.

#### Phase 2 — Functional Checks (Sandboxed Agent Runtime)

```bash
# LLD-03 launches functional checks from the committed snapshot
docker run --rm \
  --network none \
  -v /grading/<run_id>/functional/:/functional/ \
  codex-long-snapshot/<run_id> \
  sh -c 'cd /workspace && npm test > /functional/npm_test_output.log 2>&1; echo $? > /functional/npm_test_exit_code'
```

This container runs inside the agent's actual runtime: `npm` is the one the agent installed, PATH is the agent's PATH, shared libraries are what the agent left behind. The container has no network (`--network none`) and no access to verifiers or grading artifacts. The only output is the functional check results written to `/functional/`.

The test commands to run are specified per family in the `grading_invariant.functional_checks` field of the family spec (§3.1). LLD-03 reads this field and constructs the Phase 2 `docker run` command accordingly.

**What if the agent tampered with `npm` or `sh`?** Phase 2 results are not self-authenticating. If the agent replaced `npm` with a script that always exits 0, Phase 2 reports success — but Phase 3's integrity checks (trusted `sha256sum` verifying test files are unmodified, trusted `grep` checking for actual API migration) catch the shortcut. The functional check result is one input to the verdict, not the verdict itself.

#### Phase 3 — Integrity Verification (Trusted Runtime)

```bash
# Extract agent filesystem for read-only inspection (driver-agnostic)
docker create --name agent_fs_<run_id> codex-long-snapshot/<run_id> true
mkdir -p /grading/<run_id>/agent_root
docker cp agent_fs_<run_id>:/. /grading/<run_id>/agent_root/
docker rm agent_fs_<run_id>

# LLD-03 launches integrity verifier from trusted grading image
docker run --rm \
  --network none \
  -v /grading/<run_id>/agent_root/:/agent/:ro \
  -v /grading/<run_id>/functional/:/functional/:ro \
  -v /path/to/verifiers/<family_id>/:/verifier/:ro \
  -v /path/to/verifier_data/<family_id>/:/verifier_data/:ro \
  -v /grading/<run_id>/results/:/results/ \
  codex-long-grader:<pinned_digest> \
  /verifier/verify.sh

# Cleanup
rm -rf /grading/<run_id>/agent_root
docker rmi codex-long-snapshot/<run_id>
```

> **Docker runtime requirement:** The tested configuration is rootful Docker with the overlay2 storage driver on the DGX Spark (ARM64 Linux). The `docker cp` extraction path above is driver-agnostic and works with any storage backend. Earlier drafts used `docker inspect --format={{.GraphDriver.Data.MergedDir}}` for direct overlay2 layer access — this was replaced because it is not portable across storage drivers and fails on rootless Docker or non-overlay backends.

The trusted grading image (`codex-long-grader:<pinned_digest>`) is built and audited during Sprint 0b. It contains: bash, jq, sha256sum, grep, coreutils, and any other tools verifier scripts require. Its digest is recorded in `benchmark_manifest.lock` and verified before every grading run.

**Why `/agent/` is read-only in Phase 3:** The integrity verifier inspects the agent's filesystem — it does not execute inside it. All file-content checks (checksums, greps, file existence) operate on the read-only mount. Functional execution (npm test) already happened in Phase 2. If a verifier needs to run additional functional checks beyond what Phase 2 captured, those must be added to the family spec's `functional_checks` list, not run ad-hoc in Phase 3.

**Why `--network none` in both phases:** §6.5 establishes that verifiers depend on filesystem state only. Network access during grading would introduce non-reproducible external dependencies and a potential exfiltration surface for agent-controlled code in Phase 2.

### 6.5 Runtime-State Boundary — Filesystem Only

The grading protocol captures persisted filesystem state via `docker commit`. The following are available in the committed image and accessible during grading:

- All filesystem changes (files created, modified, deleted by the agent)
- ENV, WORKDIR, USER, PATH as set in the original Dockerfile and any runtime modifications persisted via `docker commit`

The following are **not** captured and **not** available to the verifier:

- Live processes (running services, daemons, background jobs)
- Listening sockets or network state
- In-memory state (environment variables set via `export` in a shell session but not persisted to a dotfile, unless captured by `docker commit`'s config snapshot)
- Mounted volumes or bind-mount contents external to the container's own filesystem

**Hard rule:** Codex-Long verifiers may depend on persisted filesystem state only. Scenarios whose success criterion depends on a live process, a listening socket, external network reachability, or any other transient runtime state are out of scope unless a replay protocol is defined and added to this LLD. No such protocol exists in v0.6.

**Authoring implication:** If a scenario requires "the service is running on port 3000," the verifier cannot check that directly. It can check that the service binary is installed, that the config file is correct, and that a startup script exists — all of which are filesystem state. Scenario authors must design grading criteria around filesystem artifacts, not runtime behavior. If a scenario's natural success criterion is "the service responds to HTTP requests," the verifier should instead check the preconditions for that (correct binary, correct config, correct startup script, test suite that exercises the service passes via Phase 2 functional checks).

---

## 7. Post-Run Injection Protocol

This protocol governs the sequencing of verifier and test resource injection relative to the agent session. It matches Terminal-Bench's integrity model. LLD-03 implements the execution; this LLD defines the contract.

### 7.1 Execution Sequence

```
Phase 1a — Agent Session (LLD-03 manages)
┌─────────────────────────────────────────────────────┐
│  Agent container running                            │
│  - Agent sees: /workspace, AGENTS.md, tools         │
│  - Agent does NOT see: verifiers, milestones,        │
│    oracle solutions, test reference data             │
│  - Session terminates: codex exec completes or       │
│    hits timeout                                      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
Phase 1b — Snapshot (LLD-03)
┌─────────────────────────────────────────────────────┐
│  docker commit <agent_container>                     │
│    codex-long-snapshot/<run_id>                       │
│  Agent container removed after commit.               │
└─────────────────────────────────────────────────────┘
                         │
                  ┌──────┴──────┐
                  ▼             ▼
Phase 2 — Functional        Phase 3 — Integrity
(runs first)                (runs after Phase 2)
┌──────────────────────┐  ┌───────────────────────────┐
│ Container FROM       │  │ Container FROM trusted     │
│ committed snapshot   │  │ codex-long-grader:<digest> │
│ --network none       │  │ --network none             │
│                      │  │                            │
│ Runs: npm test (or   │  │ Bind-mounts:               │
│ other functional     │  │  /agent/       ← snapshot  │
│ checks from family   │  │                  fs (ro)   │
│ spec)                │  │  /functional/  ← Phase 2   │
│                      │  │                  results   │
│ Writes:              │  │  /verifier/    ← scripts   │
│  /functional/        │  │  /verifier_data/← ref data │
│  exit codes + logs   │  │  /results/     ← output    │
│                      │  │                            │
│ No verifier access.  │  │ Runs: /verifier/verify.sh  │
│ No milestone access. │  │ Trusted bash, jq, sha256sum│
└──────────────────────┘  │                            │
                          │ Writes: verify_result.json │
                          │ → LLD-05 → LLD-12          │
                          └───────────────────────────┘
```

### 7.2 Test Resource Injection Timing

Any test files referenced by the verifier (expected outputs, reference configurations, golden test data) are copied into the grading container **after** the agent session ends. The agent sees only the repository state as checked out at run start.

This prevents agents from reading test expectations and reverse-engineering solutions to match them. The verifier may reference files that do not exist in the agent's workspace — for example, a reference `package-lock.json` that the agent was expected to regenerate, or a golden output file that the agent was expected to produce from scratch.

### 7.3 AGENTS.md Contract

Each variant includes an `AGENTS.md` file in the repository root, visible to the agent at runtime. This file describes the task objective in plain language — what is broken, what needs to be fixed, and what "done" looks like from the agent's perspective. It does not reveal:

- Verifier implementation details or grading criteria
- Milestone definitions or partial credit weights
- Oracle solution approach or specific code changes expected
- File paths or names of verifier scripts

The AGENTS.md is the agent's only task description. It should be sufficient for a competent developer to understand the task without additional context.

---

## 8. Oracle Solution Isolation

Each family has an oracle solution — a known-good resolution of the injected breakage. Oracle solutions exist solely to prove that the task is solvable during authoring. They are never distributed, never included in training data, and never accessible to the agent.

### 8.1 Oracle Storage

```
scenario_families/<family_id>/
  oracle/
    solution.patch          # diff against the broken repo state
    solve_log.md            # authoring notes: approach, time, edge cases
    verify_confirmation.log # output of running the verifier against the oracle solve
```

The `oracle/` directory is:

- Excluded from all Docker image builds (via `.dockerignore` and build-context exclusion)
- Not committed to any repository branch that is distributed as part of the benchmark release
- Stored in a separate access-controlled directory (or git branch) that is never checked out during collection or evaluation runs

### 8.2 Oracle Validation During Authoring

Before a family enters the frozen split, the oracle must:

- [ ] Apply cleanly to the broken repo state for at least one variant
- [ ] Pass the verifier with `"pass": true` and all milestones satisfied
- [ ] Not trigger any shortcut detection checks
- [ ] Complete in a reasonable time (< 30 min for manual application — this is a sanity check on task complexity, not a performance target)

The oracle is a solvability proof. If the oracle cannot pass the verifier, the verifier or the task design is wrong.

---

## 9. Milestone Check Scripts

Milestones provide partial-credit signals for RL reward shaping (Phase 2b, stretch). They are also useful diagnostic metadata for understanding where agents get stuck.

### 9.1 Design Principles

- Milestones are **ordered by task progression**, not by difficulty. `m1` should be the first meaningful step; `mN` should be close to completion.
- Partial credit weights are monotonically non-decreasing: later milestones are worth at least as much as earlier ones. Reaching `m3` without `m2` should not score higher than reaching `m2` alone.
- A milestone check may report `achieved: true` even if the final verifier verdict is `pass: false` (the agent made progress but didn't finish). The verifier's binary pass/fail is the source of truth for solve rate; milestones provide a finer-grained signal.
- Milestone checks follow the same injection protocol as the verifier — Phase 3 only, never visible to the agent.

### 9.2 Execution Authority — `verify.sh` Is the Single Source of Truth

**`verify.sh` is the only script that LLD-03 executes during Phase 3.** It produces a single `verify_result.json` containing both the binary pass/fail verdict AND all milestone results. LLD-03 and LLD-05 consume this file and nothing else — they do not independently execute milestone scripts.

Milestone scripts are **helper functions called by `verify.sh`**, not standalone executables invoked by the orchestrator. This eliminates the ambiguity of having two execution paths for milestone results (verifier-internal vs. orchestrator-external) and ensures that milestones and the final verdict are always consistent — both computed from the same Phase 3 invocation, under the same trusted tools, against the same read-only agent filesystem snapshot.

### 9.3 Implementation Pattern

Milestone scripts are sourced by `verify.sh`:

```bash
#!/bin/bash
# verifiers/dependency-migration-npm/milestones/m1_dep_updated.sh
# Helper function sourced by verify.sh in Phase 3.
# Colocated under verifiers/ so the single Phase 3 bind-mount
# (verifiers/<family_id>/ → /verifier/) makes milestones available
# at /verifier/milestones/ with no additional mount.

check_m1_dep_updated() {
    local agent_ws="$1"
    if grep -q '"lodash": "^4\.' "$agent_ws/package.json" 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}
```

And `verify.sh` calls them:

```bash
# Inside verify.sh (Phase 3, trusted grading container)
source /verifier/milestones/m1_dep_updated.sh
source /verifier/milestones/m2_imports_fixed.sh

M1=$(check_m1_dep_updated "$AGENT_WS")
jq --arg v "$M1" '.milestones.m1_dep_updated = ($v == "true")' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"

M2=$(check_m2_imports_fixed "$AGENT_WS")
jq --arg v "$M2" '.milestones.m2_imports_fixed = ($v == "true")' "$RESULT_FILE" > tmp.json && mv tmp.json "$RESULT_FILE"
# ...
```

This pattern keeps milestone logic modular (one file per milestone, testable in isolation during authoring) while maintaining a single execution path through `verify.sh` at grading time.

---

## 10. Integrity Protocol

This section codifies all anti-cheat and isolation requirements in one place. These rules are enforced at the harness level (LLD-03, this LLD), not relied on via agent cooperation.

### 10.1 Verifier and Oracle Isolation

| Artifact | Location | Agent Access | Phase 2 Access | Phase 3 Access |
|---|---|---|---|---|
| Verifier scripts | `verifiers/<family_id>/` | **Never** | **Never** | Bind-mounted at `/verifier/:ro` |
| Milestone check scripts | `verifiers/<family_id>/milestones/` | **Never** | **Never** | Available at `/verifier/milestones/` via the same bind-mount as the verifier script — no separate mount needed |
| Oracle solutions | `oracle/` (access-controlled) | **Never** | **Never** | Not needed |
| Test reference data | `verifier_data/<family_id>/` | **Never** | **Never** | Bind-mounted at `/verifier_data/:ro` |
| AGENTS.md | `/workspace/AGENTS.md` | **Yes** | Present (in snapshot) | Present at `/agent/workspace/AGENTS.md` |
| Agent's filesystem | Container writable layer | **Yes** | **Yes** (native rootfs from committed image) | Read-only mount at `/agent/` |
| Trusted tools (bash, jq, sha256sum) | `codex-long-grader` image | N/A | **No** (agent's own tools used) | **Yes** (from trusted image) |

### 10.2 Test Resource Injection Timing

Test resources (reference checksums, expected output files, golden test data) are copied into the grading container **after** the agent session ends. The agent cannot access them at any point during execution. This prevents reverse-engineering from test expectations.

### 10.3 Shortcut Prevention

Scenarios must not be solvable by:

- Deleting failing tests — verifier must check that the full test suite is present and unmodified (checksum comparison)
- Disabling CI checks or test runners — verifier must run the test suite independently, not trust the agent's reported test results
- Mocking external state — verifier must check for actual implementation, not just passing tests
- Producing trivially wrong final state — verifier's expected-state checks must be specific enough to catch cosmetic or no-op changes

Each scenario's verifier is reviewed for shortcut resistance during Sprint 0b authoring. The adversarial audit (§11 Tier 2) explicitly tests these attack vectors.

### 10.4 Reproducibility

- Each variant is pinned to a specific Docker base image digest — not a floating tag
- **Verifier release timing:** Only Public-Dev verifiers are published pre-evaluation (alongside the Public-Dev data release). Test-Long, Val-Long, and Train-Long verifiers remain sealed until Sprint 3 evaluation is complete. The full verifier suite for all splits is released after Sprint 3 evaluation completes and results are final — never before, to prevent reverse-engineering of sealed-split grading criteria.
- Reproduction instructions allow independent grading of any submitted trajectory without access to oracle solutions
- The `build_hash` tagging scheme (§5.2) ensures identical inputs produce identical environments
- The `benchmark_manifest.lock` (§12.6) provides per-variant hash verification for full artifact-level reproducibility

---

## 11. Two-Tier Benchmark Audit Protocol

Both tiers must pass before the family-to-split assignment is frozen. No family enters the frozen split without passing its applicable audit tier.

### 11.1 Tier 1 — Type Coverage Audit

**Requirement:** At least one variant per scenario type (5 types minimum) fully audited end-to-end.

**Audit procedure per variant:**

1. Build the environment image from the Dockerfile
2. Apply the oracle solution to the broken repo state
3. Run the full three-phase grading pipeline (Phase 2 functional checks + Phase 3 `verify.sh`) against the oracle-solved state → `verify_result.json` must have `"pass": true` and all milestones `"achieved": true`
4. Attempt at least one known shortcut exploit (from the family's `shortcut_resistance.known_exploits_tested`) → re-run grading pipeline → verifier must return `"pass": false`
5. Verify that the agent container does not contain verifier scripts, milestone checks, or oracle solutions

**Pass criterion:** All five steps succeed for at least one variant in at least one family of each scenario type.

### 11.2 Tier 2 — Random Adversarial Audit

**Requirement:** 20% of all families (≥ 11 families at the full 55-family plan; ≥ 7 families at the 35-family plan) selected at random.

**Audit procedure per selected family:**

1. Select one variant at random from the family
2. Attempt to pass the verifier by each of these adversarial strategies:
   - Delete or modify failing test files → re-run verifier
   - Mock or stub the component under test → re-run verifier
   - Produce a trivially wrong final state (e.g., empty file, no-op change) → re-run verifier
   - Disable test runner exit-code checking → re-run verifier
   - **Spoofed functional success:** Shadow or replace the test runner binary (e.g., create a `/workspace/node_modules/.bin/npm` that always exits 0, or a wrapper script that intercepts `pytest`) so that Phase 2 reports success without the real test suite running. Then run the full three-phase grading pipeline. Phase 3's trusted invariants must still reject the spoof — if the verifier returns `pass: true` when the test runner was faked, the family fails the audit.
3. For each strategy: the verifier **must** return `"pass": false`
4. If any strategy produces `"pass": true` on a clearly wrong state → the family is **disqualified** and must be redesigned before freeze

**Pass criterion:** All adversarial strategies are caught by the verifier for every selected family. A human must attempt the exploit — this step cannot be delegated to LM alone.

### 11.3 Audit Log

All audit results are recorded in a structured log:

```yaml
# audit_log/<family_id>.yaml
family_id: dependency-migration-npm
audit_tier: 2
audit_date: 2026-05-15
auditor: <name>
variant_audited: lodash-3-to-4

tier1_results:
  oracle_solve: pass
  verifier_on_oracle: pass
  all_milestones: pass
  shortcut_exploit_caught: pass
  container_clean: pass

tier2_results:
  delete_tests: caught
  mock_dependency: caught
  trivial_state: caught
  disable_exit_code: caught
  spoofed_functional_success: caught

verdict: PASS
notes: >
  All adversarial strategies caught. Verifier checksum comparison
  detected test file deletion. API usage grep caught mock-based bypass.
  Spoofed npm (wrapper script exiting 0) was caught by Phase 3 trusted
  grep detecting stale lodash/string imports despite Phase 2 reporting
  test success.
```

---

## 12. Family-to-Split Assignment and Freeze Procedure

### 12.1 Assignment Protocol

Families are assigned to Train-Long / Val-Long / Test-Long / Public-Dev **before any collection run**. Assignment is frozen and published alongside results. No post-hoc reassignment based on solve rates.

**Assignment constraints:**

- Each split must contain at least one family from each of the five scenario types (§4), **subject to the Public-Dev carve-out below**. For Train-Long, Val-Long, and Test-Long this is a hard requirement.
- Public-Dev should cover all five types (HLD requirement). At the full 55-family plan (~5 Public-Dev families), this is achievable. On the 35-family path (~2 Public-Dev families), **full 5-type coverage is structurally impossible** — 2 families cannot span 5 types. In this case, Public-Dev covers as many types as family count allows, and the limitation is documented in the data release notes. **Recommended HLD errata:** the HLD v2.3 simultaneously requires Public-Dev to cover all five types AND pre-declares ~2 Public-Dev families on the 35-family path. These are contradictory. This LLD faithfully implements the pre-declared geometry and documents the gap rather than overriding either HLD rule.
- Test-Long and Val-Long must contain families **not seen in Train-Long** — not merely unseen variants
- Assignment is randomized within constraints, using a fixed seed for reproducibility

### 12.2 Freeze Procedure

```
1. Sprint 0b authoring completes → all family specs pass §3.2 checklist
2. Two-tier audit completes → all families pass applicable tier (§11)
3. Difficulty pre-screening → each family manually estimated in 20–80% solve range
4. Family-to-split assignment generated (randomized within constraints, fixed seed)
5. Assignment table published as split_assignment.yaml
6. FREEZE: No families added, removed, or reassigned after this point
7. Rule 1 check (§12.4): count Test-Long families → if < 8, B1 is dropped
```

### 12.3 Assignment Table Format

```yaml
# split_assignment.yaml (frozen — do not modify after freeze)
freeze_date: 2026-06-01
seed: 42
total_families: 55

splits:
  train_long:
    families:
      - family_id: dependency-migration-npm
        scenario_type: migration_refactor
        variant_count: 6
      - family_id: add-rest-endpoint
        scenario_type: feature_evolution
        variant_count: 5
      # ... ~28 more families
    summary:
      total_families: 30
      total_envs: 187
      type_coverage: [feature_evolution: 7, migration_refactor: 7, build_ci_breakage: 5, investigate_then_fix: 6, cross_layer_changes: 5]

  val_long:
    families: [...]
    summary: { total_families: 10, total_envs: 38, ... }

  test_long:
    families: [...]
    summary: { total_families: 10, total_envs: 47, ... }

  public_dev:
    families: [...]
    summary: { total_families: 5, total_envs: 17, ... }
```

### 12.4 Smaller-v1 Escape Hatch

If Sprint 0b cannot produce 55+ high-quality families within the 4–6 week budget, **freeze a smaller, higher-quality set rather than rushing to hit the count**. The HLD pre-registers the 35-family path as the minimum viable release.

**Pre-declared split geometry for the 35-family path:**

| Split | Families | Envs/Family | Total Envs | Notes |
|---|---|---|---|---|
| Train-Long | ~20 | ~5–8 | ~100–160 | Primary SFT/RL trajectories |
| Val-Long | ~7 | ~3–5 | ~21–35 | RL early stopping only |
| Test-Long | ~6 | ~4–6 | ~24–36 | Sealed secondary benchmark |
| Public-Dev | ~2 | ~3–4 | ~6–8 | Reproducibility release (5-type coverage not achievable — see §12.1 carve-out) |

This geometry is taken verbatim from HLD v2.3 §6 Sprint 0b. This LLD does not modify it.

**Rule 1 — Hard Test-Long family floor (applied at freeze, before Gate 4):**

If the frozen Test-Long split has fewer than 8 families, B1 is dropped automatically. This rule fires regardless of Gate 4 outcome. Gate 4 thresholds mostly concern Train-Long matched yield — they do not protect against a structurally underpowered Test-Long. With fewer than 8 Test-Long families, family-clustered bootstrap produces intervals too wide to support even a directional harness-specificity claim.

The 35-family path (Test-Long ~6 families) triggers Rule 1: **B1 is dropped on the 35-family path regardless of matched-ID count.** The 35-family path ships Contribution A + B2 only, subject to the B2-only proceed rule below.

**Rule 2 — Gate 4 threshold check (applied after Gate 4 pilot):**

- If Rule 1 does not fire AND Gate 4 thresholds are met → full B1 + B2 survive.
- If Rule 1 does not fire AND Gate 4 thresholds are not met → B1 is dropped, B2 survives.

**Pre-registered B2-only proceed rule for the 35-family path:**

After Gate 4 pilot completes on the 35-family plan:

- **PROCEED (B2 survives):** Projected Codex traces ≥ 50 AND projected Train-Long collection wall-clock ≤ 25 Spark days.
- **KILL (Contribution B dropped):** Projected Codex traces < 50 OR projected wall-clock > 25 Spark days → project reduces to Contribution A only.

These thresholds are pre-registered in the HLD and cannot be renegotiated post-Sprint-0b. 50 traces is the stated minimum for a meaningful Codex-SFT-all claim. 25 Spark days is the scaled ceiling for ~100–160 envs at the 35-family Train-Long size.

### 12.5 Post-Freeze Rules

After the freeze:

- No families may be added, removed, or moved between splits
- No variants may be added to Test-Long or Val-Long families (sealed)
- Variants may be added to Train-Long families only if Gate 4 pilot recommends expanding variant count for low-yield families (ADJUST outcome) — and only before full collection begins. Each addition triggers a manifest version bump (§12.6).
- Verifier logic may be bug-fixed but not structurally changed (shortcut resistance must not be weakened). Each bug-fix triggers a manifest version bump and a re-hash of the affected verifier (§12.6). Any eval runs completed against the pre-fix verifier must be re-evaluated or noted with the manifest version they ran against.

### 12.6 Benchmark Manifest Lock

The freeze is not complete until a `benchmark_manifest.lock` is generated and committed. This file records a per-variant immutable fingerprint that makes post-freeze drift auditable. Any change to a locked artifact requires a manifest version bump and a change note — silent changes are structurally impossible because hash mismatches are caught at run-start by LLD-03.

**Manifest schema:**

```yaml
# benchmark_manifest.lock (generated at freeze, version-bumped on any change)
manifest_version: 1
freeze_date: 2026-06-01
generator: "scripts/generate_manifest.sh v1.0"
split_assignment_hash: "sha256:aaa111..."  # hash of split_assignment.yaml itself
grader_image_digest: "sha256:hhh888..."    # pinned digest of codex-long-grader trusted image

variants:
  - family_id: dependency-migration-npm
    variant_id: lodash-3-to-4
    split: train_long
    scenario_type: migration_refactor
    family_spec_hash: "sha256:iii999..."   # sha256 of scenario_families/<family_id>/family.yaml — covers functional_checks, grading_invariant, breakage_class, milestones, shortcut_resistance, and all other grading-relevant config
    image_digest: "sha256:abc123..."       # docker image content-addressable digest
    verifier_hash: "sha256:bbb222..."      # sha256 of verifiers/dependency-migration-npm/verify.sh
    milestone_hashes:                       # sha256 of each milestone check script
      m1_dep_updated: "sha256:ccc333..."
      m2_imports_fixed: "sha256:ddd444..."
      m3_tests_passing: "sha256:eee555..."
    agents_md_hash: "sha256:fff666..."     # sha256 of the AGENTS.md inside the built image
    verifier_data_hash: "sha256:ggg777..."  # sha256 of the verifier_data/ tree (reference checksums, golden files)
  # ... one entry per variant across all splits

change_log: []
# After a post-freeze change (e.g., Train-Long variant addition or verifier bug-fix):
# change_log:
#   - manifest_version: 2
#     date: 2026-06-10
#     change: "Added variant 'express-4-to-5' to family 'dependency-migration-npm' (Train-Long)"
#     reason: "Gate 4 ADJUST — expanding low-yield family"
#     affected_variants: ["dependency-migration-npm/express-4-to-5"]
#     affected_hashes: ["image_digest", "agents_md_hash"]
#     re_gate_required: false  # new variant, no prior runs affected
#   - manifest_version: 3
#     date: 2026-06-15
#     change: "Bug-fix in verify.sh for family 'ci-toolchain-gcc'"
#     reason: "Verifier false-negative on valid gcc-13 install path"
#     affected_variants: ["ci-toolchain-gcc/gcc-12-to-13", "ci-toolchain-gcc/gcc-11-to-14"]
#     affected_hashes: ["verifier_hash"]
#     re_gate_required: true   # any runs graded with manifest_version < 3 must be re-evaluated
```

**Enforcement contract (LLD-03 responsibility) — two-phase hash verification:**

**Phase 1 — Pre-run (before launching the agent container):**

1. Read `benchmark_manifest.lock`
2. Verify that the Docker image digest for the target variant matches the manifest's `image_digest`
3. Verify that the `agents_md_hash` matches (the AGENTS.md baked into the image is the task description the agent sees — drift here changes the task)
4. Verify that the `family_spec_hash` matches (the family YAML defines `functional_checks` commands, timeouts, grading invariant structure, and shortcut resistance notes — drift here changes Phase 2 behavior and grading semantics)
5. Record the `manifest_version` in the run's metadata (trajectory header)
6. If any hash mismatch is detected: **abort the run** and log the mismatch. Do not silently proceed with a drifted artifact.

**Phase 2 — Pre-grading (after agent session terminates, before launching Phase 2 or Phase 3 containers):**

7. Verify that the `grader_image_digest` matches the manifest — the `codex-long-grader` image is the root of trust for Phase 3 integrity checks; a mismatched grader image invalidates all integrity verification
8. Verify that the verifier script hash matches the manifest's `verifier_hash`
9. Verify that each milestone check script hash matches the manifest's `milestone_hashes`
10. Verify that the verifier_data tree hash matches the manifest's `verifier_data_hash`
11. If any hash mismatch is detected: **abort grading** for this run. The agent session output is preserved (trajectory is valid), but grading is blocked until the mismatch is resolved. Log the mismatch with the manifest version and the actual hash.

The two-phase split matters because the grading artifacts (verifier, milestones, verifier_data) are not part of the agent image — they are bind-mounted into the grading container. A pre-run check on image digest does not cover them. Without Phase 2, milestone scripts and verifier_data can drift silently after freeze, affecting reward semantics (milestones) or shortcut detection (verifier_data checksums) with no audit trail.

This makes the benchmark reproducible at the artifact level — any published result can be traced back to the exact manifest version and the exact hashes of every artifact involved. The claim that silent changes are structurally impossible is justified only when both phases are enforced.

---

## 13. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-01** vLLM Serving Layer | No direct interface. LLD-13 defines the scenario environments that LLD-03 launches against LLD-01's serving endpoint. LLD-01 serves identically regardless of task type. |
| **LLD-02** Data Pool Manager | Consumes the frozen `split_assignment.yaml` from this LLD. Tracks family-level metadata (scenario type, variant list, split assignment) and enforces Train-Long-only access for LLD-10. |
| **LLD-03** Task Orchestrator | Primary consumer. Launches agent containers from the Docker env factory images (§5). After agent session terminates, commits the agent container via `docker commit` (§6.4 Phase 1). Orchestrates Phase 2 (functional checks in sandboxed snapshot container with `--network none`) and Phase 3 (integrity verification in trusted `codex-long-grader` container with agent filesystem mounted read-only). Enforces `benchmark_manifest.lock` two-phase hash checks (§12.6). |
| **LLD-05** Evaluator (Codex-Long path) | Consumes `verify_result.json` produced by `verify.sh` in Phase 3 (§6.2, §9.2). Reads binary pass/fail and milestone partial-credit fields from this single file. Does not execute verifier or milestone scripts independently — `verify.sh` is the sole execution authority (§9.2). |
| **LLD-06** Trajectory Parser | Consumes family and scenario-ID metadata from this LLD for matched-ID splitting. The `family_id` and `variant_id` fields in the split assignment table are the join keys for matched-ID logic. |
| **LLD-07** Benchmark Runner | Pulls Codex-Long task lists (env image tags by split) from this LLD's registry. Manages the Gate 4 pilot sub-campaign using pilot families defined here. |
| **LLD-09** mini-SWE-Agent | Operates on Codex-Long scenarios for the SWE-Agent data collection arm and B1/B2 evaluation. Launches agent containers from the same Docker env factory images and invokes the same verifiers. |
| **LLD-10** SFT Training Pipeline | Indirect — LLD-10 trains on traces from scenarios defined here, but does not interact with this LLD directly. Family-level metadata flows through LLD-06. |
| **LLD-11** DAPO RL Pipeline | Indirect — reward signals come from this LLD's verifiers (via LLD-05), but LLD-11 does not call verifiers directly. |
| **LLD-12** Results & Artifact Generator | Indirect — aggregates solve rates and milestone data from LLD-05, which grades against this LLD's verifiers. Public-Dev split packaging for the data release is sourced from this LLD's frozen split table. |

---

## 14. Gate 3 Proto-Scenario Exception

The HLD requires one Codex-Long pilot scenario through Codex as part of Gate 3 (RL target feasibility, Sprint 0). This proto-scenario is authored before Sprint 0b begins.

### 14.1 What the Proto-Scenario Requires

- A representative Codex-Long-style scenario: multi-turn, tool-heavy, requiring code changes across multiple files
- Sufficient to measure Qwen3.5-27B rollout wall-clock through Codex
- A basic verifier (does not need shortcut-resistance review, adversarial audit, or milestone checks)
- A Docker environment that the agent can be launched into

### 14.2 What the Proto-Scenario Does NOT Require

- The full family spec format (§3) — simplified YAML or ad-hoc setup is acceptable
- Family-to-split assignment — it is not assigned to any split
- The post-run injection protocol — verifier can be run manually
- The two-tier audit protocol
- Multiple variants — a single scenario is sufficient
- Compliance with the integrity protocol's full rigor (no shortcut-resistance review needed)

### 14.3 Relationship to Sprint 0b

The proto-scenario is a throwaway artifact for Gate 3 measurement. It may become the seed for a Sprint 0b family (upgraded to full spec compliance) or may be discarded entirely. It does not constrain Sprint 0b design decisions.

**Gate 3 KILL threshold (from HLD):** 27B rollout > 150 min → project scope reduces to Contribution A only. The proto-scenario must be representative enough for this measurement to be meaningful.

---

## 15. Sprint 0b Validation Checklist

### Scenario Authoring

- [ ] ≥ 55 family specs written (or ≥ 35 for smaller-v1 path), each satisfying §3.2 checklist
- [ ] All five scenario types represented with ≥ 6 families each (or ≥ 3 for 35-family path)
- [ ] Each family has ≥ 3 variants with distinct variant IDs and Dockerfiles
- [ ] All derived variants (`repo_source: derived:*`) have complete provenance fields: `source_repo`, `license`, `redistribution_ok`, `modification_notice`. No derived variant with `redistribution_ok: false` appears in Public-Dev.

### Docker Environment Factory

- [ ] All variant Dockerfiles build successfully
- [ ] Build-time smoke test (injected breakage confirmed real) passes for every variant
- [ ] No verifier/oracle/milestone artifacts present inside any built container image
- [ ] Image tags follow the `codex-long/<family_id>/<variant_id>:<build_hash>` scheme

### Verifier Suite

- [ ] Every family has a verifier script that produces the §6.2 JSON schema
- [ ] Every family has ≥ 1 milestone check script
- [ ] Every family spec has ≥ 1 `functional_checks` entry with command and timeout
- [ ] Oracle solution passes verifier with `"pass": true` for at least one variant per family
- [ ] Trusted grading image (`codex-long-grader`) built, audited, and digest pinned: contains bash, jq, sha256sum, grep, coreutils

### Audit

- [ ] Tier 1 complete: ≥ 1 family per scenario type fully audited end-to-end (§11.1)
- [ ] Tier 2 complete: ≥ 20% of families (≥ 11 at 55 families; ≥ 7 at 35 families) adversarially audited by a human (§11.2)
- [ ] No family has a verifier that passes on a clearly wrong state
- [ ] Audit log written for every audited family (§11.3)

### Freeze

- [ ] Family-to-split assignment generated with fixed seed
- [ ] `split_assignment.yaml` published and committed
- [ ] `benchmark_manifest.lock` generated (§12.6): per-variant hashes for family spec (`family_spec_hash`), image digest, verifier script, milestone scripts, AGENTS.md, and verifier_data recorded; `grader_image_digest` for the trusted grading image pinned; all committed
- [ ] Rule 1 evaluated: Test-Long family count ≥ 8? If not, B1 is dropped (35-family path triggers this automatically).
- [ ] All five scenario types represented in Train-Long, Val-Long, and Test-Long. Public-Dev: 5-type coverage if family count allows; otherwise document which types are missing (see §12.1 carve-out).
- [ ] Post-freeze rules (§12.5) communicated to all contributors

### Difficulty Pre-Screening

- [ ] Each family's estimated solve rate is in the 20–80% range
- [ ] Families outside this range flagged for redesign before freeze

---

## 16. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **Authoring velocity — 55 families in 4–6 weeks is aggressive** | HIGH | Smaller-v1 escape hatch (§12.4) is pre-registered. Freeze a smaller, higher-quality set rather than rushing. LM-assisted authoring for boilerplate; human review for integrity. |
| **Verifier shortcut resistance is hard to guarantee** | HIGH | Two-tier audit protocol (§11). Tier 2 adversarial audit must be performed by a human, not delegated to LM. Families that fail are redesigned before freeze. |
| **Difficulty calibration is imprecise before Gate 4** | MEDIUM | Manual estimates in §3.2 are placeholders. Gate 4 pilot (5 families) measures actual solve rates. Families outside 20–80% range in Gate 4 are flagged for redesign or exclusion from RL training (retained for SFT if solve rate > 0). |
| **Docker image size bloat** | MEDIUM | Build validation (§5.4) flags images > 5 GB. Keep base images minimal. Avoid installing unnecessary development tools. |
| **Variant diversity within a family may be too low** | MEDIUM | Variants that differ only in trivial ways (variable names, comment text) do not add structural diversity. Review each variant for meaningful breakage differences during authoring. |
| **Oracle solution may not exist for all variants** | MEDIUM | Oracle validation (§8.2) is required before freeze. If an oracle cannot be produced for a variant, the variant is removed (not the family, unless all variants fail). |
| **Post-run grading adds latency via commit + two-phase execution** | LOW | Phase 1 (`docker commit`) is fast (< 5s typically). Phase 2 (functional checks) wall-clock depends on test suite runtime — typically < 60s. Phase 3 (integrity verification) is fast (< 30s). Total grading overhead: typically < 2 min per run. Flag slow Phase 2 runs during Gate 4 pilot. LLD-03 can parallelize Phase 2+3 across completed runs. |
| **Trusted grading image must be built and audited** | MEDIUM | The `codex-long-grader` image is a new Sprint 0b deliverable. It must contain bash, jq, sha256sum, grep, coreutils. Its digest is pinned in `benchmark_manifest.lock`. Building it is trivial (small Dockerfile); auditing it is a one-time task. If the image is compromised, all integrity checks are compromised — treat it as a root-of-trust artifact. |
| **Derived-repo licensing blocks publication** | MEDIUM | Provenance fields (§3.1) are required for all derived variants. Variants with `redistribution_ok: false` are restricted to non-published splits (Train-Long, Val-Long). Public-Dev and the open-source data release must use only `authored` or `redistribution_ok: true` variants. Sprint 0b authoring must include a licensing review pass before freeze. |
| **Test-Long family count too low for B1 (35-family path)** | LOW (pre-registered) | Rule 1 fires automatically at the HLD's 8-family floor. B1 is dropped; B2 survives subject to the B2-only proceed rule. This is a planned degradation path, not a failure. |
| **Public-Dev 5-type coverage impossible on 35-family path** | LOW (documented) | HLD v2.3 simultaneously requires 5-type Public-Dev coverage and pre-declares ~2 Public-Dev families. This LLD faithfully implements the geometry and documents the gap (§12.1 carve-out). Recommended HLD errata filed — not overridden locally. |

---

## 17. Open Questions — Status

| Question | Status |
|---|---|
| Final family count (55 vs 35 vs intermediate) | **OPEN — Sprint 0b authoring determines. Smaller-v1 escape hatch pre-registered for ≥ 35.** |
| Specific repository sources for variants | **OPEN — Authored repos (synthetic) vs derived from real open-source projects. Both are valid; authored repos allow tighter control over breakage injection and verifier design. Decision per family during Sprint 0b.** |
| Verifier execution environment | **Resolved (§6.1, §6.4):** Three-phase model. Functional checks (Phase 2) run FROM the committed snapshot — agent's own toolchain and runtime. Integrity checks (Phase 3) run FROM a trusted `codex-long-grader` image with agent filesystem mounted read-only at `/agent/`. Verifier interpreter and tools (bash, jq, sha256sum) come from the trusted image, not the agent's rootfs. |
| LM-assisted authoring scope | **OPEN — Which authoring steps are LM-delegated vs human-only? Tier 2 adversarial audit is human-only (HLD requirement). Variant generation, boilerplate verifier logic, and milestone check scaffolding are LM-delegable. Detailed scoping in Sprint 0b.** |
| Milestone partial credit weighting scheme | **OPEN — Linear vs exponential? Fixed per family or normalized across families? Affects Phase 2b RL reward shaping only (stretch). Default to linear, family-specific weights. Revisit if Phase 2b becomes live (Gate 5 pass).** |
| Cross-platform reproducibility of Docker environments | **OPEN — Built and tested on ARM64 (Spark). x86 reproducibility for external users depends on multi-arch base images. Public-Dev release should include x86-compatible Dockerfiles or note the limitation.** |

---

*LLD-13 · Codex-Long Scenario Framework · Signed Off v0.6 · April 2026*
