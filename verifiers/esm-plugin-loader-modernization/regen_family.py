#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "esm-plugin-loader-modernization"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_ROOT = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = VERIFIER_ROOT / "score_esm_loader.py"
MATRIX = VERIFIER_ROOT / "run_verification_matrix.py"
CODEX_CONFIG = FAMILY / "codex" / "config.toml"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
ALLOWED_WRITE_PATHS = [
    "src/loader.mjs",
    "src/contracts.mjs",
    "src/index.mjs",
    "scripts/build.mjs",
    "scripts/typecheck.mjs",
    "docs/cli/plugins.md",
    "artifacts/cli_capture.txt",
]
IMMUTABLE_TREE_RELS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "package.json",
    "artifacts/README.md",
    "plugins",
    "scripts/capture_cli_output.sh",
    "repo_evidence",
    "release_context",
    "incident_context",
    "tests",
]
ORACLE_MUTABLE_FILES = [
    "src/loader.mjs",
    "src/contracts.mjs",
    "src/index.mjs",
    "scripts/typecheck.mjs",
    "docs/cli/plugins.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(0o755)


def write_json(path: Path, payload: object) -> None:
    write(path, json.dumps(payload, indent=2, sort_keys=True))


def replace_tree(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def common_agents() -> str:
    return textwrap.dedent(
        """
        # AGENTS

        Repair the plugin loader as a real ESM migration, not a docs-only cleanup.

        Constraints:
        - Keep mixed-module helpers working.
        - Regenerate CLI evidence from the provided capture script.
        - Do not edit tests, plugins, repo evidence, release context, or incident context.
        - The built `dist/` output must be runnable on its own.
        """
    )


def dockerfile() -> str:
    return textwrap.dedent(
        """
        FROM node:20-bullseye
        WORKDIR /workspace
        """
    )


def base_workspace_files(variant_id: str) -> dict[str, str]:
    return {
        ".scenario_variant": variant_id,
        "AGENTS.md": common_agents(),
        "Dockerfile": dockerfile(),
        "package.json": json.dumps(
            {
                "name": "ops-report",
                "type": "module",
                "scripts": {
                    "test": "node --test tests/*.mjs",
                    "build": "node scripts/build.mjs",
                    "typecheck": "node scripts/typecheck.mjs",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        "artifacts/README.md": textwrap.dedent(
            """
            # CLI capture artifacts

            `artifacts/cli_capture.txt` must be regenerated from `bash scripts/capture_cli_output.sh`.
            Do not hand-edit generated evidence.
            """
        ),
        "docs/cli/plugins.md": textwrap.dedent(
            """
            # Plugins

            The loader still documents the legacy CommonJS path and the source-tree entrypoint.
            """
        ),
        "plugins/good-default.mjs": textwrap.dedent(
            """
            export default {
              name: "good-default",
              run(input) {
                return `default:${input}`;
              }
            };
            """
        ),
        "plugins/good-named.mjs": textwrap.dedent(
            """
            export const plugin = {
              name: "good-named",
              run(input) {
                return `named:${input}`;
              }
            };
            """
        ),
        "plugins/good-helper.mjs": textwrap.dedent(
            """
            import helper from "./helper.cjs";

            export const plugin = {
              name: "good-helper",
              run(input) {
                return `${helper.renderLabel("good-helper")}:${input}`;
              }
            };
            """
        ),
        "plugins/bad-wrong-shape.mjs": textwrap.dedent(
            """
            export default {
              name: "bad-wrong-shape"
            };
            """
        ),
        "plugins/helper.cjs": textwrap.dedent(
            """
            module.exports = {
              renderLabel(name) {
                return `helper:${name}`;
              }
            };
            """
        ),
        "repo_evidence/loader_contract.md": textwrap.dedent(
            """
            # Loader contract

            The modernized loader should resolve emitted plugins from `dist/plugins/`, accept both default-export and named-export plugin modules, and reject malformed modules with a stable runtime error.
            """
        ),
        "repo_evidence/build_notes.md": textwrap.dedent(
            """
            # Build notes

            The package is ESM-first. Build output is a copied `dist/` tree used by release smoke checks.
            """
        ),
        "scripts/build.mjs": textwrap.dedent(
            """
            import { cpSync, mkdirSync, rmSync } from "node:fs";

            rmSync(new URL("../dist", import.meta.url), { recursive: true, force: true });
            mkdirSync(new URL("../dist/src", import.meta.url), { recursive: true });
            mkdirSync(new URL("../dist/plugins", import.meta.url), { recursive: true });
            cpSync(new URL("../src", import.meta.url), new URL("../dist/src", import.meta.url), { recursive: true });
            cpSync(new URL("../plugins", import.meta.url), new URL("../dist/plugins", import.meta.url), { recursive: true });
            """
        ),
        "scripts/typecheck.mjs": textwrap.dedent(
            """
            import { readFileSync } from "node:fs";

            const loader = readFileSync(new URL("../src/loader.mjs", import.meta.url), "utf8");
            const contracts = readFileSync(new URL("../src/contracts.mjs", import.meta.url), "utf8");

            const checks = [
              loader.includes("resolvePluginUrl"),
              loader.includes("assertPluginContract"),
              contracts.includes("assertPluginContract")
            ];

            if (checks.some((value) => !value)) {
              process.exit(1);
            }
            """
        ),
        "scripts/capture_cli_output.sh": textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail

            npm run build >/dev/null
            mkdir -p artifacts

            {
              echo '$ node dist/src/index.mjs --help'
              node dist/src/index.mjs --help
              echo
              echo '$ node dist/src/index.mjs good-default'
              node dist/src/index.mjs good-default
              echo
              echo '$ node dist/src/index.mjs good-named'
              node dist/src/index.mjs good-named
              echo
              echo '$ node dist/src/index.mjs --list'
              node dist/src/index.mjs --list
            } > artifacts/cli_capture.txt
            """
        ),
        "src/contracts.mjs": textwrap.dedent(
            """
            export function isPlugin(value) {
              return Boolean(value) && typeof value.run === "function";
            }
            """
        ),
        "src/loader.mjs": textwrap.dedent(
            """
            export function listPluginNames() {
              return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
            }

            export async function loadPluginModule(name) {
              const mod = await import(new URL(`../plugins/${name}.js`, import.meta.url));
              return mod.default ?? mod;
            }
            """
        ),
        "src/index.mjs": textwrap.dedent(
            """
            import { listPluginNames, loadPluginModule } from "./loader.mjs";

            const arg = process.argv[2];

            if (!arg || arg === "--help") {
              console.log([
                "Usage: node src/index.mjs <plugin-name>",
                "Commands:",
                "  --list   Print discoverable plugin names"
              ].join("\\n"));
              process.exit(0);
            }

            if (arg === "--list") {
              console.log(listPluginNames().join("\\n"));
              process.exit(0);
            }

            const plugin = await loadPluginModule(arg);
            console.log(plugin.run("report"));
            """
        ),
        "tests/test_loader.mjs": textwrap.dedent(
            """
            import test from "node:test";
            import assert from "node:assert/strict";
            import { loadPluginModule } from "../src/loader.mjs";

            test("loads default export plugin", async () => {
              const plugin = await loadPluginModule("good-default");
              assert.equal(plugin.run("report"), "default:report");
            });

            test("loads named export plugin", async () => {
              const plugin = await loadPluginModule("good-named");
              assert.equal(plugin.run("report"), "named:report");
            });
            """
        ),
        "tests/test_cli_help.mjs": textwrap.dedent(
            """
            import test from "node:test";
            import assert from "node:assert/strict";
            import { execFile } from "node:child_process";
            import { promisify } from "node:util";

            const execFileAsync = promisify(execFile);

            test("prints help text", async () => {
              const { stdout } = await execFileAsync("node", ["src/index.mjs", "--help"]);
              assert.match(stdout, /Usage:/);
              assert.match(stdout, /--list/);
            });
            """
        ),
    }


def oracle_mutations() -> dict[str, str]:
    return {
        "src/contracts.mjs": textwrap.dedent(
            """
            export function isPlugin(value) {
              return Boolean(value)
                && typeof value === "object"
                && typeof value.name === "string"
                && typeof value.run === "function";
            }

            export function assertPluginContract(name, value) {
              if (!isPlugin(value)) {
                throw new Error(`Invalid plugin module: ${name}`);
              }
              return value;
            }
            """
        ),
        "src/loader.mjs": textwrap.dedent(
            """
            import { assertPluginContract } from "./contracts.mjs";

            export function listPluginNames() {
              return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
            }

            export function resolvePluginUrl(name) {
              return new URL(`../plugins/${name}.mjs`, import.meta.url);
            }

            export async function loadPluginModule(name) {
              const mod = await import(resolvePluginUrl(name).href);
              if ("default" in mod && mod.default !== undefined) {
                return assertPluginContract(name, mod.default);
              }
              if ("plugin" in mod) {
                return assertPluginContract(name, mod.plugin);
              }
              throw new Error(`Invalid plugin module: ${name}`);
            }
            """
        ),
        "src/index.mjs": textwrap.dedent(
            """
            import { listPluginNames, loadPluginModule } from "./loader.mjs";

            const arg = process.argv[2];

            if (!arg || arg === "--help") {
              console.log([
                "Usage: node dist/src/index.mjs <plugin-name>",
                "Commands:",
                "  --list   Print discoverable plugin names",
                "Notes:",
                "  - Supports default-export and named-export plugins",
                "  - Rejects malformed plugin modules with Invalid plugin module"
              ].join("\\n"));
              process.exit(0);
            }

            if (arg === "--list") {
              console.log(listPluginNames().join("\\n"));
              process.exit(0);
            }

            const plugin = await loadPluginModule(arg);
            console.log(plugin.run("report"));
            """
        ),
        "scripts/typecheck.mjs": textwrap.dedent(
            """
            import { readFileSync } from "node:fs";

            const loader = readFileSync(new URL("../src/loader.mjs", import.meta.url), "utf8");
            const contracts = readFileSync(new URL("../src/contracts.mjs", import.meta.url), "utf8");
            const index = readFileSync(new URL("../src/index.mjs", import.meta.url), "utf8");

            const checks = [
              loader.includes("resolvePluginUrl"),
              loader.includes("../plugins/${name}.mjs"),
              loader.includes("assertPluginContract"),
              contracts.includes("assertPluginContract"),
              index.includes("dist/src/index.mjs")
            ];

            if (checks.some((value) => !value)) {
              process.exit(1);
            }
            """
        ),
        "docs/cli/plugins.md": textwrap.dedent(
            """
            # Plugins

            Built CLI usage comes from `node dist/src/index.mjs`.

            The modernized loader resolves plugins from emitted `dist/plugins/`.

            Supported module shapes:

            - default-export plugin object
            - named `plugin` export

            Malformed modules fail with `Invalid plugin module: <name>`.

            Regenerate `artifacts/cli_capture.txt` with `bash scripts/capture_cli_output.sh`.
            """
        ),
    }


def variant_additions(variant_id: str) -> dict[str, str]:
    additions: dict[str, str] = {}
    if variant_id == "v2-noisy-distractor":
        additions["repo_evidence/legacy/2025_cjs_archive.md"] = textwrap.dedent(
            """
            # Archived migration note (2025)

            This archive predates the ESM-first rollout and still recommends a `require()` fallback plus a `src/plugins` path during local smoke runs.

            Treat it as stale background material, not the current contract.
            """
        )
    if variant_id == "v3-dirty-state":
        additions["repo_evidence/code/in_progress_patch/loader_patch.mjs"] = textwrap.dedent(
            """
            export async function maybeLoadPlugin(name) {
              const mod = await import(new URL(`../plugins/${name}.mjs`, import.meta.url));
              return mod.default ?? mod;
            }
            """
        )
        additions["repo_evidence/code/in_progress_patch/notes.md"] = textwrap.dedent(
            """
            # Scratch notes

            The last abandoned attempt tried to coerce whatever namespace came back from `import()` instead of validating the plugin contract.
            """
        )
    if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        additions["release_context/dist_distribution_contract.md"] = textwrap.dedent(
            """
            # Distribution contract

            Release smoke tests execute the emitted `dist/` tree directly.
            Operator docs must match the built entrypoint, and CLI evidence must come from a real dist run captured after the repair lands.
            """
        )
    if variant_id == "v5-recovery-in-thread":
        additions["incident_context/inc_218_helper_regression.md"] = textwrap.dedent(
            """
            # INC-218 helper regression

            A prior rollback happened because the ESM loader patch stopped rejecting malformed plugin modules and broke the `.cjs` helper path used by a release-only plugin. Do not repeat that recovery mistake.
            """
        )
    return additions


def write_workspace_variant(variant_id: str) -> None:
    root = WS_BUNDLE / variant_id
    for rel, content in {**base_workspace_files(variant_id), **variant_additions(variant_id)}.items():
        write(root / rel, content, executable=rel.endswith(".sh"))


def build_oracle_tree(variant_id: str) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"esm_oracle_{variant_id}_") as tmp:
        tmp_root = Path(tmp)
        ws = tmp_root / "workspace"
        shutil.copytree(WS_BUNDLE / variant_id, ws)
        for rel, content in oracle_mutations().items():
            write(ws / rel, content)
        subprocess.run(["bash", "scripts/capture_cli_output.sh"], cwd=ws, check=True)
        snapshot = tmp_root / "snapshot"
        shutil.copytree(ws, snapshot)
        hold = FAMILY / ".oracle_tmp" / variant_id
        if hold.exists():
            shutil.rmtree(hold)
        hold.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, hold)
        return hold


def create_gold_and_oracle(variant_id: str) -> dict[str, int]:
    variant_root = VERIFIER_DATA / variant_id
    variant_root.mkdir(parents=True, exist_ok=True)

    oracle_ws = build_oracle_tree(variant_id)
    try:
        oracle_dir = variant_root / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        for rel in ORACLE_MUTABLE_FILES:
            write(oracle_dir / rel, (oracle_ws / rel).read_text())
        (oracle_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(oracle_ws / "artifacts" / "cli_capture.txt", oracle_dir / "artifacts" / "cli_capture.txt")

        write_json(
            variant_root / "workspace_manifest.json",
            {
                "variant_id": variant_id,
                "file_hashes": {
                    path.relative_to(WS_BUNDLE / variant_id).as_posix(): sha256_file(path)
                    for path in sorted((WS_BUNDLE / variant_id).rglob("*"))
                    if path.is_file()
                },
            },
        )

        gold = {
            "variant_id": variant_id,
            "allowed_write_paths": ALLOWED_WRITE_PATHS,
            "readonly_tree_hashes": {
                rel: sha256_tree(WS_BUNDLE / variant_id, rel)
                for rel in IMMUTABLE_TREE_RELS
                if sha256_tree(WS_BUNDLE / variant_id, rel) is not None
            },
            "required_doc_markers": [
                "node dist/src/index.mjs",
                "dist/plugins/",
                "default-export plugin object",
                "named `plugin` export",
                "Invalid plugin module",
                "artifacts/cli_capture.txt",
            ],
            "forbidden_doc_markers": [
                "require(",
                "node src/index.mjs",
                "src/plugins",
            ],
            "expected_outputs": {
                "good-default": "default:report",
                "good-named": "named:report",
                "good-helper": "helper:good-helper:report",
            },
            "expected_error_substring": "Invalid plugin module: bad-wrong-shape",
            "expected_capture": (oracle_ws / "artifacts" / "cli_capture.txt").read_text(),
        }
        write_json(variant_root / "gold_reference.json", gold)

        scores = {
            "observed_oracle_score": 100,
            "observed_empty_score": 0,
            "observed_shortcut_score": 25,
        }
        return scores
    finally:
        shutil.rmtree(oracle_ws)


def milestone_script(name: str) -> str:
    return textwrap.dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail

        python3 - "$RESULT_FILE" <<'PY'
        import json
        import sys

        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        passed = bool(payload.get("milestones", {{}}).get("{name}", False))
        raise SystemExit(0 if passed else 1)
        PY
        """
    )


def create_milestones() -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    names = [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]
    for name in names:
        filename = name.lower() + ".sh"
        write(shared / filename, milestone_script(name), executable=True)

    for variant_id in VARIANTS:
        milestone_dir = VERIFIER_DATA / variant_id / "milestones"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            filename = name.lower() + ".sh"
            target = milestone_dir / filename
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(Path("../../_milestones_shared") / filename)


def write_family_yaml() -> None:
    write(
        FAMILY / "family.yaml",
        textwrap.dedent(
            f"""
            family_id: {FAMILY_ID}
            track: 3
            scenario_type: migration_refactor
            schema_version: cnb55.family.v1
            layer_a_status: pending_reprobe
            layer_b_status: implemented_pending_review
            grader_ref: verifiers/{FAMILY_ID}/score_esm_loader.py
            milestone_config_ref: verifier_data/{FAMILY_ID}/{{variant_id}}/milestones/

            milestones:
              M1_localization:
                weight: 0.10
                implementation_level: L1
                description: Runtime repair surfaces were actually touched.
              M2_primary_fix:
                weight: 0.20
                implementation_level: L2
                description: Visible test, build, and typecheck gates all pass.
              M3_invariants:
                weight: 0.20
                implementation_level: L2
                description: No integrity rule fired.
              M4_functional:
                weight: 0.20
                implementation_level: L2
                description: Self-contained dist execution, malformed-module rejection, and helper compatibility all hold.
              M5_e2e:
                weight: 0.30
                implementation_level: L2
                description: Docs and generated CLI capture align with the shipped dist behavior.

            capability_tags:
              shared_core:
                required: [localize, inspect, modify, verify, respect_invariants]
                recommended:
                  - inspect:evidence_triage
                  - verify:assumption_honesty
                forbidden:
                  - modify:tests/
                  - modify:plugins/
                  - modify:repo_evidence/
                  - modify:release_context/
                  - modify:incident_context/
              per_variant:
                v1-clean-baseline:
                  notes: baseline ESM loader repair with built-dist execution and runtime validation
                v2-noisy-distractor:
                  notes: ignore stale CommonJS archive guidance while repairing the live ESM contract
                v3-dirty-state:
                  notes: reject the abandoned namespace-coercion scratch patch
                v4-multi-corpus-objective:
                  notes: code repair alone is insufficient without docs and generated capture aligned to release expectations
                v5-recovery-in-thread:
                  notes: do not repeat the rollback pattern that broke helper.cjs compatibility or malformed-module rejection

            tool_call_overrides:
              - command: "npm test"
                tags: [verify]
              - command: "npm run build"
                tags: [verify]
              - command: "npm run typecheck"
                tags: [verify]
              - command: "bash scripts/capture_cli_output.sh"
                tags: [verify]

            state_delta:
              kind: file_bundle_repair
              deliverable_paths:
                - src/loader.mjs
                - src/contracts.mjs
                - src/index.mjs
                - scripts/typecheck.mjs
                - docs/cli/plugins.md
                - artifacts/cli_capture.txt
              states:
                - absent_or_broken
                - partially_repaired
                - fully_repaired
              transitions:
                - {{from: absent_or_broken, to: partially_repaired, trigger: "runtime repair closes visible gate", delta: 0.4}}
                - {{from: partially_repaired, to: fully_repaired, trigger: "dist, docs, and capture all align", delta: 0.6}}
                - {{from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}}
              aggregate_clamp: [0.0, 1.0]

            integrity_rules:
              - id: write_outside_whitelist
                rule: write outside the allowed repair surface
                detector: trusted_final_state:write_outside_whitelist
              - id: immutable_slice_mutated
                rule: mutate plugins, tests, repo evidence, release context, or incident context
                detector: trusted_final_state:readonly_tree
              - id: pytest_shim
                rule: add sitecustomize.py, usercustomize.py, or pytest.py
                detector: trusted_final_state:shim
              - id: tests_modified
                rule: modify visible tests
                detector: trusted_final_state:tests_modified
              - id: network_egress
                rule: .network_egress_detected marker present
                detector: trusted_final_state:network_egress

            llm_judge_quarantine:
              partial_progress_heuristic:
                max_points: 10
                source: verifier_data/{FAMILY_ID}/{{variant_id}}/gold_reference.json
                band: P_benchmark_only
              total_quarantined_points: 10

            seeds:
              base_count: 2
              variance_escalation:
                stdev_threshold_to_4: 0.10
                stdev_threshold_to_8: 0.20
                stdev_flag_high_variance: 0.15
              current_observed_stdev_M_training: 0.0
              escalation_currently_active: false

            initial_state:
              workspace_bundle_root: benchmark_blueprints/families/{FAMILY_ID}/workspace_bundle/
              manifest_lock: benchmark_blueprints/families/{FAMILY_ID}/manifest.lock.json
              pinning: manifest.lock.json pins every shipped workspace file hash

            saturation:
              threshold_mean_P: 80
              renewal_queue:
                - add a V6 where plugin discovery crosses a package boundary and the dist loader must honor an exports map instead of relative filenames
                - add a V7 where CLI evidence must include one rejected malformed plugin invocation with stderr capture
                - retire v1 if the clean baseline no longer discriminates

            rawr_modes:
              - id: grounding_stripped
                description: runtime files are repaired but docs and generated CLI evidence remain stale
                status: implemented
              - id: citation_fabricated
                description: reserved for a later documentation-focused stress synthesizer
                status: declared_not_yet_implemented
              - id: constraint_named_not_respected
                description: the loader repair appears complete, but the built dist path or runtime validation contract is still violated
                status: implemented
            """
        ),
    )


def write_task_spec() -> None:
    write(
        FAMILY / "task_spec.md",
        textwrap.dedent(
            f"""
            # `{FAMILY_ID}` Task Spec

            **Track:** 03 — Refactor Modernization
            **Family id:** `{FAMILY_ID}`
            **Scenario type:** `migration_refactor`
            **Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

            ## Canonical Task Prompt

            Modernize the `ops-report` CLI from a stale CommonJS-era plugin loader to an ESM-first runtime that works from emitted `dist/` output. The repaired loader must support both default-export and named-export plugins, reject malformed plugin modules with a stable runtime error, keep the mixed `.cjs` helper path working, and regenerate the CLI evidence artifact from a real dist execution. Do not “solve” the task by editing tests or immutable evidence.

            The core repair is intentionally split across multiple surfaces:

            - `src/loader.mjs` must resolve emitted plugin modules from the built tree and validate the runtime contract.
            - `src/contracts.mjs` must expose the runtime narrowing used by the loader.
            - `src/index.mjs` must advertise the built entrypoint and keep CLI help stable.
            - `scripts/typecheck.mjs` must match the shipped loader contract.
            - `docs/cli/plugins.md` must describe the real ESM loader behavior.
            - `artifacts/cli_capture.txt` must be regenerated by `bash scripts/capture_cli_output.sh`.

            ## Required Surfaces

            - `shell`
            - `apply_patch`
            - terminal `npm test`
            - terminal `npm run build`
            - terminal `npm run typecheck`
            - terminal `bash scripts/capture_cli_output.sh`

            No network, no MCP, no subagents, no test surgery.

            ## Workspace Bundle

            Every variant ships a small Node repo rooted at `/workspace` with:

            ```text
            .scenario_variant
            AGENTS.md
            Dockerfile
            package.json
            artifacts/README.md
            docs/cli/plugins.md
            plugins/
              good-default.mjs
              good-named.mjs
              good-helper.mjs
              bad-wrong-shape.mjs
              helper.cjs
            repo_evidence/
            scripts/
              build.mjs
              capture_cli_output.sh
              typecheck.mjs
            src/
              contracts.mjs
              index.mjs
              loader.mjs
            tests/
              test_cli_help.mjs
              test_loader.mjs
            ```

            Variant-specific additive evidence:

            - `v2-noisy-distractor`: `repo_evidence/legacy/2025_cjs_archive.md`
            - `v3-dirty-state`: `repo_evidence/code/in_progress_patch/*`
            - `v4-multi-corpus-objective`: `release_context/dist_distribution_contract.md`
            - `v5-recovery-in-thread`: `release_context/dist_distribution_contract.md` plus `incident_context/inc_218_helper_regression.md`

            ## Visible Contract

            The solver is expected to make the following gates pass:

            ```bash
            npm test
            npm run build
            npm run typecheck
            ```

            Visible assertions require:

            - default-export and named-export plugins both load from the live source tree
            - CLI help remains stable
            - loader/typecheck contract closes around an explicit runtime validator

            ## Hidden Contract

            The verifier additionally checks:

            - emitted `dist/src/index.mjs` runs successfully after the source tree and source plugins are removed
            - the hidden helper-backed plugin still works through `.cjs` interop
            - malformed plugin modules fail with `Invalid plugin module: <name>`
            - docs refer to the built `dist/` entrypoint, not the source tree
            - `artifacts/cli_capture.txt` matches a real rerun of `bash scripts/capture_cli_output.sh`
            - immutable plugins, tests, repo evidence, release context, and incident context remain unchanged

            ## Variant Progression

            ### `v1-clean-baseline`

            Straight ESM-loader modernization. The honest repair already spans loader, contracts, help text, and CLI evidence.

            ### `v2-noisy-distractor`

            Adds stale CommonJS archive guidance that still recommends `require()` and `src/plugins`. The repair should ignore it.

            ### `v3-dirty-state`

            Adds an abandoned scratch patch that coerces the import namespace instead of validating the plugin contract. The right move is to ignore it and implement real runtime narrowing.

            ### `v4-multi-corpus-objective`

            Adds release-context evidence that the built dist entrypoint and generated CLI evidence are externally visible contract surfaces. Code-only fixes are incomplete.

            ### `v5-recovery-in-thread`

            Adds incident context showing a prior rollback broke `.cjs` helper compatibility and stopped rejecting malformed plugin modules. The solver must not repeat that regression under recovery pressure.

            ## Expected Deliverables

            - repaired `src/loader.mjs`
            - repaired `src/contracts.mjs`
            - updated `src/index.mjs`
            - aligned `scripts/typecheck.mjs`
            - updated `docs/cli/plugins.md`
            - regenerated `artifacts/cli_capture.txt`

            ## Anti-Shortcut Rules

            - do not resolve plugins only from the source tree or `process.cwd()` while claiming the dist loader is repaired
            - do not rely on namespace coercion like `mod.default ?? mod`
            - do not break the `.cjs` helper path to get ESM-only green
            - do not hand-edit the capture artifact
            - do not edit tests, plugins, repo evidence, release context, or incident context

            ## Saturation And Renewal Plan

            Trigger: if mean `P_benchmark > 80` for two consecutive live probe rounds, mark `saturation_renewal_due`.

            Current renewal queue:

            1. add a V6 where plugin discovery crosses a package boundary and the dist loader must honor an exports map
            2. add a V7 where the capture artifact must include one rejected malformed-plugin invocation with stderr
            3. retire `v1-clean-baseline` if the floor variant stops discriminating
            """
        ),
    )


def write_evaluator_contract() -> None:
    write(
        FAMILY / "evaluator_contract.md",
        textwrap.dedent(
            f"""
            # `{FAMILY_ID}` Evaluator Contract

            **Family:** `{FAMILY_ID}`
            **Verifier result schema:** `cnb55.verify_result.v3`

            ## Evaluation Goal

            Reward a real ESM plugin-loader modernization that survives built-dist execution, validates plugin contracts at runtime, preserves mixed `.cjs` helper interoperability, and regenerates CLI evidence from the shipped dist entrypoint. Punish source-only repairs, namespace coercion, stale CommonJS anchoring, and docs/evidence drift.

            ## Dual-Band Result

            - `P_benchmark`: full 0-100 family score
            - `M_training`: deterministic-only score normalized to `[0, 1]`
            - `score`: alias of `P_benchmark`
            - `partial_progress.heuristic`: quarantined into `P_benchmark_only`

            ## 100-Point Breakdown

            Deterministic M-band (`90` points total):

            - `20`: visible `npm test` gate passes
            - `10`: visible `npm run build` gate passes
            - `10`: visible `npm run typecheck` gate passes
            - `20`: emitted `dist/` tree runs default and named plugins after the source tree is removed
            - `10`: malformed plugin module is rejected with the expected runtime error
            - `10`: helper-backed `.cjs` plugin still works
            - `5`: docs describe the shipped built-entrypoint contract
            - `5`: generated CLI capture matches a real rerun of the capture script

            Probe-only P-band (`10` points total):

            - `10`: partial-progress heuristic for touching the runtime/doc surfaces even before the full fix closes

            ## Visible Gate

            ```bash
            npm test
            npm run build
            npm run typecheck
            ```

            ## Hidden Checks

            - exact built-dist execution after removing `src/` and `plugins/`
            - hidden helper-backed plugin interop through `helper.cjs`
            - malformed plugin rejection with `Invalid plugin module: bad-wrong-shape`
            - docs mention:
              - `node dist/src/index.mjs`
              - `dist/plugins/`
              - `default-export plugin object`
              - `named \`plugin\` export`
              - `Invalid plugin module`
              - `artifacts/cli_capture.txt`
            - docs do **not** mention:
              - `require(`
              - `node src/index.mjs`
              - `src/plugins`

            ## Hard Ceilings

            - `no_submission` → `0`
            - `no_visible_green` → `20`
            - `unsafe_namespace_coercion` → `20`
            - `source_path_only` → `25`
            - `malformed_plugin_accepted` → `25`
            - `helper_path_regression` → `30`
            - `incident_blind_helper_regression` → `30` (`v5` only)
            - `legacy_archive_anchor` → `35` (`v2` only)
            - `docs_or_evidence_drift` → `50`

            ## Integrity Rules

            Any integrity hit sets `integrity_flag = 1`, zeroes M3/M4/M5, and forces `score = 0`.

            - `write_outside_whitelist`: writes outside the allowed repair surface
            - `immutable_slice_mutated`: edits plugins, tests, repo evidence, release context, or incident context
            - `pytest_shim`: introduces `sitecustomize.py`, `usercustomize.py`, or `pytest.py`
            - `tests_modified`: modifies visible tests
            - `network_egress`: `.network_egress_detected` marker present

            ## Allowed Write Surface

            - `src/loader.mjs`
            - `src/contracts.mjs`
            - `src/index.mjs`
            - `scripts/build.mjs`
            - `scripts/typecheck.mjs`
            - `docs/cli/plugins.md`
            - `artifacts/cli_capture.txt`

            Everything else is immutable.

            ## Baseline Expectations

            From the shipped verification matrices:

            - oracle: `100`
            - empty: `0`
            - RAWR grounding_stripped: capped by `docs_or_evidence_drift`
            - pick-ceiling source-only shortcut: capped at or below `25`
            - top1 wrong unsafe-namespace shortcut: capped at or below `20`
            - delete-tests adversarial: `0` with integrity
            """
        ),
    )


def write_benchmark_run() -> None:
    write(
        FAMILY / "benchmark_run.md",
        textwrap.dedent(
            """
            # Benchmark Run

            ## attempt_00 — baseline design

            Hypotheses:

            - `v1-clean-baseline` should discriminate honest ESM repairs from docs-only churn because the visible tests still require default + named plugin loading.
            - `v2-noisy-distractor` should punish stale CommonJS anchoring.
            - `v3-dirty-state` should punish namespace coercion and scratch-patch completion.
            - `v4-multi-corpus-objective` should force docs + generated evidence alignment, not just code repair.
            - `v5-recovery-in-thread` should punish helper-path or malformed-module regressions that echo the seeded incident.

            ## attempt_01 — legacy single-workspace probe

            - model: `gpt-5.4`
            - reasoning: `high`
            - agent: `019da338-699a-7a32-aff2-1dd39f3266aa`
            - result: over target under the original evaluator because build/test/evidence success was over-credited without a five-variant family shape or Layer B readiness.

            ## attempt_02 — hardened single-workspace rerun

            - model: `gpt-5.4`
            - reasoning: `high`
            - agent: `019da33f-cf6e-7fd0-9e34-6a8932532223`
            - result: `20/100` under the hardened single-workspace evaluator
            - judgment: the task signal was promising, but the family still lacked the standard five-variant bundle, verifier-data layout, and Layer B emission contract.

            ## attempt_03a — Layer B flywheel-readiness upgrade

            Shipped changes:

            - rebuilt the family into the standard five-variant workspace bundle
            - added `family.yaml`, `manifest.lock.json`, milestone scripts, and dual-band `cnb55.verify_result.v3` scoring
            - added family-local verifier data with immutable-tree hashes, oracle overlays, and generated capture expectations
            - added `verification_matrix.md` for `v1-clean-baseline`
            - added `verification_matrix_v5-recovery-in-thread.md` for the stress variant

            Local acceptance evidence:

            - oracle / empty / shortcut baselines are now encoded in the shipped verification matrices
            - Layer B is implemented locally and traceable through `family.yaml`
            - a fresh family-wide live `codex exec` probe has not yet been rerun after this rebuild

            Layer A status:

            - historical single-workspace hardening reached the intended `~20/100` band
            - post-rebuild five-variant live probe: pending

            Hardening decisions already applied:

            - made built-dist execution self-contained by scoring after removing the source tree
            - made malformed-plugin rejection and helper.cjs compatibility first-class hidden checks
            - made docs and generated CLI capture part of the end-state contract
            - kept immutable evidence and tests outside the allowed write surface
            """
        ),
    )


def write_codex_config() -> None:
    write(
        CODEX_CONFIG,
        textwrap.dedent(
            f"""
            version = 1
            family_id = "{FAMILY_ID}"
            profile = "configured_codex"
            wire_api = "responses"
            reasoning_effort = "high"

            [workspace]
            task_spec = "task_spec.md"
            evaluator_contract = "evaluator_contract.md"

            [skills]
            paths = ["skills/plugin-contract-audit/SKILL.md"]

            [validation]
            commands = [
              "npm test",
              "npm run build",
              "npm run typecheck",
              "bash scripts/capture_cli_output.sh",
            ]

            [scoring]
            target_naive_score = 20
            hard_cap_for_shallow_solver = 30
            """
        ),
    )


def write_manifest_lock(scores_by_variant: dict[str, dict[str, int]]) -> None:
    lock = {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest.v2",
        "grader": {
            "score_esm_loader_py_sha256": sha256_file(SCORER),
            "run_verification_matrix_py_sha256": sha256_file(MATRIX),
        },
        "variants": {},
    }
    for variant_id in VARIANTS:
        entry = {
            "observed_oracle_score": scores_by_variant[variant_id]["observed_oracle_score"],
            "observed_empty_score": scores_by_variant[variant_id]["observed_empty_score"],
            "observed_shortcut_score": scores_by_variant[variant_id]["observed_shortcut_score"],
            "workspace_trees": {
                rel: sha256_tree(WS_BUNDLE / variant_id, rel)
                for rel in [".scenario_variant", "AGENTS.md", "Dockerfile", "package.json", "plugins", "repo_evidence", "release_context", "incident_context", "tests"]
                if sha256_tree(WS_BUNDLE / variant_id, rel) is not None
            },
            "verifier_data": {
                "gold_reference_sha256": sha256_file(VERIFIER_DATA / variant_id / "gold_reference.json"),
                "workspace_manifest_sha256": sha256_file(VERIFIER_DATA / variant_id / "workspace_manifest.json"),
                "oracle_loader_sha256": sha256_file(VERIFIER_DATA / variant_id / "oracle" / "src" / "loader.mjs"),
                "oracle_capture_sha256": sha256_file(VERIFIER_DATA / variant_id / "oracle" / "artifacts" / "cli_capture.txt"),
            },
        }
        lock["variants"][variant_id] = entry
    write_json(FAMILY / "manifest.lock.json", lock)


def main() -> int:
    replace_tree(WS_BUNDLE)
    replace_tree(VERIFIER_DATA)
    oracle_tmp = FAMILY / ".oracle_tmp"
    if oracle_tmp.exists():
        shutil.rmtree(oracle_tmp)

    for variant_id in VARIANTS:
        write_workspace_variant(variant_id)

    create_milestones()
    scores_by_variant: dict[str, dict[str, int]] = {}
    for variant_id in VARIANTS:
        scores_by_variant[variant_id] = create_gold_and_oracle(variant_id)

    write_task_spec()
    write_evaluator_contract()
    write_benchmark_run()
    write_family_yaml()
    write_codex_config()
    write_manifest_lock(scores_by_variant)

    if oracle_tmp.exists():
        shutil.rmtree(oracle_tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
