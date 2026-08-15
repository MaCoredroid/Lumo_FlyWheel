#!/usr/bin/env python3
"""Generate the CFWD packed-v3 runtime overlay from its reviewed source commit."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_REVISION = "103030ea88ad7da28a4bcab187a57200be70756d"
SOURCE = "scripts/fr13_device_multidraft_kernel.py"
OUTPUT = ROOT / "scripts/fr13_cfwd_logit_direct_packed_runtime_overlay.py"
CHANGED_FUNCTIONS = (
    "_fr13_cfwd_logit_direct_state",
    "fr13_fixed32_cfwd_logit_direct_capture_begin",
    "fr13_fixed32_cfwd_logit_direct_capture_end",
    "_fr13_cfwd_logit_direct_walk_cuda",
    "_fr13_cfwd_logit_direct_compare",
    "fr13_fixed32_cfwd_logit_direct_warm_execute",
)
LOCAL_FUNCTIONS = (
    "fr13_fixed32_cfwd_logit_direct_capture_begin",
    "fr13_fixed32_cfwd_logit_direct_capture_end",
)
CHANGED_KERNELS = (
    "_fr13_fixed32_taw_packed_physical_slot_commit_kernel",
    "_fr13_cfwd_logit_direct_compare_kernel",
)


if False:  # Extracted into the generated runtime overlay.

    def fr13_fixed32_cfwd_logit_direct_capture_begin(
        graph_id: int,
        *,
        mode: str,
        batch_size: int,
    ) -> None:
        """Bind prewarmed committer state to the target graph identity."""
        global _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if int(batch_size) not in (1, 4):
            return
        selector = _fr13_cfwd_logit_direct_selector(
            mode=mode, batch_size=int(batch_size)
        )
        if selector == "reference":
            return
        if _FR13_CFWD_LOGIT_DIRECT_CAPTURE is not None:
            raise RuntimeError("FR13 CFWD logit-direct captures overlapped")
        identity = int(graph_id)
        if identity <= 0 or identity in _FR13_CFWD_LOGIT_DIRECT_GRAPHS:
            raise RuntimeError("FR13 CFWD logit-direct graph identity was reused")
        _, valid_mask = _fr13_fixed32_runtime_contract(mode)
        entry = _fr13_cfwd_logit_direct_entry(mode, int(batch_size))
        key = fr13_fixed32_taw_cache_key(
            mode,
            valid_mask,
            int(batch_size),
            entry["child_table"].device,
        )
        state = _FR13_CFWD_LOGIT_DIRECT_WARM.get(key)
        if (
            not isinstance(state, dict)
            or state.get("graph_id") is not None
            or state.get("mode") != mode
            or state.get("batch_size") != int(batch_size)
            or state.get("device") != entry["child_table"].device
            or state.get("bound_calls") != 0
        ):
            raise RuntimeError("FR13 CFWD logit-direct prewarmed state drift")
        state["graph_id"] = identity
        _FR13_CFWD_LOGIT_DIRECT_GRAPHS[identity] = state
        _FR13_CFWD_LOGIT_DIRECT_CAPTURE = state


    def fr13_fixed32_cfwd_logit_direct_capture_end(
        graph_id: int,
        *,
        mode: str,
        batch_size: int,
    ) -> None:
        """Verify target capture excludes CFWD, then bind its external call site."""
        global _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if int(batch_size) not in (1, 4):
            return
        selector = _fr13_cfwd_logit_direct_selector(
            mode=mode, batch_size=int(batch_size)
        )
        if selector == "reference":
            return
        state = _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if (
            not isinstance(state, dict)
            or state.get("graph_id") != int(graph_id)
            or state.get("mode") != mode
            or state.get("batch_size") != int(batch_size)
            or state.get("bound_calls") != 0
        ):
            raise RuntimeError("FR13 CFWD logit-direct capture binding drift")
        state["bound_calls"] = 1
        _FR13_CFWD_LOGIT_DIRECT_CAPTURE = None


def _candidate_source() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{CANDIDATE_REVISION}:{SOURCE}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def _definition(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"candidate definition is incomplete or ambiguous: {name}")
    node = matches[0]
    starts = [node.lineno, *(item.lineno for item in node.decorator_list)]
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[min(starts) - 1 : node.end_lineno])).rstrip()


def _indent(source: str) -> str:
    return textwrap.indent(source, "    ")


def generate() -> str:
    candidate = _candidate_source()
    local = Path(__file__).resolve().read_text(encoding="utf-8")
    definitions = "\n\n\n".join(
        _indent(_definition(local if name in LOCAL_FUNCTIONS else candidate, name))
        for name in (*CHANGED_KERNELS, *CHANGED_FUNCTIONS)
    )
    return f'''#!/usr/bin/env python3
"""Install reviewed packed-CFWD definitions without changing TAW source bytes."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any


BASE_SOURCE_SHA256 = "6e1f09f55327428a8a4b9cfdb885dcdba4c7457f58fb2b7d8b5169083dba6cf2"
CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
CANDIDATE_SOURCE_SHA256 = "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
INTEGRATION_SOURCE_SCHEMA = "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
INTEGRATION_SOURCE_SHA256 = "421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b"
CHANGED_FUNCTIONS = {CHANGED_FUNCTIONS!r}
CHANGED_KERNELS = {CHANGED_KERNELS!r}


if False:  # Parsed and installed into the credential-bound base module.
{definitions}


    def _fr13_cfwd_logit_direct_integration_source_contract() -> dict[str, str]:
        """Bind the unchanged base plus the explicit CFWD-only overlay."""
        global _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE
        if _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE is not None:
            return dict(_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE)

        try:
            base_tree = ast.parse(Path(__file__).resolve().read_text(encoding="utf-8"))
            overlay_tree = ast.parse(
                Path(_FR13_CFWD_LOGIT_DIRECT_OVERLAY_PATH)
                .resolve()
                .read_text(encoding="utf-8")
            )
        except (OSError, SyntaxError) as error:
            raise RuntimeError(
                "FR13 CFWD logit-direct cannot inspect composed integration source"
            ) from error
        expected_functions = set(
            _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS
        )
        expected_kernels = set(
            _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS
        )
        expected = expected_functions | expected_kernels
        overlay_names = set(_FR13_CFWD_LOGIT_DIRECT_OVERLAY_DEFINITIONS)
        definitions: dict[str, list[Any]] = {{name: [] for name in expected}}
        for tree, use_names in (
            (base_tree, expected - overlay_names),
            (overlay_tree, expected & overlay_names),
        ):
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in use_names
                ):
                    definitions[node.name].append(node)
        if any(len(nodes) != 1 for nodes in definitions.values()):
            raise RuntimeError(
                "FR13 CFWD composed integration source is incomplete or ambiguous"
            )

        normalized = {{
            name: ast.dump(
                definitions[name][0],
                annotate_fields=True,
                include_attributes=False,
            )
            for name in sorted(expected)
        }}
        canonical = json.dumps(
            {{
                "schema": _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA,
                "candidate": {{
                    "name": _FR13_CFWD_LOGIT_DIRECT_CANDIDATE,
                    "schema": _FR13_CFWD_LOGIT_DIRECT_SCHEMA,
                    "source_sha256": _FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256,
                }},
                "geometry": _FR13_FIXED32_TAW_GEOMETRY,
                "functions": {{
                    name: normalized[name] for name in sorted(expected_functions)
                }},
                "kernels": {{
                    name: normalized[name] for name in sorted(expected_kernels)
                }},
            }},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        if digest != _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256:
            raise RuntimeError(
                "FR13 CFWD composed integration source identity drifted: " + digest
            )
        contract = {{
            "integration_source_schema": (
                _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA
            ),
            "integration_source_sha256": digest,
        }}
        _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE = dict(contract)
        return contract


def _definitions(tree: ast.AST) -> dict[str, ast.AST]:
    wanted = {{
        *CHANGED_FUNCTIONS,
        *CHANGED_KERNELS,
        "_fr13_cfwd_logit_direct_integration_source_contract",
    }}
    found: dict[str, list[ast.AST]] = {{name: [] for name in wanted}}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in found:
                found[node.name].append(node)
    if any(len(nodes) != 1 for nodes in found.values()):
        raise RuntimeError("packed CFWD overlay definition set drifted")
    return {{name: nodes[0] for name, nodes in found.items()}}


def install(module: ModuleType) -> dict[str, str]:
    """Install only reviewed CFWD names into an unchanged TAW base module."""
    overlay_path = Path(__file__).resolve()
    base_path = Path(module.__file__).resolve()
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SOURCE_SHA256:
        raise RuntimeError("packed CFWD overlay base source identity drifted")
    topology = module._fr13_fixed32_topology()
    taw_before = module._fr13_fixed32_taw_source_contract(topology, batch_size=1)
    taw_functions = {{
        name: getattr(module, name)
        for name in module._FR13_FIXED32_TAW_SOURCE_FUNCTIONS
    }}

    tree = ast.parse(overlay_path.read_text(encoding="utf-8"))
    definitions = _definitions(tree)
    namespace = module.__dict__
    namespace.update(
        {{
            "_FR13_CFWD_LOGIT_DIRECT_CANDIDATE": CANDIDATE,
            "_FR13_CFWD_LOGIT_DIRECT_SCHEMA": CANDIDATE_SCHEMA,
            "_FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256": CANDIDATE_SOURCE_SHA256,
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA": (
                INTEGRATION_SOURCE_SCHEMA
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256": (
                INTEGRATION_SOURCE_SHA256
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS": (
                CHANGED_KERNELS
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE": None,
            "_FR13_CFWD_LOGIT_DIRECT_MODULE": None,
            "_FR13_CFWD_LOGIT_DIRECT_OVERLAY_PATH": str(overlay_path),
            "_FR13_CFWD_LOGIT_DIRECT_OVERLAY_DEFINITIONS": (
                *CHANGED_FUNCTIONS,
                *CHANGED_KERNELS,
            ),
        }}
    )
    function_nodes = [
        copy.deepcopy(definitions[name])
        for name in (
            *CHANGED_FUNCTIONS,
            "_fr13_cfwd_logit_direct_integration_source_contract",
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=function_nodes, type_ignores=[])),
            str(overlay_path),
            "exec",
        ),
        namespace,
    )
    if module.triton is not None:
        kernel_nodes = [copy.deepcopy(definitions[name]) for name in CHANGED_KERNELS]
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=kernel_nodes, type_ignores=[])
                ),
                str(overlay_path),
                "exec",
            ),
            namespace,
        )

    taw_after = module._fr13_fixed32_taw_source_contract(topology, batch_size=1)
    if taw_after != taw_before or any(
        getattr(module, name) is not value for name, value in taw_functions.items()
    ):
        raise RuntimeError("packed CFWD overlay changed the TAW source contract")
    contract = module._fr13_cfwd_logit_direct_integration_source_contract()
    if contract != {{
        "integration_source_schema": INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": INTEGRATION_SOURCE_SHA256,
    }}:
        raise RuntimeError("packed CFWD overlay integration contract drifted")
    return contract
'''


def main() -> int:
    OUTPUT.write_text(generate(), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
