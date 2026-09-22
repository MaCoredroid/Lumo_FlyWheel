#!/usr/bin/env python3
"""Audit archived provenance and reconstruct sampler patching; no model inference.

The stock-image input is read from the pinned image on the user's Spark. This
script executes only the patcher's text-rewrite function against disposable
source files, never imports the generated vLLM code or starts a model.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import textwrap

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[2]
OUT = PAPER / "p0"
PATCHER = "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
ARMS = ("native5_control_kvr1", "native11_control_kvr1", "kvremap_tail6_kvr1")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_source(rev, name):
    return subprocess.check_output(["git", "show", f"{rev}:{name}"], cwd=REPO)


def env_values(path):
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def reconstruct(patcher, stock, label):
    root = ast.parse(patcher)
    node = next(n for n in root.body if isinstance(n, ast.FunctionDef)
                and n.name == "_patch_rejection_sampler_tree_lcp")
    # No module imports/main execution: only the existing text-rewrite function.
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    path = OUT / "stock-image" / f"reconstructed-{label}.py"
    path.write_text(stock)
    namespace = {"REJECTION_SAMPLER_PATH": path, "os": os, "re": re,
                 "json": json, "Path": Path}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if label == "current":
            # The modern full patcher has unrelated deployment prerequisites.
            # Execute its exact constraint-insertion block with its literal
            # anchors; do not stub those prerequisites or claim a full boot.
            wanted = {"target_apply", "target_precap", "target_processor_anchor", "stock_call"}
            for assignment in ast.walk(node):
                if isinstance(assignment, ast.Assign):
                    for target in assignment.targets:
                        if isinstance(target, ast.Name) and target.id in wanted:
                            namespace[target.id] = ast.literal_eval(assignment.value)
            body = ast.get_source_segment(patcher, node)
            begin = body.index("        stock_call_pos = text.find(stock_call)")
            end = body.index("        text = text.replace(stock_call, stock_call_new, 1)", begin)
            namespace["text"] = stock
            exec(textwrap.dedent(body[begin:end]), namespace)
            path.write_text(namespace["text"])
            changed = namespace["text"] != stock
        else:
            exec(compile(module, PATCHER, "exec"), namespace)
            changed = namespace[node.name]()
    source = path.read_text()
    parsed = ast.parse(source)
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef)
               and n.name == "RejectionSampler")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == "forward")
    calls = []
    for n in ast.walk(forward):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "apply_sampling_constraints"
                and n.args and isinstance(n.args[0], ast.Name)
                and n.args[0].id == "target_logits"):
            calls.append({"line": n.lineno, "source": ast.get_source_segment(source, n)})
    return {"changed": changed, "target_constraint_calls_in_forward": len(calls),
            "calls": calls, "patcher_sha256": sha(patcher.encode()),
            "reconstructed_sampler_sha256": sha(source.encode()),
            "reconstructed_file": str(path.relative_to(PAPER)),
            "patcher_messages": output.getvalue().strip(),
            "scope": ("exact current constraint-insertion block; not a full patcher boot"
                      if label == "current" else
                      "full historical sampler text-rewrite function; not a recovered runtime file")}


def main():
    stock_input = json.loads((OUT / "stock-image/sampling-source.json").read_text())
    stock = stock_input["files"]["v1/sample/rejection_sampler.py"]["source"]
    reports = []
    interesting = {
        "FR10_DECODE_MODE_DEFAULT", "FR13_ATTN_KV_REMAP", "FR13_SLOT_REORDER",
        "FR13_ENABLE_APC", "FR13_TEMP_LEGACY_DOUBLE", "FR13_GPU_COMMITTER",
        "VLLM_BATCH_INVARIANT", "FR13_FIXED32_GDN_MODE", "FR13_DRAFT_SOURCE",
        "FR13_STATELESS_TREE", "FR13_TREE_TAIL_DEPTH", "FR13_DECODE_FORWARD_GPU_TIMER",
        "FR13_STEP_WALL_TIMER", "FR13_DFWD_GPU_TIMER", "FR13_CFWD_GPU_TIMER",
    }
    for arm in ARMS:
        d = REPO / "output/fr13_kvremap_tail6" / arm
        rev = (d / "git_head.txt").read_text().strip()
        patcher = git_source(rev, PATCHER).decode()
        reconstruction = reconstruct(patcher, stock, arm)
        assert reconstruction["target_constraint_calls_in_forward"] == 2
        boot = (d / "boot_log_snapshot.txt").read_text()
        env = env_values(d / "container_env.txt")
        proxy = env_values(d / "proxy_env.txt")
        metrics = json.loads((d / "deploy_speed_kvr1.json").read_text())
        args_line = next(s.split("non-default args: ", 1)[1]
                         for s in boot.splitlines() if "non-default args: " in s)
        # vLLM logs Python enum reprs inside the otherwise literal dictionary.
        args = ast.literal_eval(re.sub(r"<[^<>]+>", lambda m: repr(m.group(0)), args_line))
        hashes = [{"path": str((d / n).relative_to(REPO)),
                   "sha256": sha((d / n).read_bytes())}
                  for n in ["git_head.txt", "container_env.txt", "proxy_env.txt",
                            "boot_log_snapshot.txt", "deploy_speed_kvr1.json"]]
        requested = {k: proxy.get(k) for k in ["LUMO_PROXY_FORCE_TEMPERATURE",
                    "LUMO_PROXY_FORCE_TOP_P", "LUMO_PROXY_FORCE_TOP_K"]}
        reports.append({
            "evidence_id": "H3", "arm": arm, "source_revision": rev,
            "started_utc": (d / "arm_started_at.txt").read_text().strip(),
            "ended_utc": (d / "arm_ended_at.txt").read_text().strip(),
            "requested_sampling": requested,
            "recorded_flags": {k: env[k] for k in sorted(interesting) if k in env},
            "boot_arguments": {k: args.get(k) for k in ["model", "attention_backend",
                "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                "gpu_memory_utilization", "mamba_ssm_cache_dtype", "block_size",
                "enable_prefix_caching", "speculative_config", "compilation_config"]},
            "vllm_version": "0.19.2rc1.dev134+gfe9c3d6c5 (boot record)",
            "sampler_reconstruction": reconstruction,
            "target_temperature_scale_inferred": 0.36,
            "sampling_caveat": "Two full constraint passes include top-k/top-p; bonus/self-logit paths can differ. Not a globally equivalent clean temperature-0.36 run.",
            "measurement": {k: metrics[k] for k in ["schema", "speed_basis",
                "s_per_fwd_gpu_basis", "measured_tps_fullstep_wall_note",
                "component_gpu_note", "events_per_step", "accept_per_event",
                "committed_per_event", "wall_s_per_event", "engagement"]},
            "artifact_hashes": hashes,
            "historical_binary_identity": "Unresolved: no run-time FA2/JIT binary digest in selected run artifacts; current disk hash cannot backfill it",
            "historical_model_revision": "Unresolved: boot revision=None and mutable /models path; current model hashes do not prove July content",
            "decision": "RELABEL: retain archived observation; requested T=0.6, duplicated target constraints source-reconstructed; do not claim nominal sampling equivalence or a current matched-build comparison",
        })
    current_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    current = reconstruct(git_source(current_rev, PATCHER).decode(), stock, "current")
    assert current["target_constraint_calls_in_forward"] == 1
    result = {"audit": "P0 source/configuration reconciliation without model inference",
              "base_image_versions": stock_input["versions"],
              "historical_runs": reports,
              "selected_revision": current_rev,
              "selected_revision_sampler": current,
              "remaining_runtime_gate": "Fresh boot must archive actual patched files, native extension and JIT identities, model checksums, and sampler engagement before final measurements"}
    (OUT / "historical-run-manifests.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": len(reports), "historical_target_constraint_calls": [x["sampler_reconstruction"]["target_constraint_calls_in_forward"] for x in reports], "current_target_constraint_calls": current["target_constraint_calls_in_forward"]}))


if __name__ == "__main__":
    main()
