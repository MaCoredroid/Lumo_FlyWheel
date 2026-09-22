#!/usr/bin/env python3
"""Fail-closed offline E8 stage/run validation; never launches an experiment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from e8_shim import render


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError("E8 REFUSED: " + message)


def stage(path):
    root = Path(path).resolve()
    m = json.loads((root / "manifest.json").read_text())
    check(m["schema"] == "e8.freeze.v1" and m["batch"] == 1, "manifest scope")
    for rel, expected in m["files"].items():
        p = root / rel
        check(p.resolve().is_relative_to(root) and p.is_file(), "missing/escaping source " + rel)
        check(sha(p) == expected, "source hash " + rel)
    return m


def head_report(d, arm):
    check(d.get("schema") == "e8.head.v1" and d.get("arm") == arm and d.get("qualify") is True, "head report identity")
    check(d.get("closed") is True and not d.get("failures"), "unsealed/failed head report")
    check(type(d.get("owner_pid")) is int and d["owner_pid"] > 0, "head report owner missing")
    n = d.get("proposals", 0)
    check(type(n) is int and n >= 8, "insufficient proposal coverage")
    check(d.get("qualified_proposals") == n, "unqualified proposal")
    for k, expected in {"primary_head_calls": 5*n, "legacy_head_calls": 5*n, "checked_heads": 5*n,
                        "root_heads": n, "loop_heads": 4*n}.items():
        check(d.get(k) == expected, "head-call/branch census " + k)
    check(d.get("dispatch_ops_checked", 0) >= 10*n, "mutation guard not engaged")


def run(stage_root, run_root, arm):
    root, r = Path(stage_root), Path(run_root)
    m = stage(root)
    d = json.loads((r / "docker_inspect.json").read_text())[0]
    env = dict(v.split("=", 1) for v in d["Config"]["Env"] if "=" in v)
    expected = {"E8_ARM": arm, "E8_QUALIFY": "1", "FR13_FIX1_SELFCHECK": "1", "FR13_DRAFTER_SINGLE_LOGITS": "1" if arm == "on" else "0",
                "FR13_REPLAY_ROUTE": "1", "FR13_EAGER_PACK": "1", "FR13_TREE_CONV_FUSED": "1",
                "FR13_ATTN_KV_REMAP": "1", "FR13_SLOT_REORDER": "0", "FR13_KV_REMAP_SYNCFREE": "1",
                "E1_RECORD": "/logs/e1_events.jsonl", "FR13_SFWD_GPU_TIMER": "1",
                "FR10_ENABLE_TREE_GDN": "1", "FR10_ALLOW_LINEAR_FALLBACK": "0", "FR10_DECODE_MODE_DEFAULT": "tree_mtp",
                "FR13_TREE_RUNROW_INIT": "1", "FR13_FORCE_SPINE_COMMIT": "0", "FR13_COMMIT_ARGMAX_GATE": "0",
                "FR10_METRICS": "0", "E7A_CAPTURE_SHIM": "0", "E7B_SHIM_PATH": "",
                "E1_SHIM_PATH": "/e8/frozen/e1/e1_event_recorder_shim.py", "E1_RUNTIME_DIR": "/e8/frozen/e1"}
    for k, value in expected.items():
        check(env.get(k) == value, "loaded env " + k)
    check(d["Config"]["Image"] == m["image"], "image pin")
    spec = json.loads(env['SPEC_CONFIG'])
    wanted_tree = [(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,1), (0,0,1), (0,0,0,1), (0,0,0,0,1)]
    check(set(spec) == {'method', 'num_speculative_tokens', 'speculative_token_tree'} and spec['method'] == 'qwen3_5_mtp'
          and spec['num_speculative_tokens'] == 9 and ast.literal_eval(spec['speculative_token_tree']) == wanted_tree, "exact Cat10 draft topology")
    for flag in ('FR13_FIXED32_MODE', 'FR13_TAW', 'FR13_DM_DEPTHSYNC', 'FR13_STEP_GRAPH', 'FR13_SAMPLED_REPLAY_BATCHED',
                 'FR13_COMMITTER_NATIVE_BATCHED', 'FR13_COMMITTER_GRAPH', 'FR13_REPLAY_MULTISTREAM'):
        check(env.get(flag, '') in ('', '0'), 'unexpected route arm ' + flag)
    check(env.get('FR13_DEVICE_MULTIDRAFT', '1') == '1', 'device committer default changed')
    check(not list((r/'logs').glob('*.arm')), 'unexpected route marker')
    for flag in ('FR10_TREE_GDN_CAPTURE_PAYLOAD', 'FR10_LAYER_HIDDEN_CAPTURE', 'FR13_DECODE_GDN_CAPTURE',
                 'LUMO_MTP_DRAFT_TRACE_FILE', 'LUMO_TREE_SAMPLER_DEBUG_LOG', 'LUMO_TREE_PATH_LCP_LOG'):
        check(not env.get(flag), 'unexpected heavy capture ' + flag)
    cmd = " ".join(d["Config"]["Cmd"])
    for token in ["--seed '20260921'", "--max-num-seqs '1'", "--attention-backend 'TREE_ATTN'", "--enforce-eager", "--no-enable-prefix-caching", "--no-async-scheduling",
                  "--gpu-memory-utilization '0.6'", "--max-model-len '16384'", "vllm serve " + m['model_path'], "--served-model-name " + m['model_name']]:
        check(token in cmd, "engine argv " + token)
    for name, expected_hash in m["unchanged_loaded_modules"].items():
        check(sha(r / "loaded_backend" / name) == expected_hash, "unchanged commit/state/backend source " + name)
    sr = json.loads((r / "logs/e8_shim.json").read_text())
    check(sr["arm"] == arm and sr["qualify"] is True, "shim identity")
    check(sha(r / "loaded_backend/eagle.py") == sr["output_sha256"] == m["eagle_variants"][arm + "_qualification"], "actual emitted eagle identity")
    model = json.loads((r / "logs/served_model.json").read_text())
    check(model["path"] == m["model_path"] and model["checkpoint_identity"] == m["checkpoint_identity"], "weight directory identity")
    head = json.loads((r / "logs/e8_head_gate.json").read_text())
    head_report(head, arm)
    check(json.loads((r/'logs/e8_head_owner.json').read_text()).get('pid') == head['owner_pid'], "head receipt owner mismatch")
    if arm == "on":
        old = json.loads((r / "logs/fr13_fix1_selfcheck.json").read_text())
        new = json.loads((r / "logs/e8_head_gate.json").read_text())
        check(old.get("mismatch_steps") == 0 and old.get("steps_checked") == new["checked_heads"], "existing FIX1 report/census")
    requests = sorted((r / "cohort").glob("*/capture_request.json"))
    check(len(requests) == 8, "qualification must contain exactly eight requests")
    frozen = json.loads((root / "frozen/e1/frozen_prefixes.json").read_text())
    prefixes = {x['id']: x['prefix_sha256'] for x in frozen['pilot']}
    ids = []
    for p in requests:
        q = json.loads(p.read_text())
        check(not q.get("error") and q.get("response_id"), "API failure " + str(p))
        check(q["request"].get("return_tokens_as_token_ids") is True and q["request"].get("temperature") == 0 and q["seed"] == q['request'].get('seed') == 20260921, "API request contract")
        check(q.get('phase') == 'preflight' and q.get('max_tokens') == q['request'].get('max_tokens') == 32, "qualification phase/token budget")
        check(q.get('prefix_id') in prefixes and q.get('prefix_sha256') == q.get('prompt_sha256') == prefixes[q['prefix_id']], "frozen prompt bytes")
        check(q["response_logprobs_tokens"] and all(str(t).startswith("token_id:") for t in q["response_logprobs_tokens"]), "direct API IDs")
        ids.append(q["prefix_id"])
    check(sorted(ids) == sorted(x["id"] for x in frozen["pilot"]), "frozen prefix coverage")
    joined = json.loads((r / "join.json").read_text())
    check(not joined.get("invalid") and not joined.get("refused"), "full API/recorder binding failed")
    check((r / "join.rc").read_text().strip() == "3", "qualification join must contain no timing support")
    check(len(joined.get('token_evidence', {})) == 8 and joined.get('n_usable') == 0, "incomplete API join or leaked timed phase")
    events = [json.loads(line) for line in (r/'logs/e1_events.jsonl').read_text().splitlines() if line.strip()]
    pure_ids = {row['request_id'] for e in events if e.get('event') == 'output_rows' and e.get('kind') == 'pure' for row in e['rows']}
    check(pure_ids == set(joined['token_evidence']), "some frozen request has no pure-decode qualification exposure")
    # rc3 is only allowed for the all-UNTIMED phase manifest; it provides no
    # timing support. The complete API sequence and closing seal are still checked.
    return {"arm": arm, "status": "PASS", "untimed": True,
            "manifest_sha256": sha(root / "manifest.json"),
            "evidence_sha256": {str(p.relative_to(r)): sha(p) for p in sorted(r.rglob('*')) if p.is_file()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--run")
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--out")
    a = ap.parse_args()
    result = run(a.stage, a.run, a.arm) if a.run else {"status": "STAGE_HASHES_PASS", "files": len(stage(a.stage)["files"])}
    if a.out:
        p = Path(a.out)
        check(not p.exists(), "receipt already exists")
        p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence_sha256'}))


if __name__ == '__main__':
    main()
