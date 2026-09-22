#!/usr/bin/env python3
"""Read-only CPU review of sealed E8 timing artifacts; never launches workloads.

Reconstructs the ledger, API-bound support, exclusions, ownership and statistics
independently, then compares them with the frozen implementation's receipts.
An incomplete run produces only per-cell findings, never a partial estimate.
"""
import argparse
import ast
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

TIMING_SHA = "8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1"
QUAL_SHA = "bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030"
QUAL_PASS_SHA = "815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def verify_hashes(root, mapping):
    for name, expected in mapping.items():
        p = root / name
        need(p.resolve().is_relative_to(root.resolve()), "escaping artifact " + name)
        need(p.is_file() and digest(p) == expected, "hash mismatch " + str(p))


def source_and_qualification(root):
    stage = root / "stage"
    need(digest(stage / "timing_manifest.json") == TIMING_SHA, "unexpected timing freeze")
    m = read(stage / "timing_manifest.json")
    verify_hashes(stage, m["files"])
    need([c["arm"] for c in m["cells"]] == ["off", "on", "on", "off", "off", "on"], "cell schedule")
    need(digest(stage / "qualification-source/manifest.json") == QUAL_SHA, "qualified source")
    qr = root / "qualification_evidence"
    need(not (qr / "FAILED.json").exists(), "qualification failed")
    need(digest(qr / "QUALIFICATION_PASS.json") == QUAL_PASS_SHA, "qualification pass changed")
    final = read(qr / "QUALIFICATION_PASS.json")
    need(final["arms"] == ["on", "off"] and final["stage_manifest_sha256"] == QUAL_SHA, "qualification arms/source")
    need(digest(qr / "stage/manifest.json") == QUAL_SHA, "qualification snapshot")
    verify_hashes(qr / "stage", read(qr / "stage/manifest.json")["files"])
    for arm in ("on", "off"):
        receipt = qr / ("qualification_" + arm + ".json")
        need(digest(receipt) == final["receipts"][arm], "qualification receipt " + arm)
        verify_hashes(qr / ("qualification_" + arm), read(receipt)["evidence_sha256"])
    started = read(root / "CAMPAIGN_STARTED.json")
    need(started["timing_manifest_sha256"] == TIMING_SHA and started["cells"] == m["cells"], "campaign start")
    need(started["qualification"]["pass_sha256"] == QUAL_PASS_SHA, "campaign qualification")
    # Re-run the real, source-verified admission gate as a separate cross-check.
    sys.path.insert(0, str(stage))
    import timing_verify
    timing_verify.qualification(stage, qr)
    return m, timing_verify


def ownership(r, inspected):
    receipt = read(r / "ownership_receipt.json")
    samples = [json.loads(x) for x in (r / "ownership_samples.jsonl").read_text().splitlines()]
    need(receipt["status"] == "PASS" and len(samples) >= 2, "ownership receipt")
    cid = inspected["Id"]
    run_id = inspected["Config"]["Labels"]["lumo.e8.run"]
    need(receipt["container_id"] == cid and receipt["run_id"] == run_id, "ownership binding")
    hosts = set()
    for i, sample in enumerate(samples):
        need(sample["status"] == "PASS", "ownership failure sample")
        raw = sample["raw"]
        need(set(raw) == {"inspect", "containers", "gpu", "top"}, "ownership queries")
        need(all(x["returncode"] == 0 for x in raw.values()), "ownership query failure")
        d = json.loads(raw["inspect"]["stdout"])[0]
        need(d["Id"] == cid and d["State"]["Running"] and d["Config"]["Labels"]["lumo.e8.run"] == run_id, "ownership sample identity")
        need(set(raw["containers"]["stdout"].split()) == {cid}, "foreign container")
        gpu = {int(x) for x in raw["gpu"]["stdout"].split()}
        lines = raw["top"]["stdout"].splitlines()
        need(lines[0].strip() == "PID", "host PID header")
        owned = {int(x.strip()) for x in lines[1:] if x.strip()}
        need(gpu and gpu <= owned, "foreign/missing GPU host PID")
        hosts.update(gpu)
        need(sample["end_monotonic"] >= sample["start_monotonic"], "sample duration")
        if i:
            need(sample["start_monotonic"] >= samples[i-1]["end_monotonic"], "sample order")
    need(samples[0]["end_monotonic"] <= receipt["workload_start_monotonic"] <= receipt["workload_end_monotonic"] <= samples[-1]["start_monotonic"], "workload bracketing")
    return {"samples": len(samples), "gpu_host_pids": sorted(hosts),
            "max_gap_seconds": max(b["start_monotonic"]-a["end_monotonic"] for a, b in zip(samples, samples[1:])),
            "workload_start_monotonic": receipt["workload_start_monotonic"],
            "workload_end_monotonic": receipt["workload_end_monotonic"],
            "visibility_limit": "occasional host process samples; not continuous isolation proof"}


def raw_cell(root, stage, manifest, cell, frozen_verifier):
    r = root / cell["name"]
    seal = read(r / "cell_result.json")
    need(seal["terminal_seal"] is True and seal["cell"] == cell, "terminal cell identity")
    need(seal["timing_manifest_sha256"] == TIMING_SHA and seal["qualification_pass_sha256"] == QUAL_PASS_SHA, "terminal source binding")
    need(not (r / "FAILED.json").exists(), "cell failed")
    actual = {str(p.relative_to(r)): digest(p) for p in r.rglob("*") if p.is_file() and p.name != "cell_result.json"}
    need(actual == seal["evidence_sha256"], "sealed evidence set/hash changed")
    reference = frozen_verifier.sealed_cell(stage, r, cell, QUAL_PASS_SHA)
    inspected = read(r / "docker_inspect.json")[0]
    env = dict(x.split("=", 1) for x in inspected["Config"]["Env"] if "=" in x)
    arm = cell["arm"]
    need(env["E8_ARM"] == arm and env["E8_QUALIFY"] == "0" and env["FR13_FIX1_SELFCHECK"] == "0", "clean arm flags")
    need([env[k] for k in ("FR13_ATTN_KV_REMAP", "FR13_SLOT_REORDER", "FR13_KV_REMAP_SYNCFREE")] == ["1", "0", "1"], "KV policy")
    need(env["FR13_TREE_RUNROW_INIT"] == env["FR13_REPLAY_ROUTE"] == env["FR13_EAGER_PACK"] == env["FR13_TREE_CONV_FUSED"] == "1", "route flags")
    cmd = " ".join(inspected["Config"]["Cmd"])
    need(all(x in cmd for x in ("--seed '20260921'", "--max-num-seqs '1'", "--attention-backend 'TREE_ATTN'", "--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching")), "actual engine settings")
    spec = json.loads(env["SPEC_CONFIG"])
    need(spec["num_speculative_tokens"] == 9 and len(ast.literal_eval(spec["speculative_token_tree"])) == 9, "draft topology")
    stopped = read(r / "docker_inspect_final.json")[0]["State"]
    need(not stopped["Running"] and not stopped["OOMKilled"] and stopped["ExitCode"] == 0, "unclean container exit")
    own = ownership(r, inspected)
    head = read(r / "logs/e8_head_gate.json")
    proposals = head["proposals"]
    need(head["closed"] and not head["failures"] and head["qualify"] is False and head["arm"] == arm, "head seal")
    need(head["primary_head_calls"] == 5*proposals and head["legacy_head_calls"] == (0 if arm == "on" else 5*proposals), "head census")
    need(all(head.get(k, 0) == 0 for k in ("qualified_proposals", "checked_heads", "root_heads", "loop_heads", "dispatch_ops_checked")), "qualification instrumentation active")
    need(not (r / "logs/fr13_fix1_selfcheck.json").exists(), "unexpected selfcheck")

    events = [json.loads(x) for x in (r / "logs/e1_events.jsonl").read_text().splitlines() if x.strip()]
    need(not (r / "logs/e1_recorder_FAILED.flag").exists(), "recorder fail flag")
    need([e["n"] for e in events] == list(range(1, len(events)+1)), "event continuity")
    need(events[0]["event"] == "recorder_probe" and events[-1]["event"] == "run_close", "recorder boundaries")
    need(not events[-1]["sink_failed"] and all(e.get("errors", 0) == e.get("sink_failures", 0) == 0 and e["event"] != "error" for e in events), "recorder error")
    need(events[0]["pid"] == head["owner_pid"] == read(r / "logs/e8_head_owner.json")["pid"], "head/recorder ownership")
    forwards, physical, outputs, breaks = {}, {}, {}, []
    for e in events:
        if e["event"] in ("forward_entry", "physical_step", "output_rows"):
            target = {"forward_entry": forwards, "physical_step": physical, "output_rows": outputs}[e["event"]]
            key = e["physical_step_id"] if e["event"] == "physical_step" else e["seq"]
            need(key not in target, "duplicate event identity")
            target[key] = e
        elif e["event"] == "chain_break":
            breaks.append(e["seq"])
    need(sorted(forwards) == list(range(1, len(forwards)+1)) and set(outputs) == set(forwards), "complete forward/output ledger")
    close_event = events[-1]
    need((close_event["n_events"], close_event["n_forwards"], close_event["n_physical_steps"], close_event["n_output_records"]) == (len(events), len(forwards), len(physical), len(outputs)), "close counts")
    need(proposals == len(forwards), "proposal/forward census")
    per_request = collections.defaultdict(list)
    discarded_empty = []
    for seq, out in sorted(outputs.items()):
        need(out["num_reqs"] == forwards[seq]["num_reqs"] == len(out["rows"]) == 1, "B1 output row")
        row = out["rows"][0]
        rid = row["request_id"]
        need(row["row"] == 0 and row["step_idx"] == len(per_request[rid]) and row["n_emitted"] == len(row["emitted_ids"]), "request/row sequence")
        per_request[rid].append((seq, row))
        if out["kind"] == "pure":
            ps = physical[out["physical_step_id"]]
            need(ps["seq"] == seq and ps["request_ids"] == [rid] and ps["num_reqs"] == 1, "pure row binding")
            need(row["num_draft_tokens"] == 9 and forwards[seq]["num_tokens"] == 10, "actual Cat10 occupancy")
        else:
            need(out["kind"] == "nonpure", "unexpected output kind")
        if row["discarded"] and not row["emitted_ids"]:
            discarded_empty.append(seq)
    need(all(outputs[p["seq"]]["kind"] == "pure" for p in physical.values()), "orphan physical interval")

    pilot = read(stage / "qualification-source/frozen/e1/frozen_prefixes.json")["pilot"]
    pool = {p["id"]: p for p in read(stage / "qualification-source/frozen/e1/prefix_pool.json")["prefixes"]}
    amap = read(r / "api_tokens.json")
    requests = [read(p) for p in (r / "cohort").glob("*/capture_request.json")]
    need(len(requests) == 10, "ten API requests")
    slots = {q["slot"]: q for q in requests}
    order = ["w0", "w1"] + ["t" + str(i) for i in range(8)]
    need(set(slots) == set(order), "workload slots")
    phases = {"warmup": set(), "timed": set()}
    prefixes, bound, streams, coverage, clips, finishes = {}, {}, {}, {}, {}, {}
    for i, slot in enumerate(order):
        q = slots[slot]
        fp = pilot[0 if i < 2 else i-2]
        phase, budget = ("warmup", 32) if i < 2 else ("timed", 128)
        need(not q.get("error") and q["phase"] == phase and q["prefix_id"] == fp["id"], "API phase/prefix")
        need(q["prefix_sha256"] == q["prompt_sha256"] == fp["prefix_sha256"], "prompt identity")
        need(q["max_tokens"] == q["request"]["max_tokens"] == budget and q["seed"] == q["request"]["seed"] == 20260921, "request settings")
        need(q["request"]["temperature"] == 0 and q["request"]["return_tokens_as_token_ids"] is True, "API greedy direct IDs")
        prompt = pool[fp["id"]]["text"]
        need(hashlib.sha256(prompt.encode()).hexdigest() == fp["prefix_sha256"], "frozen prompt bytes")
        body = {"model": manifest["model_name"], "prompt": prompt, "max_tokens": budget, "temperature": 0,
                "seed": 20260921, "logprobs": 1, "echo": False, "return_tokens_as_token_ids": True}
        need(q["request"] == {k: v for k, v in body.items() if k != "prompt"}, "exact request parameters")
        need(q["request_body_sha256"] == hashlib.sha256(json.dumps(body).encode()).hexdigest(), "actual request body identity")
        need(q["t_end_perf"] >= q["t_start_perf"] and (i == 0 or q["t_start_perf"] >= slots[order[i-1]]["t_end_perf"]), "sequential requests")
        need(own["workload_start_monotonic"] <= q["t_start_perf"] <= q["t_end_perf"] <= own["workload_end_monotonic"], "request outside sampled workload")
        matches = [rid for rid in amap if rid == q["response_id"] or rid.startswith(q["response_id"] + "-")]
        need(len(matches) == 1, "unique API-engine binding")
        rid = matches[0]
        need(rid not in phases["warmup"] | phases["timed"], "reused engine ID")
        phases[phase].add(rid)
        need(all(str(x).startswith("token_id:") for x in q["response_logprobs_tokens"]), "direct token-ID domain")
        ids = [int(x.split(":", 1)[1]) for x in q["response_logprobs_tokens"]]
        need(ids == amap[rid] and 0 < len(ids) <= budget, "complete API sequence/budget")
        rows = per_request[rid]
        ledger = [token for _, row in rows for token in row["emitted_ids"]]
        need(ledger[:len(ids)] == ids and len(ledger) >= len(ids), "API/ledger token equality")
        need(len(ledger)-len(rows[-1][1]["emitted_ids"]) <= len(ids), "nonterminal token clipping")
        remaining = len(ids)
        for seq, row in rows:
            n = min(remaining, len(row["emitted_ids"]))
            bound[seq, rid] = n
            remaining -= n
        clips[rid] = len(ledger)-len(ids)
        finishes[slot] = {"prefix": fp["id"], "phase": phase, "api_tokens": len(ids), "structural_tokens": len(ledger), "terminal_clip": clips[rid], "finish_reason": q["finish_reason"]}
        if phase == "timed":
            prefixes[rid] = fp["id"]
            streams[fp["id"]] = ids
            coverage[fp["id"]] = {"intervals": 0, "wall_s": 0.0, "api_bound_tokens": 0}
    need(set(amap) == set(per_request) == phases["warmup"] | phases["timed"], "full warmup and timed coverage")
    jm = read(r / "join_manifest.json")
    need(all(set(jm["phases"][p]) == phases[p] for p in phases), "join phases")
    retained, excluded = [], {}
    for pid, ps in sorted(physical.items()):
        nxt = physical.get(pid+1)
        wall, reason = None, None
        if nxt is None:
            reason = "terminal: no successor physical step id+1"
        else:
            wall = nxt["t_start"] - ps["t_start"]
            rid = ps["request_ids"][0]
            if rid in phases["warmup"]:
                reason = "warmup phase (prespecified exclusion)"
            elif nxt["seq"] != ps["seq"]+1:
                reason = "mixed/prefill forward between (seq gap): not bridged"
            elif any(ps["seq"] <= x <= nxt["seq"] for x in breaks):
                reason = "chain_break between this step and its successor"
            elif nxt["request_ids"] != ps["request_ids"]:
                reason = "cohort change at the successor (ramp/drain)"
            elif any(row["discarded"] for row in outputs[ps["seq"]]["rows"]):
                reason = "discarded row(s) in this step"
        if reason:
            excluded[str(pid)] = {"reason": reason, "wall_s": wall}
            continue
        need(wall > 0, "nonpositive retained wall")
        row = outputs[ps["seq"]]["rows"][0]
        rid = row["request_id"]
        need(rid in prefixes, "retained untimed request")
        cov = coverage[prefixes[rid]]
        cov["intervals"] += 1
        cov["wall_s"] += wall
        cov["api_bound_tokens"] += bound[ps["seq"], rid]
        retained.append({"physical_step_id": pid, "seq": ps["seq"], "wall_s": wall, "num_reqs": 1, "rows": [row], "over_cap": wall > 1.5})
    joined = read(r / "join.json")
    need(joined["invalid"] is None and joined["refused"] is None, "join invalid")
    need(joined["steps"] == retained and joined["excluded"] == excluded, "independent support/exclusions differ")
    need(joined.get("truncated_tail_tokens", {}) == {rid: n for rid, n in clips.items() if n}, "terminal clip report")
    tokens = sum(x["api_bound_tokens"] for x in coverage.values())
    wall = sum(x["wall_s"] for x in coverage.values())
    floor = len(retained) >= 8 and all(x["intervals"] >= 1 for x in coverage.values())
    rate = tokens/wall if floor and wall > 0 else None
    need(tokens == joined["sum_emitted_tokens_api_bound_pure_support"] and close(wall, joined["sum_wall_s_unique_physical_steps"]), "join numerator/denominator")
    need(tokens == reference["api_bound_tokens"] and close(wall, reference["unique_wall_s"]) and len(retained) == reference["n_intervals"], "sealed support totals")
    need(coverage == reference["per_prefix"] and streams == reference["streams"], "sealed per-prefix evidence")
    need(reference["status"] == ("VALID" if floor else "INSUFFICIENT_SUPPORT"), "support status")
    need((rate is None and reference["rate"] is None) or close(rate, reference["rate"]), "sealed rate")
    return {"status": reference["status"], "cell": cell, "rate": rate, "n_intervals": len(retained), "api_bound_tokens": tokens,
            "unique_wall_s": wall, "per_prefix": coverage, "exclusion_reasons": dict(collections.Counter(x["reason"] for x in excluded.values())),
            "over_cap_retained": sum(x["over_cap"] for x in retained), "requests": finishes, "streams": streams,
            "proposals": proposals, "primary_head_calls": head["primary_head_calls"], "legacy_head_calls": head["legacy_head_calls"],
            "empty_discarded_rows": len(discarded_empty), "ownership": own,
            "receipt_files": len(actual), "terminal_receipt_sha256": digest(r / "cell_result.json")}


def statistics(cells):
    pairs = []
    for b in (1, 2, 3):
        off, on = cells[b, "off"]["rate"], cells[b, "on"]["rate"]
        pairs.append((on-off)/off)
    rng = random.Random(20260921)
    draws = []
    for _ in range(10000):
        indices = [rng.randrange(3) for _ in range(3)]
        draws.append(sum(pairs[i] for i in indices)/3)
    draws.sort()
    def quantile(p):
        position = 9999*p
        lower = int(math.floor(position))
        return draws[lower] + (draws[math.ceil(position)]-draws[lower])*(position-lower)
    return {"paired_relative_differences": pairs, "primary_mean_paired_relative_difference": sum(pairs)/3,
            "primary_95pct_percentile_interval": [quantile(.025), quantile(.975)],
            "diagnostic_ratio_of_arm_means_minus_one": sum(cells[b, "on"]["rate"] for b in (1, 2, 3))/sum(cells[b, "off"]["rate"] for b in (1, 2, 3))-1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    m, verifier = source_and_qualification(root)
    need({p.name for p in root.glob("cell_*") if p.is_dir()} <= {c["name"] for c in m["cells"]}, "unplanned cell directory")
    checked, cells, missing = {}, {}, []
    for cell in m["cells"]:
        if not (root / cell["name"] / "cell_result.json").exists():
            missing.append(cell["name"])
            continue
        result = raw_cell(root, root/"stage", m, cell, verifier)
        checked[cell["name"]] = result
        cells[cell["block"], cell["arm"]] = result
    report = {"status": "SEALED_CELLS_PASS_AGGREGATE_PENDING", "run": str(root), "timing_manifest_sha256": TIMING_SHA,
              "qualification_pass_sha256": QUAL_PASS_SHA, "cells": checked, "missing_cells": missing,
              "primary": None, "script_sha256": digest(__file__)}
    if not checked:
        report["status"] = "NO_SEALED_CELL_YET"
    if (root / "FAILED.json").exists():
        report.update(status="FAILED_CAMPAIGN_NO_PRIMARY", campaign_failure=read(root / "FAILED.json"))
    if not missing and (root / "aggregate.json").exists() and (root / "CAMPAIGN_COMPLETE.json").exists():
        need(not (root / "FAILED.json").exists(), "campaign failed")
        agg = read(root / "aggregate.json")
        done = read(root / "CAMPAIGN_COMPLETE.json")
        need(done["status"] == "COMPLETE" and done["timing_manifest_sha256"] == TIMING_SHA and done["aggregate_sha256"] == digest(root / "aggregate.json"), "campaign completion seal")
        valid = all(c["status"] == "VALID" for c in cells.values())
        primary = statistics(cells) if valid else None
        if primary:
            for key in ("primary_mean_paired_relative_difference", "diagnostic_ratio_of_arm_means_minus_one"):
                need(close(primary[key], agg[key]), "aggregate " + key)
            need(all(close(a, b) for a, b in zip(primary["primary_95pct_percentile_interval"], agg["primary_95pct_percentile_interval"])), "aggregate bootstrap interval")
        else:
            need(agg["primary_mean_paired_relative_difference"] is None and agg["primary_95pct_percentile_interval"] is None, "unsupported aggregate")
        comparisons = []
        cross = [((b, "on"), (b, "off"), "within_pair_across_arms") for b in (1, 2, 3)]
        within = [((a, arm), (b, arm), "across_blocks_within_arm") for arm in ("on", "off") for a, b in itertools.combinations((1, 2, 3), 2)]
        for a, b, kind in cross+within:
            for prefix, stream in sorted(cells[a]["streams"].items()):
                other = cells[b]["streams"][prefix]
                n = min(len(stream), len(other))
                first = next((i for i in range(n) if stream[i] != other[i]), None)
                if first is None and len(stream) != len(other):
                    first = n
                comparisons.append({"kind": kind, "left": list(a), "right": list(b), "prefix": prefix, "equal": stream == other,
                                    "left_tokens": len(stream), "right_tokens": len(other), "first_divergence_zero_based": first,
                                    "common_prefix_tokens": n if first is None else first})
        need(comparisons == agg["stream_diagnostic"]["comparisons"] and len(comparisons) == 72, "stream diagnostic")
        report.update(status="COMPLETE_INDEPENDENT_REVIEW_PASS", primary=primary,
                      stream_comparisons={"n": len(comparisons), "equal": sum(x["equal"] for x in comparisons),
                                          "by_kind": {kind: {"n": sum(x["kind"] == kind for x in comparisons), "equal": sum(x["kind"] == kind and x["equal"] for x in comparisons)} for kind in ("within_pair_across_arms", "across_blocks_within_arm")}},
                      aggregate_sha256=digest(root / "aggregate.json"), completion_sha256=digest(root / "CAMPAIGN_COMPLETE.json"))
    need(not args.output.exists(), "review output already exists")
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "cells": len(checked), "missing": len(missing), "output": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
