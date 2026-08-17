#!/usr/bin/env python3
"""Reduce FR14 ablation arm A step 2: their sglang chain running our SWE agent traffic."""
import json
import re
import sys
from pathlib import Path

OUT = Path("/home/mark/shared/tmp-scratch/fr14_ablation_a/step2")

DEC = re.compile(
    r"^\[(?P<ts>[^\]]+)\] Decode batch, #running-req: (?P<run>\d+), .*?"
    r"accept len: (?P<acc>[0-9.]+), accept rate: (?P<rate>[0-9.]+), .*?"
    r"gen throughput \(token/s\): (?P<tps>[0-9.]+)"
)


def decode_lines(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = DEC.search(line)
        if m:
            rows.append(
                {
                    "ts": m.group("ts"),
                    "running": int(m.group("run")),
                    "accept": float(m.group("acc")),
                    "rate": float(m.group("rate")),
                    "tps": float(m.group("tps")),
                }
            )
    return rows


def sglang_metrics(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"(sglang:[a-z_0-9]+)\{([^}]*)\}\s+([0-9.e+-]+)", line)
        if m:
            key = m.group(1)
            labels = m.group(2)
            stream = "stream" if 'is_streaming="true"' in labels else (
                "nostream" if 'is_streaming="false"' in labels else "")
            d[f"{key}|{stream}" if stream else key] = float(m.group(3))
    return d


def main():
    rows = decode_lines(OUT / "sglang_live.log")
    full = decode_lines(OUT / "sglang_full.log")
    if len(full) > len(rows):
        rows = full
    res = {"schema": "fr14.ablation_a.step2.v1", "decode_samples": len(rows)}
    if rows:
        acc = [r["accept"] for r in rows]
        tps = [r["tps"] for r in rows]
        res["accept_mean"] = round(sum(acc) / len(acc), 4)
        res["accept_min"] = min(acc)
        res["accept_max"] = max(acc)
        res["gen_tps_mean"] = round(sum(tps) / len(tps), 3)
        s = sorted(tps)
        res["gen_tps_median"] = s[len(s) // 2]
        res["gen_tps_p10"] = s[int(len(s) * 0.10)]
        res["gen_tps_p90"] = s[int(len(s) * 0.90)]
        res["running_req_max"] = max(r["running"] for r in rows)
        res["decode_window_first_ts"] = rows[0]["ts"]
        res["decode_window_last_ts"] = rows[-1]["ts"]

    pre = sglang_metrics(OUT / "sglang_metrics_pre.txt")
    post = sglang_metrics(OUT / "sglang_metrics_post.txt")
    if not post:
        post = sglang_metrics(OUT / "sglang_metrics_now.txt")
    if pre and post:
        keys = set(pre) | set(post)
        delta = {k: post.get(k, 0.0) - pre.get(k, 0.0) for k in sorted(keys)
                 if k.startswith(("sglang:generation_tokens_total",
                                  "sglang:prompt_tokens_total",
                                  "sglang:num_requests_total"))}
        res["sglang_counter_delta"] = delta
        gen = sum(v for k, v in delta.items() if k.startswith("sglang:generation_tokens_total"))
        pro = sum(v for k, v in delta.items() if k.startswith("sglang:prompt_tokens_total"))
        res["output_tokens_total"] = gen
        res["prompt_tokens_total"] = pro

    # per-task: runner_metadata.json (agent wall) + eval report + sglang bracket
    import re as _re

    def _sg(path):
        d = {}
        if not path.exists():
            return d
        for l in path.read_text(errors="replace").splitlines():
            m = _re.match(r'(sglang:[a-z_0-9]+)\{([^}]*)\}\s+([0-9.e+-]+)', l)
            if m:
                lab = m.group(2)
                suf = "|s" if 'is_streaming="true"' in lab else ("|n" if 'is_streaming="false"' in lab else "")
                d[m.group(1) + suf] = float(m.group(3))
        return d

    tasks = []
    for md in sorted((OUT / "swe_out").rglob("runner_metadata.json")):
        try:
            s_ = json.loads(md.read_text())
        except Exception:
            continue
        td = md.parent
        agent = s_.get("agent") or {}
        ev = {}
        erp = td / "eval" / "eval_report.json"
        if erp.exists():
            try:
                ev = json.loads(erp.read_text())
            except Exception:
                ev = {}
        pre = _sg(td / "vllm_metrics_pre.txt")
        post = _sg(td / "vllm_metrics_post.txt")
        gen = sum(post.get(k, 0.0) - pre.get(k, 0.0) for k in
                  ("sglang:generation_tokens_total|s", "sglang:generation_tokens_total|n"))
        pro = sum(post.get(k, 0.0) - pre.get(k, 0.0) for k in
                  ("sglang:prompt_tokens_total|s", "sglang:prompt_tokens_total|n"))
        nreq = sum(post.get(k, 0.0) - pre.get(k, 0.0) for k in
                   ("sglang:num_requests_total|s", "sglang:num_requests_total|n"))
        wall = agent.get("elapsed_s")
        patch = td / "patch.diff"
        tasks.append({
            "instance_id": s_.get("instance_id"),
            "agent_wall_s": wall,
            "agent_exit_code": agent.get("exit_code"),
            "timed_out": agent.get("timed_out"),
            "output_tokens": gen,
            "prompt_tokens": pro,
            "model_requests": nreq,
            "e2e_output_tps": round(gen / wall, 3) if wall else None,
            "patch_bytes": patch.stat().st_size if patch.exists() else 0,
            "verdict": ev.get("verdict"),
            "eval_wall_s": ev.get("eval_wall_clock_seconds"),
        })
    res["per_task"] = tasks
    if tasks:
        walls = [t["agent_wall_s"] for t in tasks if t["agent_wall_s"]]
        gens = [t["output_tokens"] for t in tasks if t["output_tokens"]]
        if walls:
            res["agent_wall_total_s"] = round(sum(walls), 1)
        if gens:
            res["output_tokens_task_sum"] = sum(gens)
            res["e2e_output_tps_overall"] = round(sum(gens) / sum(walls), 3)
        res["resolved"] = sum(1 for t in tasks if t.get("verdict") == "resolved")
        res["n_tasks"] = len(tasks)
    started = (OUT / "swe_started_at.txt")
    ended = (OUT / "swe_ended_at.txt")
    if started.exists():
        res["started_at"] = started.read_text().strip()
    if ended.exists():
        res["ended_at"] = ended.read_text().strip()
    print(json.dumps(res, indent=2))
    (OUT / "step2_reduced.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
