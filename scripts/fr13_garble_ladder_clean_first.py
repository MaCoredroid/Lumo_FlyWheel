"""FR13 garble ladder: CLEAN-first driver so node0 CLEAN (teacher-force) is captured as call0, then LIVE.

Same-boot per-layer gate: send the teacher-force FIRST (its tree-verify node0 = CLEAN pos190 pred '_row',
captured as FR10_LAYER_HIDDEN call0), THEN drive matrix_build greedy (LIVE, garble node0 '_rows' at a later
call). Diff call0 (clean) vs the garble call per-layer => first divergent layer = corrupt op.
Prefix comes from the saved deterministic live_text (output/fr13_garble_ladder/live_text.txt).
"""
import sys, json, time, urllib.request
sys.path.insert(0, "scripts")
from fr13_garble_gate import PROMPTS, score_sample  # noqa
EP = "http://127.0.0.1:9950"


def post(p, b, t=300):
    r = urllib.request.Request(EP + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as resp:
        s = resp.read().decode()
    return json.loads(s) if s else None


def wait_health(m=600):
    t0 = time.time()
    while time.time() - t0 < m:
        try:
            with urllib.request.urlopen(EP + "/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[health] {time.time()-t0:.0f}s", flush=True); return True
        except Exception:
            pass
        time.sleep(6)
    return False


def main():
    if not wait_health():
        print("NEVER HEALTHY", flush=True); sys.exit(2)
    M = json.loads(urllib.request.urlopen(EP + "/v1/models", timeout=10).read())["data"][0]["id"]
    name, content = [(n, c) for (n, c) in PROMPTS if n == "matrix_build"][0]
    live_text = open("output/fr13_garble_ladder/live_text.txt").read()
    marker = "computed_slice_shape = (expected"
    prefix = live_text[:live_text.find(marker) + len(marker)]
    print(f"[prefix] ends ...{prefix[-40:]!r}", flush=True)
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass

    # ---- CLEAN FIRST: teacher-force -> node0 CLEAN captured as call0 ----
    body = {"model": M, "messages": [{"role": "user", "content": content},
                                     {"role": "assistant", "content": prefix}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
            "add_generation_prompt": False, "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    dc = post("/v1/chat/completions", body)
    lp = dc["choices"][0]["logprobs"]["content"][0]
    print(f"[CLEAN call0] argmax {lp['token']!r} lp={lp['logprob']:.3f}", flush=True)

    # ---- LIVE: matrix greedy -> garble node0 captured as a later call ----
    d = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": content}],
             "max_tokens": 700, "temperature": 0.0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}})
    sc = score_sample(d["choices"][0]["message"]["content"])
    print(f"[LIVE] undefined={sc['undefined']} names={sc['undefined_names'][:4]}", flush=True)
    print("DONE - node0 captures: call0=CLEAN, later call=garble; run the per-layer diff analysis.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
