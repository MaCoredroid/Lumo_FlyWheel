"""FR13 garble per-layer LADDER driver (same-boot, chat-based for cat9 fr13_launch_locked).

Definitive top-down per-layer gate to localize the node0 verify corruption at the deterministic matrix garble.
Captures are SERVER-SIDE via boot env:
  LIVE  : FR10_LAYER_HIDDEN_CAPTURE (ROWS=0 node0 root, LIMIT=N, SKIP=0) -> per-layer node0 hidden + FR13_FINAL_LOGIT
          for the first N tree-verify forwards of the matrix_build greedy generation.
  CLEAN : FR10_ROOT_HIDDEN_CAPTURE -> per-layer last-row hidden of a teacher-forced prefix (through 'expected').
Diff LIVE(node0 at the garble forward) vs CLEAN(root) per-layer => FIRST divergent layer = the corrupt op.

cat9 exposes only /v1/chat/completions (not /v1/completions), so LIVE = chat matrix_build greedy; CLEAN = chat
assistant-prefix continue_final_message (same as the proven teacher-force).
"""
import sys, json, time, urllib.request

sys.path.insert(0, "scripts")
from fr13_garble_gate import PROMPTS, score_sample  # noqa

EP = "http://127.0.0.1:9950"


def post(path, body, t=300):
    r = urllib.request.Request(EP + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as resp:
        b = resp.read().decode()
    return json.loads(b) if b else None


def wait_health(max_s=600):
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            with urllib.request.urlopen(EP + "/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[health] ready {time.time()-t0:.0f}s", flush=True); return True
        except Exception:
            pass
        time.sleep(6)
    return False


def model_id():
    try:
        return json.loads(urllib.request.urlopen(EP + "/v1/models", timeout=10).read())["data"][0]["id"]
    except Exception:
        return "qwen3.6-27b"


def main():
    if not wait_health():
        print("NEVER HEALTHY", flush=True); sys.exit(2)
    M = model_id()
    name, content = [(n, c) for (n, c) in PROMPTS if n == "matrix_build"][0]
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass

    # ---- PHASE LIVE: matrix_build greedy; server captures node0 per-layer (FR10_LAYER_HIDDEN, ROWS=0) ----
    d = post("/v1/chat/completions", {"model": M,
             "messages": [{"role": "user", "content": content}], "max_tokens": 700,
             "temperature": 0.0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}})
    text = d["choices"][0]["message"]["content"]
    sc = score_sample(text)
    print(f"[live] undefined={sc['undefined']} names={sc['undefined_names'][:6]}", flush=True)
    # prefix through '(expected' (right before the garbled '_rows')
    marker = "computed_slice_shape = (expected"
    idx = text.find(marker)
    prefix = text[:idx + len(marker)] if idx >= 0 else text[:400]
    print(f"[live] prefix ends: ...{prefix[-40:]!r}", flush=True)

    # ---- PHASE CLEAN: teacher-force the prefix; server captures root per-layer (FR10_ROOT_HIDDEN) ----
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass
    body = {"model": M, "messages": [{"role": "user", "content": content},
                                     {"role": "assistant", "content": prefix}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
            "add_generation_prompt": False, "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    dc = post("/v1/chat/completions", body)
    lp = dc["choices"][0]["logprobs"]["content"][0]
    print(f"[clean] argmax after (expected: {lp['token']!r} lp={lp['logprob']:.3f}", flush=True)
    print("[clean] top5:", [(c["token"], round(c["logprob"], 2)) for c in lp["top_logprobs"][:5]], flush=True)

    out = {"live_undefined": sc["undefined_names"], "live_text": text, "clean_argmax": lp["token"],
           "clean_top": [(c["token"], c["logprob"]) for c in lp["top_logprobs"][:20]], "prefix_tail": prefix[-60:]}
    with open("output/fr13_garble_ladder/drive_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote output/fr13_garble_ladder/drive_result.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
