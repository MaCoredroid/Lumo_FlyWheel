"""Branch-specificity check for the DETERMINISTIC matrix-greedy garble.

Waits for health, sends the matrix_build prompt at temp 0.0 (greedy), scores undefined names, then the
caller correlates the garbled token position to the winner log's winner_spine (0 = spine commit, >0 = branch
commit). If the garble commits on the SPINE, branch-specificity is REFUTED for this repro.
"""
import sys, time, json, urllib.request

sys.path.insert(0, "scripts")
from fr13_garble_gate import PROMPTS, score_sample  # noqa

EP = "http://127.0.0.1:9957"


def post(path, body, timeout=120):
    req = urllib.request.Request(EP + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(max_s=540):
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            with urllib.request.urlopen(EP + "/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[health] ready after {time.time()-t0:.0f}s", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(6)
    return False


def main():
    if not wait_health():
        print("SERVER NEVER HEALTHY", flush=True); sys.exit(2)
    model = post("/v1/models", None) if False else None
    try:
        model = urllib.request.urlopen(EP + "/v1/models", timeout=10)
        model = json.loads(model.read())["data"][0]["id"]
    except Exception:
        model = "default"
    name, content = [(n, c) for (n, c) in PROMPTS if n == "matrix_build"][0]
    # GREEDY (temp 0) -> deterministic garble; fixed seed for reproducibility
    d = post("/v1/chat/completions", {"model": model,
             "messages": [{"role": "user", "content": content}], "max_tokens": 700,
             "temperature": 0.0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}, 300)
    text = d["choices"][0]["message"]["content"]
    sc = score_sample(text)
    print(f"=== matrix_build GREEDY: undefined={sc['undefined']} syntax_err={sc['syntax_error']} "
          f"names={sc['undefined_names'][:6]} ===", flush=True)
    print("---- generated code ----", flush=True)
    print(text, flush=True)
    with open("output/fr13_winnerspine/gen.json", "w") as f:
        json.dump({"text": text, "score": sc}, f, indent=2)


if __name__ == "__main__":
    main()
