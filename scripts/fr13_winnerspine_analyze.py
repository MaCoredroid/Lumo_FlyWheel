"""Correlate the deterministic matrix-greedy garble to the winner log's per-step routing.

Reads gen.json (committed text) + the winner log (tree_path_lcp_max.jsonl). Reconstructs the committed token
sequence from per-step emitted_tokens, detokenizes to find the garble region (identifier truncation), and
prints the winner-log rows around it: winner_spine (0=spine,>0=branch), winner_path (accepted node indices),
self_target_ids (verify argmax per node), draft_token_ids, path_scores. Answers:
  (1) does the garble COMMIT on a branch (winner_spine>0) => branch-specific for THIS repro; or spine (==0) => refuted.
  (2) is the emitted garble token a DRAFT (accepted) or a TARGET (bonus/verify-argmax)?
  (3) is the node indexing internally consistent (path nodes vs targets)?
"""
import sys, json, urllib.request

EP = "http://127.0.0.1:9957"
LOGP = sys.argv[1] if len(sys.argv) > 1 else "output/fr13_winnerspine/tree_path_lcp_max.jsonl"


def detok(ids):
    try:
        req = urllib.request.Request(EP + "/detokenize",
                                     data=json.dumps({"model": "default", "tokens": list(ids)}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()).get("prompt", "")
    except Exception as e:
        return f"<detok failed: {e}>"


def main():
    rows = [json.loads(l) for l in open(LOGP) if l.strip()]
    print(f"=== winner log: {len(rows)} committer steps ===", flush=True)
    spines = [r.get("winner_spine") for r in rows if "winner_spine" in r]
    from collections import Counter
    print("winner_spine distribution:", dict(Counter(spines)), flush=True)
    print("  (0 = spine path committed; >0 = a BRANCH path committed)", flush=True)
    nbranch = sum(1 for s in spines if s and s > 0)
    print(f"  branch commits: {nbranch}/{len(spines)}  ({'BRANCHES DO WIN => branch-specific plausible' if nbranch else 'SPINE ALWAYS WINS => branch-commit REFUTED for this run'})", flush=True)

    # reconstruct committed sequence + locate garble region by detok
    seq, step_of_tok = [], []
    for i, r in enumerate(rows):
        for t in r.get("emitted_tokens", []):
            seq.append(int(t)); step_of_tok.append(i)
    full = detok(seq)
    print(f"\n=== reconstructed committed text ({len(seq)} tokens) ===\n{full[:1600]}", flush=True)

    # find garble: search for known truncations
    for needle in ["expected_rows", "expected_row ", "expected_rows("]:
        idx = full.find(needle)
        if idx >= 0:
            print(f"\n=== garble '{needle}' found at char {idx}; dumping the committer steps that emit around it ===", flush=True)
            # map char->token is approximate; dump the steps whose detok contains the needle fragment
            for i, r in enumerate(rows):
                seg = detok([int(t) for t in r.get("emitted_tokens", [])])
                if "expected" in seg or "rows" in seg or "_row" in seg:
                    print(json.dumps({"step": i, "winner_spine": r.get("winner_spine"),
                                      "winner_acc": r.get("winner_acc"), "winner_path": r.get("winner_path"),
                                      "emitted": seg, "self_target_ids": r.get("self_target_ids"),
                                      "draft_token_ids": r.get("draft_token_ids"),
                                      "path_scores": r.get("path_scores")}, indent=1), flush=True)
            break
    else:
        print("\n(no known truncation string found in committed text; inspect full text above)", flush=True)


if __name__ == "__main__":
    main()
