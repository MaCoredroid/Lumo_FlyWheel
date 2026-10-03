#!/usr/bin/env python3
"""Capture vLLM-rendered prompt token IDs (chat template + tools + generation prompt) for every corpus
request from a running vLLM server. Tokenize calls do no generation. Usage: tokenize_corpus.py REQ_DIR OUT_JSON"""
import glob, hashlib, http.client, json, os, sys
MODEL = "qwen3.8-27b-nvfp4-radixark"
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    rec = json.load(open(f))
    body = {"model": MODEL, "messages": rec["messages"], "add_generation_prompt": True}
    if rec.get("tools"):
        body["tools"] = rec["tools"]
    c = http.client.HTTPConnection("127.0.0.1", 9950, timeout=300)
    c.request("POST", "/tokenize", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse(); data = json.loads(r.read())
    if r.status != 200 or "tokens" not in data:
        sys.exit(f"tokenize failed for {f}: {r.status} {str(data)[:300]}")
    ids = data["tokens"]
    out[os.path.basename(f)] = {"ids": ids, "count": len(ids), "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    print(os.path.basename(f), len(ids), flush=True)
json.dump(out, open(sys.argv[2], "w"))
print("wrote", len(out), "requests")
