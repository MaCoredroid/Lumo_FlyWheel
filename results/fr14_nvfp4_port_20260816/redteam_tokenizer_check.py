"""FR14 red-team pass 1 verification transcript (2026-08-16).

Reproduces the tokenizer-identity and quant-config checks in
REDTEAM_20260816.md against the local model dirs. Read-only.
"""
import json

M = "/home/mark/shared/models"
t36 = json.load(open(f"{M}/qwen3.6-27b-fp8/tokenizer.json"))
t38 = json.load(open(f"{M}/qwen3.8-27b-nvfp4/tokenizer.json"))

assert t36["model"]["vocab"] == t38["model"]["vocab"], "vocab mapping drifted"

def norm(ms):
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in ms]

assert norm(t36["model"]["merges"]) == norm(t38["model"]["merges"]), "merges drifted"

added36 = {t["id"] for t in t36.get("added_tokens", [])}
added38 = {t["id"] for t in t38.get("added_tokens", [])}
new_ids = sorted(added38 - added36)
assert new_ids == list(range(248070, 248077)), f"unexpected added tokens: {new_ids}"

qc = json.load(open(f"{M}/qwen3.8-27b-nvfp4/config.json"))["quantization_config"]
assert qc["quant_method"] == "compressed-tensors" and qc["format"] == "mixed-precision"
assert any("mtp" in str(i) for i in qc["ignore"]), "MTP not in quant ignore list"

print("PASS: vocab+merges identical 3.6<->3.8-unsloth; +7 audio/TTS added tokens;")
print("      quant config = compressed-tensors mixed-precision, MTP ignored.")
