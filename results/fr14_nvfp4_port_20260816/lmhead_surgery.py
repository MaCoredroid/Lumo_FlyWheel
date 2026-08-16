"""FR14 checkpoint surgery: dequantize unsloth NVFP4 checkpoint's FP8 lm_head to BF16.

Why: vLLM 0.19.2's qwen3_5 builds lm_head as an unquantized ParallelLMHead, so the
checkpoint's lm_head.weight_scale fail-closes the load. BF16 lm_head reproduces the
served-3.6 configuration exactly (drafter DVK slice stays byte-compatible). All other
tensors pass through byte-identical. Original preserved with sha256 manifest.
"""
import json, hashlib, shutil, os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/qwen3.8-27b-nvfp4/model.safetensors"
DST = "/models/qwen3.8-27b-nvfp4/model.safetensors.new"
ORIG_DIR = "/models/_fr14_orig_nvfp4_fp8head"

tensors, meta = {}, None
with safe_open(SRC, framework="pt") as f:
    meta = f.metadata()
    names = list(f.keys())
    assert "lm_head.weight" in names and "lm_head.weight_scale" in names, names[:5]
    for n in names:
        tensors[n] = f.get_tensor(n)

w = tensors["lm_head.weight"]
s = tensors.pop("lm_head.weight_scale")
print("lm_head.weight", w.dtype, tuple(w.shape), "| scale", s.dtype, tuple(s.shape))
assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (248320, 5120)
assert tuple(s.shape) in ((248320, 1), (248320,)), s.shape
deq = (w.to(torch.float32) * s.to(torch.float32).reshape(-1, 1)).to(torch.bfloat16)
tensors["lm_head.weight"] = deq
print("dequantized to", deq.dtype, tuple(deq.shape))

save_file(tensors, DST, metadata=meta)
print("wrote", DST, os.path.getsize(DST))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

os.makedirs(ORIG_DIR, exist_ok=True)
manifest = {
    "surgery": "lm_head fp8-perchannel -> bf16 dequant; lm_head.weight_scale dropped",
    "src_sha256": sha(SRC), "dst_sha256": sha(DST),
    "src_size": os.path.getsize(SRC), "dst_size": os.path.getsize(DST),
}
shutil.move(SRC, ORIG_DIR + "/model.safetensors.orig")
shutil.move(DST, SRC)
json.dump(manifest, open(ORIG_DIR + "/surgery_manifest.json", "w"), indent=1)
json.dump(manifest, open("/models/qwen3.8-27b-nvfp4/.lumo_lmhead_surgery.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
