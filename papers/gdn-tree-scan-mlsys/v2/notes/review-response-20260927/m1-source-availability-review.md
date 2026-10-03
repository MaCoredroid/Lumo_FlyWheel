# M1 source availability: independent static review

**Disposition: source availability accepted for the declared isolated computational-module route; no blocking missing computational source or hidden arithmetic replacement was found.** Adapter implementation, runtime loading, numerical qualification, lifecycle correctness and timing readiness are outside this review. No author module was imported, and no SSH, Docker, GPU, model or evaluator operation was performed. Only this note was written; campaign code and gates were unchanged.

## Identity and static verification

Reviewed `experiments/review-response-20260927/m1/M1-AUTHOR-SOURCE-MANIFEST.v1.json`, SHA-256 **`890da0e3333eb21a7cb531e79c3aa5904a0f63adfcd8d0a222ca85cca6694e14`**. The paths in that document resolve relative to `experiments/review-response-20260927/`, not its containing `m1/` directory.

Using only standard-library JSON, SHA-256 and AST parsing, I independently verified:

- **22/22 files** match local bytes, byte sizes, `sha256` and `manifest_sha256`: 14 original author files and eight helper/license files.
- Every row has one matching repository/commit/upstream-path entry in the appropriate upstream manifest, with identical resolved local path, hash and size.
- Original manifest SHA-256: `49e48f5060dc04b557d9f7d8c6533783dd2b56643914b35b2cd4657abbf90ff2`; helper manifest: `aa07a576e1747a6c6d8e4a43fa37cda78d11fad063790b7b649e65895f744c03`.
- The canonical hash also reproduces: `6237cf56b3d4d41e9cfab6301e639dcccd90cb4ba3e86763636e4a327983f4c0`, using sorted-key compact JSON after excluding `generated_utc` and the canonical-hash field itself.
- Sources remain bound to TreeWY `sneha5gsm/vllm@b073ed6cacfa1cf5ae23111854729e662a72a6d1` and Weaver `trymirai/sglang@aeac03f0d4c8789559411be95c5c127bdff24d1c`; both author licenses and the unchanged helper license are present.

This checks local source provenance against the two retained manifests; no fresh upstream download or claim of complete upstream checkout validation is implied.

## Import closure and shim

The selected computation graph is closed at the source level: TreeWY's Triton/reference/topology modules use torch/Triton and standard-library helpers. Weaver's `gdn_tree_triton` imports `gdn_tree_fused` (including the real **msgspec** dependency); `chunk_tree_verify` imports `l2norm`, `op`, `utils`, and `wy_fast`; `wy_fast → index → utils`; `l2norm/op → utils`; gating uses torch/Triton. All corresponding author files are present. External torch, Triton (including libdevice), packaging and msgspec still require the eventual pinned runtime; this audit does not establish their installed compatibility.

The only serving-stack symbol consumed by these helper modules is `utils.py:18` importing `common.torch_release`. The proposed shim exactly reproduces `common.py:106`, `pkg_version.parse(torch.__version__).release`, with `pkg_version` bound to `packaging.version`. The retained `common.py` hash is `e6cf3c2dcf4a1176b119f6d891861568b87ac88988ad6ef1cd9f304dce653e5d`. This substitutes package plumbing with the same version expression, not normalization, gating, state arithmetic, a capability constant, or an author computational helper.

`utils.require_version` contains a lazy transformers import, but no selected module calls or decorates with that helper. It is not an unbound dependency of this selected execution graph. Conversely, the retained full `gdn_backend.py`, full `common.py`, and some integration tests require additional SGLang/vLLM modules; their unchanged full-stack import is **not** closed. They must remain reference material for this loader scope.

Implementation must honor the already declared isolation: bind the exact absolute `sglang.srt.layers.attention.fla.*` names required by author imports to these retained modules in an isolated process/namespace, without executing unbound parent-package initializers or borrowing an installed helper accidentally. A `weaver.*` alias alone does not satisfy those absolute imports. This is a requirement of the proposed loader, not an assertion that the unfinished adapter already does it.

## Arithmetic and remaining boundary

No arithmetic is hidden in the one shim. The real `utils.py` device probes and capability values remain active; `op.py` retains its `FLA_USE_FAST_OPS` dispatch; `gdn_tree_triton.py` retains its BF16-operand mode. Later runtime receipts must bind those actual settings rather than replace them with constants. Normalization remains FP32 division by `sqrt(sum(x*x)+eps)` with destination-dtype storage, and `fused_gdn_gating` still casts sigmoid beta to the input `b` dtype before its FP32 stash write. The author-internal verifier/replay beta-rounding seam is therefore preserved, not silently aligned to native arithmetic.

The manifest's device-import warning is directionally correct: `utils.py:264–280` probes the active device during initialization. Its phrase “every computational module imports triton” is overly broad (`tree_wy_ref.py` uses torch only), but does not affect source availability or the instruction to avoid host imports.

**No source-availability repair is required.** Preserve real external dependencies, the single exact shim, source hashes and the isolated import mapping in the forthcoming implementation. Runtime/import success and complete-cycle correctness remain untested here; no timing approval, scientific result, full-author-serving-stack claim, or Bole author-code availability is added.
