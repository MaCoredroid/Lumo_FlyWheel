# FR13 — codex-rs v0.128 patch to give qwen3.6-27b a routable function-form apply_patch

**Date:** 2026-07-02. **Status:** DESIGN APPROVED-PENDING (research workflow wdw900bfd, 7 agents, verified
vs real rust-v0.128.0 source). Build NOT yet executed. Full workflow output:
`/tmp/claude-1000/.../tasks/wdw900bfd.output`.

## Corrected mechanism (0.128, NOT 0.139)
- 0.128's ApplyPatchHandler `matches_kind` accepts **BOTH `ToolPayload::Function` AND `Custom`**
  (apply_patch.rs:302-307); `handle()` parses `ApplyPatchToolArgs{input}` from Function args (:352-356).
- A **function-form** apply_patch tool already exists: `create_apply_patch_json_tool()` → `ToolSpec::Function`
  (wire `type:"function"`, param `input:string`), marked "for gpt-oss models" (apply_patch_tool.rs:101-122).
- vLLM emits `function_call` → router maps to `ToolPayload::Function` (router.rs:181-205) → routes to the
  handler. **No proxy bridge, no custom_tool_call, no lark grammar.**
- The ONLY reason it fails today: apply_patch is registered only when `apply_patch_tool_type.is_some()`
  (tool_registry_plan.rs:322-341). Unknown slug `qwen3.6-27b` → `model_info_from_slug` fallback sets it
  `None` (model_info.rs:87) → not registered → `unsupported call: apply_patch` → shell fallback.
- Other native tools: `update_plan`/`view_image`/`request_user_input`/`exec_command` already register
  unconditionally + route as function tools. `web_search`/`image_gen`/`tool_search` are server-hosted /
  unroutable over vLLM → **do NOT enable**. So apply_patch is the ONLY missing routable native tool.

## THE PATCH (Variant B — zero blast radius; qwen-only)
File: `codex-rs/models-manager/src/model_info.rs`
1. Add import near the other `use codex_protocol::openai_models::*` lines:
   ```rust
   use codex_protocol::openai_models::ApplyPatchToolType;
   ```
2. In `fn model_info_from_slug` (lines 66-101), replace `apply_patch_tool_type: None,` (line 87) with a
   qwen-guarded value:
   ```rust
   apply_patch_tool_type: if slug.starts_with("qwen") {
       Some(ApplyPatchToolType::Function)
   } else {
       None
   },
   ```
That is the entire required change. `Some(Function)` passes straight through tool_config.rs:191 →
tool_registry_plan.rs pushes `create_apply_patch_json_tool()` (ToolSpec::Function) + registers the handler.
gpt-5.x slugs resolve via the compiled models.json catalog and NEVER hit this fallback → untouched.

DO NOT EDIT (verified already correct in 0.128): apply_patch.rs handler, router.rs mapping,
apply_patch_tool.rs json tool, tool_config.rs Some(Function) arm.

## Build (x86_64-musl on alienware = the ACTIVE OFFLOAD_CODEX=1 path; only build strictly required)
1. `git clone --depth 1 --branch rust-v0.128.0 https://github.com/openai/codex` (commit e4310be5). The tag
   exists. A FULL checkout is required (workspace Cargo.toml + rust-toolchain.toml).
2. Toolchain: **verify `rust-toolchain.toml`** (asserted 1.93.0 + rust-src). `rustup toolchain install 1.93.0`,
   `rustup component add rust-src`, `rustup target add x86_64-unknown-linux-musl`.
3. musl deps (per .github/scripts/install-musl-build-tools.sh): musl-tools pkg-config libcap-dev g++ clang
   libc++-dev libc++abi-dev lld xz-utils; static libcap 2.75. Fragile parts = aws-lc/BoringSSL + rusty_v8
   (native x86 build sidesteps the zig-cc cross).
4. Apply the patch, then `cargo build --release --target x86_64-unknown-linux-musl --bin codex`.
5. Output `target/x86_64-unknown-linux-musl/release/codex` (~185-193MB static musl ELF); `./codex --version`
   → `codex-cli 0.128.0`. First build ~20-40 min (rusty_v8/aws-lc dominate).

## Deploy (alienware; reversible)
1. **Verify the vendor path first:** on alienware `docker run --rm codex-runner:v1 ls
   /usr/local/lib/node_modules/@openai/codex/node_modules/@openai/` (expect `codex-linux-x64`).
2. `docker tag codex-runner:v1 codex-runner:v1-stock` (one-command rollback).
3. Layer the patched binary:
   ```dockerfile
   FROM codex-runner:v1
   COPY codex-x64 /usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex
   RUN chmod +x /usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex
   ```
   `docker build -t codex-runner:v1 …` on alienware; verify `codex --version`.
(arm64/DGX build only if OFFLOAD_CODEX=0 is ever used; DGX-idle-gated per no-compute-on-test-machine.)

## Probe ladder (MUST pass before any campaign)
- **L1** (patched binary, mock /v1/responses): tools[] contains `{"type":"function","name":"apply_patch",
  "parameters":{…"input"…}}` — type MUST be `"function"`, not `"custom"`.
- **L2** (1 task, real vLLM, OFFLOAD arch): PASS = (a) qwen emits an apply_patch **FunctionCall** (not a
  `{command:["apply_patch",…]}` shell call); (b) NO `unsupported call: apply_patch`; (c) NO
  intercept_apply_patch warning; (d) `git diff` non-empty via the ApplyPatch handler. Then a ≥8-task control
  vs the shell-fallback resolve rate before campaign.

## Risks / open questions (workflow verdict: will_compile=yes, will_route=LIKELY, must_probe=true, conf=high)
1. **Biggest unknown (prompt-vs-schema shape):** base prompt.md:132 nudges the SHELL command-array envelope
   `{"command":["apply_patch","*** Begin Patch…"]}`, NOT the function `{"input":"*** Begin Patch…"}` the
   registered schema requires. A compliant grammar-constrained model emits `input=…`, but our vLLM
   (qwen3_xml parser) may not constrain. **L2 must confirm Qwen emits `{input:…}` that deserializes into
   `ApplyPatchToolArgs`.** If not → proxy-side arg normalization (separate work).
2. char-8: routing patch content through function args re-exposes truncation (MAX_OUTPUT 32768); the shipped
   transcript json_repair covers the transcript side, but verify on a truncated apply_patch arg.
3. Build toolchain (aws-lc/rusty_v8 musl) is the fragile part; verify rust-toolchain.toml on a fresh clone.
4. Alienware `codex-linux-x64` vendor path unverified — confirm before the COPY layer.
5. No evidence apply_patch editing resolves MORE than the working shell fallback — the ≥8-task control gates
   campaign use. Keep `codex-runner:v1-stock` for one-command rollback.

---
## L2 RESULT + DECISION (2026-07-02) — routing PROVEN, but apply_patch is a Qwen training mismatch → ROLLED BACK
Built + deployed the 2-line Variant-B patch (codex-runner:v1, v1-stock kept). Probes:
- **L1 PASS**: patched codex advertises 12 tools incl. `apply_patch` as `type:"function"` (absent on stock).
- **L2 (astropy-12907, 1 task, live vLLM)**: `unsupported call: apply_patch` = **0** (was 4 on stock) → **routing PROVEN**. BUT **14 `apply_patch verification failed`, 0 successes** (e.g. "Expected update hunk to start with a @@ context marker, got: 'def _cdot(...)'"). The model then **fell back to shell** and landed an 851B patch (verdict=failed, wrong fix — same as stock, + 14 wasted turns).

**Root cause (online research, confirmed):** apply_patch's V4A `*** Begin Patch` exact-context grammar is an OpenAI/GPT-trained format. Qwen3-Coder's edit training is **search-and-replace diff patches** + the XML `qwen3_coder` tool format (not V4A). So Qwen cannot reliably emit valid apply_patch hunks → net-negative (wasted turns, 0 success, shell fallback anyway). Qwen's own recommended harness = `qwen-code` (native XML tools + Qwen-suited edit tool). Sources: qwenlm.github.io/blog/qwen3-coder, qwen.readthedocs.io function_call, Qwen3-Coder-Next tech report (search-and-replace FIM).

**DECISION: rolled back** (`docker tag codex-runner:v1-stock codex-runner:v1` on alienware) → the proven shell-fallback harness (Qwen-aligned, 5/5 baseline). The patch is validated + reversible (re-tag the built image to reinstate). **Proper future path if structured edits are wanted:** add a Qwen-native **search-and-replace edit tool** (matching its training), not apply_patch. The char-8 transcript-repair fix + uncapped wall remain in effect (independent of this).
