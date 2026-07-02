# qwen-code-runner:v1 — Qwen's own agentic CLI as the SWE agent (SWE_AGENT=qwen_code).
# Uses Qwen-native XML tools + search/replace edits over /v1/chat/completions (aligned with
# Qwen's training), unlike codex's V4A apply_patch. Analog of codex-runner:v1.
FROM node:22-bookworm
# Debian userland so qwen-code's shell/edit tools have real binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3 python3-venv ripgrep patch sed grep ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# NOTE: pin a version tag for cross-host/rebuild reproducibility once a good one is chosen;
# @latest is used for the first build and the resolved version is logged by the build script.
RUN npm install -g @qwen-code/qwen-code@latest
# Preempt git "dubious ownership" when qwen-code runs git inside the rsynced /workspace as uid 1000.
RUN git config --system --add safe.directory '*'
RUN useradd -m -u 1000 -s /bin/bash runner || true
# Runtime runs -u 1000:1000 with -e HOME=/tmp (writable ~/.qwen for CLI config/session).
ENV HOME=/tmp
WORKDIR /workspace
# ENTRYPOINT = the qwen binary so the harness template appends flags + `-p <prompt>` as argv.
ENTRYPOINT ["qwen"]
