# FR14 STOP-LEVEL FINDING — the agent reaches the internet through the shell, and always could

**Status: OPEN. The no-net settings change (`1a68a0b31`) does NOT close this.**
Quality claims from any arm whose agent had network are void; timing claims are
perturbed. Closing it needs a container-topology change, not a settings change.

## What happened

`config/fr13_fixed32/qwen_system_settings.json` was changed to
`tools.exclude: ["web_fetch","web_search","tool_search"]` to delete the
`web_fetch` class that killed two arms. It worked, exactly as measured: the
session-init registry went 61 → 59 tools and `web_fetch` is gone from all three
of tonight's traces.

The model then routed around it.

From `output/fr14_b1_stock_20260817T031507Z/.../astropy__astropy-13236/qwen_trace.jsonl`:

| # | line | command | outcome |
|---|---:|---|---|
| 1 | 259 | `curl -sL https://raw.githubusercontent.com/astropy/astropy/v5.2/CHANGES.rst` | **DENIED** — `Tool "run_shell_command" is denied by permission rules.` |
| 2 | 263 | `python -c "import urllib.request; url='https://raw.githubusercontent.com/.../CHANGES.rst' ..."` | **SUCCEEDED**, 544,282 B |
| 3 | 271 | `python -c "import urllib.request; url='https://patch-diff.githubusercontent.com/raw/astropy/astropy/pull/13236.diff' ..."` | **SUCCEEDED**, 6,375 B — **the gold patch** |

The task's own result text then says the fix is *"byte-identical to upstream
PR #13236"*.

### Why curl was denied and python was not

qwen-code enforces `WebFetch` permission rules against *equivalent shell
commands* (its settings.md: "Permission rules for `Read`, `Edit`, and `WebFetch`
are also enforced when the agent runs equivalent shell commands... the agent
cannot bypass it via `cat .env`"). `tools.exclude:["web_fetch"]` auto-migrates
to `permissions.deny`, so the shell-equivalence detector caught `curl`. It does
not recognise `python -c "import urllib.request"`.

**That detector is a heuristic, so every settings-level fix is bypassable.** A
denylist of command shapes is theatre: the next route is `requests`, a
base64-encoded script, `socket`, or a here-doc.

## It is not new, and it touches a banked artifact

Scanned every FR14 trace on disk for network-shaped shell commands whose results
came back with real content:

| runroot | task | web_fetch hits | shell net attempts | reached net |
|---|---|---:|---:|---|
| `20260816T204931Z` (**arm A banked stock**) | 13398 | 2 | **8** | **8 / 8** |
| `20260817T031507Z` (no-net) | 13236 | 0 | 3 | 2 / 3 (1 denied) |
| all other tasks, all runroots | — | 1–6 | 0 | 0 |

In arm A's banked stock arm, 13398's agent enumerated the astropy PR list via
the GitHub API and fetched **`pull/13398.diff` — that task's gold patch** (line
81). It still returned `verdict=failed`, so no false resolve was recorded, but
the trajectory — and therefore the 2,375.8 s wall that feeds the timing — was
spent partly on network work.

**Consequence for the record:** `fr14_b1_stock_20260816T204931Z` is the source
of the banked arm-A stock headline (218.764 ms step wall / 25.261 TPS / accept
4.526) and of REDTEAM pass 10's "resolve 2/4, quality pattern identical to 3.6"
claim. The speed numbers are weakly perturbed (one of four tasks had a
contaminated trajectory); the **quality** claim for 13398 is void. The same
applies to every earlier arm that used `web_fetch` freely.

## What tonight's no-net run DOES establish

Both clean tasks are clean by measurement, not assumption — 0 `web_fetch`,
0 `web_search`, 0 `tool_search`, 0 network-shaped shell commands, 0 permission
denials:

* **`astropy__astropy-13033`: `verdict=resolved`, 857.672 s, 1,515 B patch —
  the first honest 13033 resolve in campaign history.** Every prior resolve of
  this task was obtained with the agent fetching the astropy issue timeline or
  PR diff.
* `astropy__astropy-12907`: `verdict=resolved`, 540.05 s, 504 B patch, clean.

So the settings change is a real improvement and should stay. It is just not a
boundary.

## The fix

A boundary has to be enforced where the agent cannot argue with it: the network
namespace. The agent container needs exactly one reachable endpoint — the
offload proxy that carries `OPENAI_BASE_URL` — and nothing else. It does not
need the internet: `SWE_AGENT_ENV=instance_image` bakes the repo and its
dependencies into the per-instance image.

Today the agent runs `--network=host` (QWEN_CODE_TEMPLATE), which gives it the
offload host's full egress.

Recommended, in preference order:

1. **Own netns + egress allowlist.** Run the agent on a user-defined bridge,
   point `OPENAI_BASE_URL` at the bridge gateway, and add a `DOCKER-USER` rule
   dropping everything from that subnet except the proxy port. Robust against
   any in-container technique.
2. **`--network=none` + a socket-activated forwarder** for the proxy port only.
   Strongest, but needs a shim because qwen-code wants an HTTP URL.
3. DNS denial alone (block resolution, keep the proxy reachable by IP) — cheap,
   raises the bar a lot, but a hardcoded IP defeats it. Mitigation, not a
   boundary.

Any of these is a harness topology change with its own boot-refusal probe, and
it changes workload identity again, so it wants its own commit and its own
before/after run. **It is a prerequisite for the exact16 QC ladder**, where a
single gold-patch fetch would silently manufacture a resolve.

## Verification recipe (reusable)

The scan that produced the table is worth keeping as a gate — it is cheap and it
runs on evidence that already exists:

```
for each qwen_trace.jsonl:
    count tool_use(run_shell_command) whose command matches
        (urllib|requests|httpx|socket\.|curl|wget|https?://)
    and whose paired tool_result did NOT come back
        "denied by permission rules"
```

Any non-zero count on a QC arm should void that task's quality verdict.
