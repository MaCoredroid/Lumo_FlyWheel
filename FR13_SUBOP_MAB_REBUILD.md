# FR13_SUBOP_MAB_REBUILD — make the L0-GDN sub-op A/B reliable + never vacuous

Status: READY-TO-APPLY (write-only doc; the patcher
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` is READ-ONLY here — a GPU bake is
booting from it). Apply these edits to the patcher AFTER the bake frees it.

Reward-hacks BANNED: FR13_GDN_SUBOP_MAB is an OBSERVE-ONLY instrument — no
splice, no copy-recurrent, no dense-route, no mutation of the live forward. The
locked cat9 default path stays byte-identical when OFF. Front B bounds-guard fix
(8cdda4c4/d30755c8) stays verbatim.

> **FR13_BUG_CLASS_PLAYBOOK row 9 — Silent fallback / vacuous instrument**
> (FR10_REQUIRE_TREE; diag[12]; launcher silent-OFF `FR13_FA2_PREFILL_NATIVE`):
> *symptom* = "a run 'passes' while measuring nothing"; *discriminator* =
> "engagement asserts (sentinel in logs, backend line, flag in container env)
> BEFORE trusting any number"; *prevention* = "fail-loud on disengagement; boot
> needles; record flag-state headers in every artifact"; *scope* = "every
> flag-gated feature + every launcher". **All four SUBOP_MAB failures are
> textbook row-9: infra disengagement masquerading as 'no result'.** The rebuild
> adds a per-stage engagement marker + a non-zero record-count gate so the
> FAILING STAGE is always obvious.

---

## 1. ROOT-CAUSE (verified against the live patcher 2026-06-14)

Line numbers are the live patcher (`scripts/fr10_phase4_patch_vllm_tree_gdn.py`).

### 1.1 Where the call site is reached vs skipped

The scan-site call `_fr13_gdn_subop_mab(...)` lives at **:3942**, nested:

```
:3889  if spec_sequence_masks is not None:        # spec-verify forward
:3897      use_fr10_tree = (FR10_ENABLE_TREE_GDN=="1"
                            and _fr10_active_decode_mode=="tree_mtp"
                            and attn_metadata.fr10_tree_parent is not None
                            and attn_metadata.num_spec_decodes > 0)
:3921      if use_fr10_tree:                       # <-- GATES THE CALL
:3936          if _fr13_gdn_subop_mab_enabled():
:3942              _fr13_gdn_subop_mab(self, ...)
```

REACHED iff ALL hold on the verify forward:
1. `spec_sequence_masks is not None` (a spec-decode verify forward — true on
   every cat9 tree-verify; pure-decode steps have it `None`).
2. `use_fr10_tree is True`, which (:3897-3902) requires:
   - `os.environ["FR10_ENABLE_TREE_GDN"]=="1"` (launcher :306 sets it; passed as
     a docker `-e` flag so it reaches the worker via spawn-inheritance), AND
   - `_fr10_active_decode_mode=="tree_mtp"` (read live from
     `rejection_sampler._FR10_DECODE_MODE`, set per-request by the driver
     `vllm_xargs fr10_decode_mode=tree_mtp` — `fr13_gdn_subop_mab_drive.py:62`;
     module default also `"tree_mtp"`), AND
   - `attn_metadata.fr10_tree_parent is not None`, AND
   - `attn_metadata.num_spec_decodes > 0`.
3. `_fr13_gdn_subop_mab_enabled()` (:3936) — env or sidecar resolves ON.

SKIPPED when ANY of those is false: a pure-decode step (legit — the deep-spine
carrier only exists on a tree-verify); a request without `fr10_decode_mode=
tree_mtp` (native non-tree scan path taken, the hook is not in that branch); tree
disengaged (`fr10_tree_parent is None` / `num_spec_decodes==0`).

### 1.2 The decisive split — why failure #4 was silent ("call genuinely not executing its body, given use_fr10_tree…")

There are TWO separate `if spec_sequence_masks is not None:` blocks in the SAME
forward:

- **CONV-site stash (:1843-1910)**: gated at :1900 ONLY on
  `_fr13_gdn_subop_mab_on = _fr13_gdn_subop_mab_enabled()` (:1892) — NOT on
  `use_fr10_tree_conv`. It stashes `self._fr13_gdn_subop_mab_pre_conv` (:1907) +
  `self._fr13_gdn_subop_mab_conv_state` (:1908) whenever the flag is ON and spec
  verify runs.
- **SCAN-site call (:3889-3961)**: the call at :3942 is gated at :3921 on
  `use_fr10_tree`.

So when the flag resolves ON but `use_fr10_tree` is False (decode-mode not
`tree_mtp`, OR `FR10_ENABLE_TREE_GDN` not in the worker forward, OR tree
disengaged on that forward), the stash WRITES into `self._fr13_gdn_subop_mab_*`
and the call site at :3942 is NEVER reached. The stash silently accumulates and
is never consumed → **zero records, zero log lines** = exactly the failure-#4
signature. The `if _fr13_gdn_subop_mab_enabled():` guard at :3936 is never even
evaluated because its enclosing `if use_fr10_tree:` is False, and the only
nearby log (the FR10-tree-active `warning_once` at :3933) is itself behind
`FR10_METRICS=="1"`.

### 1.3 "env present but body not executing" — two non-exclusive class-9 roots

- **ROOT-A (call site skipped)**: body lives only under `if use_fr10_tree:`. If
  `FR10_ENABLE_TREE_GDN` did not reach the worker OR the live
  `_FR10_DECODE_MODE` global was not `"tree_mtp"` at that forward, `use_fr10_tree`
  is False and the call at :3942 is bypassed entirely. No log of any kind.
- **ROOT-B (body reached, helper early-returns / asserts, swallowed)**: if the
  call IS reached, the helper can return SILENTLY at six gates — :1384 not
  `enabled()`; :1387 stream-capturing; :1391 `conv_state_snapshot is None`
  (logs a warning, the only early-return that does); :1404-1407 LAYER prefix
  mismatch; :1413 outside skip..skip+limit window. AND every class-9 engagement
  assert (:1422 pre_conv None, :1425 no path0, :1428 no parent, :1431
  num_spec<1, :1439 tree_n!=expected, :1452 degenerate spine, plus the per-arm
  bank/row guards) is inside the single `try:` at :1419 whose only handler at
  :1806 does `logger.warning("FR13_GDN_SUBOP_MAB A/B failed: %s", exc)` (:1807) —
  the SAME `logger.warning` level as the SUCCESS path warning at :1790, lost in
  the worker log flood.

Most probable single cause of #4: `use_fr10_tree` False at the scan site because
`FR10_ENABLE_TREE_GDN` / the `tree_mtp` mode did not hold in the worker forward
the driver produced → the gate was never evaluated (ROOT-A).

### 1.4 Why 14 sibling FR13_* vars reached the worker but not the master (the env mechanics)

The EngineCore worker is spawned by `CoreEngineProcManager`
(`/tmp/vllm_live_019/vllm/v1/engine/utils.py:101-133`) via
`get_mp_context().Process(target=EngineCoreProc.run_engine_core, kwargs=...)`
with **NO `env=` kwarg** (verified: the `context.Process(...)` call passes only
`common_kwargs | {dp_rank,...}`). An mp/spawn child therefore inherits the FULL
parent (`pid 1`) `os.environ` at `.start()` time.

The ray allowlist `get_env_vars_to_copy`
(`/tmp/vllm_live_019/vllm/ray/ray_env.py`, `DEFAULT_ENV_VAR_PREFIXES={VLLM_,
LMCACHE_,NCCL_,UCX_,HF_,HUGGING_FACE_}`) is used ONLY by
`CoreEngineActorManager` — the RAY-executor path — which never fires here (no
"Copying the following environment variables" log; `ray_env` never invoked).
Therefore `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` / `_PREFIXES_TO_COPY=FR13_`
(launcher :145-161, commit c05c662c / failure #3) are **INERT on this mp/spawn
path** — which is exactly why the allowlist fix did nothing.

The 14-vs-66 split is NOT a curated allowlist on the master: the 14 FR13_* that
reached the worker (incl. the SUBOP siblings `_LAYER`/`_LIMIT`/`_SKIP`) are the
ones passed as docker `-e` flags (launcher :340-384), so they sit in container
`pid-1 os.environ` and the spawn child inherits them. In the failure-2 era the
master `FR13_GDN_SUBOP_MAB` was being routed via the (inert) ray allowlist, not
as a real `-e` var pid-1 held, so it alone was absent from
`/proc/<worker>/environ`.

**Net: mp/spawn inherits whatever is in pid-1 `os.environ`; the master "dropped"
because it was supplied through the ray channel this spawn path ignores.** The
channel the 14 siblings rely on is the plain docker `-e` flag — which the current
launcher already gives the master at **:363** (`-e FR13_GDN_SUBOP_MAB=...`) plus
the siblings :364-369, AND the patcher writes a pid-1 sidecar
(`_fr13_write_subop_mab_sidecar` :13646, called from `main()` :13682) consumed by
`_fr13_gdn_subop_mab_enabled()` (:1320-1328). So as of the CURRENT build the
env-to-worker is solved two ways (belt-and-suspenders). The rebuild keeps both
and adds a FAIL-LOUD worker-env stage marker so a future regression is caught.

> KEY CORRECTION to the task premise: the patcher does NOT run at vLLM import in
> both pid 1 and the worker. It is a ONE-SHOT disk editor in pid 1 (launcher
> :402) that edits the in-image vLLM `.py` files; the worker imports the
> already-edited files. So "patcher-import-time robustness" = the pid-1 sidecar
> write (`main()` :13682), already implemented; the env reaches the worker via
> plain spawn-inheritance of the `-e` master, NOT via any allowlist.

---

## 2. THE REBUILD — design

Three pillars, each fail-loud at its stage so a future chase is never vacuous:

(a) **ROBUST ENV-TO-WORKER** — already solved via the `-e FR13_GDN_SUBOP_MAB`
    docker flag (the channel the 14 siblings use) + the pid-1 sidecar. The
    rebuild ADDS a one-time worker-env STAGE MARKER (EDIT-1) so a future
    regression is observable, and a sidecar-resolution marker.

(b) **CALL-SITE ENGAGEMENT** — hoist a `_fr13_gdn_subop_mab_enabled()` probe
    ABOVE the `if use_fr10_tree:` gate (EDIT-3). When the flag is ON but
    `use_fr10_tree` is False on a spec-verify forward, emit a one-time
    ERROR-level "call site skipped" marker naming WHICH precondition failed
    (this is the exact failure-#4 hole). The actual A/B call stays inside
    `use_fr10_tree` (where the tree tensors are valid).

(c) **FAIL-LOUD AT EVERY STAGE** — a single observable signal at each stage via
    a stage-marker helper (EDIT-2): (i) env-not-in-worker, (ii)
    call-site-not-reached, (iii) engagement-assert-fail (split OUT of the
    swallowing try → distinct ERROR tag), (iv) capture-written (record count).
    All markers carry the `FR13_SUBOP_STAGE=` tag at ERROR level so grep finds
    the failing stage instantly.

Default-OFF invariant: every new line is inside an `if
_fr13_gdn_subop_mab_enabled():`/`if _fr13_subop_stage(...)`-style guard that
short-circuits to a no-op when the flag is OFF. No locked-path needle text is
deleted; all edits are additive insertions around existing needles.

### 2.1 The stage-marker helper + counters (NEW, emitted next to the resolver)

A module-global counter dict + a one-time-per-tag logger so each stage emits
exactly one ERROR line (no flood) and a final non-zero record-count gate. ERROR
level (not warning) so it stands out from the `logger.warning` flood and from
the success/`A/B failed` warnings.

---

## 3. READY-TO-APPLY PATCHER EDITS (exact needles + replacements, AST-valid)

All edits are inside the python-string literals the patcher injects into
`gdn_linear_attn.py`. Each `old_string` is unique in the patcher; verify with
`grep -c`. Apply in order. After applying, re-run the CPU tests in §4.

### EDIT-1 — stage-marker helper + worker-env gate marker (insert after the resolver, before `_fr13_gdn_subop_mab`)

**Needle (unique — the resolver's closing lines + the helper def opener at :1330-1332):**

```
    except Exception:
        _FR13_GDN_SUBOP_MAB_FLAG = False
    return _FR13_GDN_SUBOP_MAB_FLAG


def _fr13_gdn_subop_mab(
```

**Replacement (insert the marker helper between the resolver and the A/B helper):**

```
    except Exception:
        _FR13_GDN_SUBOP_MAB_FLAG = False
    return _FR13_GDN_SUBOP_MAB_FLAG


_FR13_SUBOP_STAGE_SEEN = set()
_FR13_SUBOP_STAGE_COUNTS = {}


def _fr13_subop_stage(tag, msg, once=True, level="error"):
    """FR13_GDN_SUBOP_MAB class-9 stage marker (NEVER silently vacuous).

    Emits a single grep-able `FR13_SUBOP_STAGE=<tag>` line at ERROR level so the
    FAILING STAGE of a chase is unmistakable in the worker log flood.  `once`
    de-dups per (prefix-less) tag so a per-forward stage logs exactly one line.
    Always increments a monotone counter so the reducer can assert a non-zero
    record count vs a non-zero engaged count (vacuous-boot discriminator).

    Default-OFF safe: callers guard every invocation behind
    _fr13_gdn_subop_mab_enabled(); this fn itself does no env work and never
    touches the live forward.
    """
    _FR13_SUBOP_STAGE_COUNTS[tag] = int(_FR13_SUBOP_STAGE_COUNTS.get(tag, 0)) + 1
    if once:
        if tag in _FR13_SUBOP_STAGE_SEEN:
            return
        _FR13_SUBOP_STAGE_SEEN.add(tag)
    try:
        emit = getattr(logger, level, None) or logger.error
        emit("FR13_SUBOP_STAGE=%s %s", tag, msg)
    except Exception:
        pass


def _fr13_subop_worker_env_gate():
    """STAGE (i): on first resolution, log whether the master reached THIS proc
    (the mp/spawn EngineCore worker) via env or the pid-1 sidecar bridge, and
    self-check /proc/self/environ.  One ERROR line names the channel so a future
    'env did not reach the worker' regression (failures #2/#3) is loud.

    Observe-only: reads env + the sidecar flag path; writes nothing.
    """
    if not _fr13_gdn_subop_mab_enabled():
        return
    env_present = os.environ.get("FR13_GDN_SUBOP_MAB") in ("1", "0") and bool(
        os.environ.get("FR13_GDN_SUBOP_MAB")
    )
    flag_path = os.environ.get(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", "/logs/fr13_gdn_subop_mab.flag"
    )
    sidecar_present = False
    try:
        with open(flag_path, "r") as _fh:
            sidecar_present = _fh.read().strip() == "1"
    except Exception:
        sidecar_present = False
    proc_present = False
    try:
        with open("/proc/self/environ", "rb") as _pf:
            proc_present = b"FR13_GDN_SUBOP_MAB=" in _pf.read()
    except Exception:
        proc_present = None
    channel = (
        "env" if env_present else ("sidecar" if sidecar_present else "NONE")
    )
    _fr13_subop_stage(
        "worker-env",
        (
            "engaged channel=" + str(channel)
            + " env_master=" + repr(os.environ.get("FR13_GDN_SUBOP_MAB"))
            + " sidecar=" + str(sidecar_present)
            + " proc_self_has_master=" + str(proc_present)
            + " pid=" + str(os.getpid())
        ),
    )


def _fr13_gdn_subop_mab(
```

### EDIT-2 — split the engagement asserts OUT of the swallowing try (helper body)

The fix is NOT to remove the `try:` (the native-kernel arms must stay guarded —
a CUDA fault should not crash the engine) but to run the class-9 ENGAGEMENT
ASSERTS *before* the try, raising a distinct stage marker on failure so they are
unmistakable, and to keep only the arm-execution inside the swallowing try.

**Needle (the assert block currently inside the try, :1419-1453):**

```
    try:
        # ----- class-9 ENGAGEMENT ASSERTS (fail loud, never a vacuous number) --
        if pre_conv_spec is None:
            raise RuntimeError("FR13_GDN_SUBOP_MAB: pre_conv_spec not captured")
        path0 = getattr(attn_metadata, "fr10_tree_path0_nodes", None)
        if path0 is None:
            raise RuntimeError("FR13_GDN_SUBOP_MAB: tree DISENGAGED (no path0)")
        tree_parent = getattr(attn_metadata, "fr10_tree_parent", None)
        if tree_parent is None:
            raise RuntimeError("FR13_GDN_SUBOP_MAB: tree DISENGAGED (no parent)")
        num_spec = int(attn_metadata.num_spec_decodes)
        if num_spec < 1:
            raise RuntimeError("FR13_GDN_SUBOP_MAB: num_spec_decodes < 1")
        tree_n = int(tree_parent.numel())
        # tok/draft engagement: assert the served tree matches the expected
        # geometry (cat9 => 10 nodes incl root).  An env override allows other
        # shapes; otherwise fail loud on a silent shape change (class 9).
        expect_tree_n = os.environ.get("FR13_GDN_SUBOP_MAB_EXPECT_TREE_N", "10")
        if expect_tree_n and expect_tree_n != "*":
            if tree_n != int(expect_tree_n):
                raise RuntimeError(
                    "FR13_GDN_SUBOP_MAB: tree_n="
                    + str(tree_n)
                    + " != expected "
                    + str(expect_tree_n)
                    + " (silent tree-shape change; class 9)"
                )
        dev = mixed_qkv_spec.device
        path0_list = [int(x) for x in path0.detach().cpu().reshape(-1).tolist()]
        # Only nodes that actually exist in the served tree (< tree_n).
        spine_rows = [r for r in path0_list if 0 <= r < tree_n]
        if len(spine_rows) < 2:
            raise RuntimeError(
                "FR13_GDN_SUBOP_MAB: degenerate spine " + str(spine_rows)
            )
        deep_row = spine_rows[-1]
```

**Replacement (HOIST the asserts above the try, tag each with a loud stage
marker, keep the arm body inside the swallowing try):**

```
    # ----- class-9 ENGAGEMENT ASSERTS (HOISTED above the swallowing try so a
    # disengagement is a LOUD `FR13_SUBOP_STAGE=engage-fail` ERROR, never a
    # warning lost in the flood — the call WAS reached, so a vacuous skip here is
    # a real defect, not a legit pure-decode skip).  Each guard records the
    # specific reason then returns WITHOUT writing a record.
    def _engage_fail(reason):
        _fr13_subop_stage("engage-fail", reason, once=False)
    if pre_conv_spec is None:
        _engage_fail("pre_conv_spec not captured (conv-site stash disengaged)")
        return
    path0 = getattr(attn_metadata, "fr10_tree_path0_nodes", None)
    if path0 is None:
        _engage_fail("tree DISENGAGED (no path0)")
        return
    tree_parent = getattr(attn_metadata, "fr10_tree_parent", None)
    if tree_parent is None:
        _engage_fail("tree DISENGAGED (no parent)")
        return
    num_spec = int(attn_metadata.num_spec_decodes)
    if num_spec < 1:
        _engage_fail("num_spec_decodes < 1 (= " + str(num_spec) + ")")
        return
    tree_n = int(tree_parent.numel())
    # tok/draft engagement: the served tree must match expected geometry
    # (cat9 => 10 nodes incl root).  Env override allows other shapes; otherwise
    # a silent shape change is class-9 and must be loud.
    expect_tree_n = os.environ.get("FR13_GDN_SUBOP_MAB_EXPECT_TREE_N", "10")
    if expect_tree_n and expect_tree_n != "*":
        if tree_n != int(expect_tree_n):
            _engage_fail(
                "tree_n=" + str(tree_n) + " != expected " + str(expect_tree_n)
                + " (silent tree-shape change; class 9)"
            )
            return
    dev = mixed_qkv_spec.device
    path0_list = [int(x) for x in path0.detach().cpu().reshape(-1).tolist()]
    # Only nodes that actually exist in the served tree (< tree_n).
    spine_rows = [r for r in path0_list if 0 <= r < tree_n]
    if len(spine_rows) < 2:
        _engage_fail("degenerate spine " + str(spine_rows))
        return
    deep_row = spine_rows[-1]
    # STAGE (iv-pre): the event is fully engaged; the next failure point is the
    # arm execution (native kernels) which stays inside the swallowing try.
    _fr13_subop_stage(
        "engaged",
        "layer=" + str(prefix) + " tree_n=" + str(tree_n)
        + " deep_row=" + str(deep_row) + " num_spec=" + str(num_spec),
        once=False,
    )
    try:
```

> NOTE: this replacement deletes the original `try:` opener line and re-opens a
> NEW `try:` right after `deep_row = spine_rows[-1]`. The body that follows
> (`# The FULL-tree arm ...`, `start = 0`, etc., currently :1455 onward) is
> unchanged and is now the first statement of the new try block — indentation is
> identical (8 spaces), so the AST is preserved. The per-arm bank/row guards
> (`_guard_rows`, the prior-bank guards) and the arm bodies remain inside this
> new try (a CUDA fault there is still caught), but every arm-guard `raise`
> message is now ALSO surfaced loud: see EDIT-2b.

### EDIT-2b — make the swallow handler LOUD + distinct from success

**Needle (the swallow handler, :1806-1807):**

```
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("FR13_GDN_SUBOP_MAB A/B failed: %s", exc)
```

**Replacement (distinct ERROR-level stage tag, not the warning level shared with
the success path):**

```
    except Exception as exc:  # pragma: no cover - diagnostic only
        # ARM-execution failure (e.g. a per-arm bank/row bounds-guard raise, a
        # native-kernel fault).  LOUD + distinct tag so it is never confused with
        # the success warning at the M10-vs-M5 emit; the event produced NO record.
        _fr13_subop_stage(
            "arm-fail",
            "layer=" + str(getattr(self, "prefix", "")) + " exc=" + repr(exc),
            once=False,
        )
```

### EDIT-2c — record-written stage marker (STAGE iv)

**Needle (the success emit, the `logger.warning("FR13_GDN_SUBOP_MAB layer=%s ...` block tail — match its first line + the `out_path` write just above it). Use the JSONL write line which is unique (:1788-1789):**

```
        with open(out_path, "a", buffering=1) as fh:
            fh.write(json.dumps(rec) + chr(10))
        logger.warning(
            "FR13_GDN_SUBOP_MAB layer=%s event=%s deep_row=%s | M10-vs-M5 "
```

**Replacement (add a non-zero record-count stage marker right after the write):**

```
        with open(out_path, "a", buffering=1) as fh:
            fh.write(json.dumps(rec) + chr(10))
        _fr13_subop_stage(
            "record-written",
            "layer=" + str(prefix) + " event=" + str(seen)
            + " path=" + str(out_path),
            once=False,
        )
        logger.warning(
            "FR13_GDN_SUBOP_MAB layer=%s event=%s deep_row=%s | M10-vs-M5 "
```

### EDIT-3 — call-site engagement marker (hoist a probe ABOVE `if use_fr10_tree:`)

This closes the exact failure-#4 hole: the flag resolves ON, the conv-site stash
runs, but `use_fr10_tree` is False so the call is never reached and nothing logs.
Insert a one-time ERROR marker that names WHICH precondition of `use_fr10_tree`
failed — BEFORE the `if use_fr10_tree:` branch at the scan site.

**Needle (the scan-site `use_fr10_tree` resolution + the diag block + the
branch opener, :3897-3921). Match from the `use_fr10_tree = (` assignment
through the `if use_fr10_tree:` line. The unique anchor is the
`_fr10_scan_branch_diag[22].add_(...)` line immediately followed by `if
use_fr10_tree:`):**

```
                _fr10_scan_branch_diag[22].add_(float(attn_metadata.num_spec_decodes))
            if use_fr10_tree:
                assert spec_query_start_loc is not None
```

**Replacement (insert the call-site engagement probe between the diag block and
the branch; the probe is itself guarded by the resolver so it is a no-op when
OFF):**

```
                _fr10_scan_branch_diag[22].add_(float(attn_metadata.num_spec_decodes))
            # STAGE (ii) — CALL-SITE ENGAGEMENT (class-9, failure-#4 hole).  The
            # conv-site stash runs on enabled() ALONE, but the A/B CALL below is
            # gated on use_fr10_tree.  If the flag is ON yet use_fr10_tree is
            # False on a spec-verify forward, the call is SILENTLY skipped (stash
            # accumulates, zero records, zero log).  Emit ONE loud ERROR naming
            # the failing precondition so the chase is never vacuous.  Observe-
            # only; no effect on the live forward; no-op when the flag is OFF.
            if _fr13_gdn_subop_mab_enabled() and not use_fr10_tree:
                _fr13_subop_worker_env_gate()
                _fr13_subop_stage(
                    "callsite-skip",
                    (
                        "flag ON but use_fr10_tree False on a spec-verify forward "
                        "(stash accumulates, A/B call skipped). reasons:"
                        " FR10_ENABLE_TREE_GDN="
                        + repr(os.environ.get("FR10_ENABLE_TREE_GDN"))
                        + " decode_mode=" + repr(_fr10_active_decode_mode)
                        + " tree_parent_set="
                        + str(getattr(attn_metadata, "fr10_tree_parent", None) is not None)
                        + " num_spec_decodes=" + str(attn_metadata.num_spec_decodes)
                    ),
                )
                # release the orphaned stash so it cannot leak into a later event
                self._fr13_gdn_subop_mab_pre_conv = None
                self._fr13_gdn_subop_mab_conv_state = None
            elif _fr13_gdn_subop_mab_enabled() and use_fr10_tree:
                _fr13_subop_worker_env_gate()
            if use_fr10_tree:
                assert spec_query_start_loc is not None
```

> The `_fr10_active_decode_mode` referenced is the local already resolved at
> :3892-3896 (same scope). The `_fr13_subop_worker_env_gate()` call here is the
> first time the worker-env STAGE (i) marker can fire on a real verify forward
> (it `_once`-dedups, so it logs exactly one line per boot regardless of how many
> verify forwards happen). It runs in BOTH the skip and the engaged branch so the
> env channel is logged whether or not the A/B proceeds.

### EDIT-4 (optional, recommended) — call-site `worker-env` marker is also fired from inside the helper on the FIRST engaged event

Already covered: `_fr13_subop_worker_env_gate()` is invoked from EDIT-3 in both
branches. No additional edit. (Do NOT also call it from inside
`_fr13_gdn_subop_mab` to avoid a duplicate-tag race; the `_once` set makes it
idempotent anyway.)

---

## 4. CPU WIRING TESTS TO ADD (`tests/test_fr13_gdn_subop_mab_wiring.py`)

Extend the existing file (do not rewrite). All are pure-CPU, patcher-text +
emitted-helper-exec tests in the established style.

### 4.1 Patcher-text assertions (cheap, catch a needle drift)

```python
def test_stage_marker_helper_emitted() -> None:
    assert "def _fr13_subop_stage(" in SRC
    assert "FR13_SUBOP_STAGE=%s %s" in SRC
    assert "def _fr13_subop_worker_env_gate(" in SRC
    assert "/proc/self/environ" in SRC

def test_callsite_skip_marker_above_use_fr10_tree() -> None:
    # The call-site engagement probe must sit BEFORE the `if use_fr10_tree:`
    # branch at the scan site (failure-#4 hole).
    assert "callsite-skip" in SRC
    assert re.search(
        r"if _fr13_gdn_subop_mab_enabled\(\) and not use_fr10_tree:.*?"
        r"_fr13_subop_stage\(\s*\n\s+\"callsite-skip\"",
        SRC, re.DOTALL,
    )
    # The orphaned stash is released on the skip (no leak into a later event).
    assert SRC.count("self._fr13_gdn_subop_mab_pre_conv = None") >= 2

def test_engagement_asserts_hoisted_out_of_try() -> None:
    # The disengagement guards now emit a loud `engage-fail` marker and RETURN,
    # they are no longer bare `raise` inside the swallowing try.
    assert "engage-fail" in SRC
    assert "def _engage_fail(reason):" in SRC
    # The swallow handler is now a distinct ERROR tag, not the success-level
    # warning.
    assert 'FR13_GDN_SUBOP_MAB A/B failed' not in SRC  # old swallow text gone
    assert '"arm-fail"' in SRC
    assert '"record-written"' in SRC
    assert '"engaged"' in SRC
```

### 4.2 Emitted-helper behavioral tests (exec the helper on CPU stubs)

`_make_module` must add `_fr13_subop_stage`, `_fr13_subop_worker_env_gate`,
`_FR13_SUBOP_STAGE_SEEN`, `_FR13_SUBOP_STAGE_COUNTS` to the exec namespace by
extracting the WHOLE `mab_helper` block (they are emitted in the same block as
the resolver). The existing `_extract_helper()` already returns the full block,
so the new defs exec alongside; just give the `logger` stub an `.error` method.

```python
class _Logger:
    def warning(self, *a, **k): pass
    def error(self, *a, **k):
        self.errors.append(a)
    errors = []

def test_engage_marker_on_disengaged_tree_no_record(tmp_path, monkeypatch):
    # Same as the existing disengaged-tree test, but ALSO assert an `engage-fail`
    # stage marker fired (the chase is never silent).
    # -> capture logger.error calls; assert tag "engage-fail" appears, no record.

def test_record_written_marker_on_success(tmp_path, monkeypatch):
    # The ON happy-path test additionally asserts a `record-written` stage marker
    # fired exactly once for the event.

def test_worker_env_gate_marker_env_channel(tmp_path, monkeypatch):
    # FR13_GDN_SUBOP_MAB=1 in env + absent sidecar -> the gate logs
    # `FR13_SUBOP_STAGE=worker-env ... channel=env`.

def test_worker_env_gate_marker_sidecar_channel(tmp_path, monkeypatch):
    # env absent, sidecar flag=1 -> channel=sidecar.

def test_stage_marker_once_dedup(tmp_path, monkeypatch):
    # Two calls with once=True on the same tag -> exactly one logger.error; the
    # counter increments to 2 (record-count discriminator stays live).
```

### 4.3 Default-OFF byte-identity (extend the existing additive test)

```python
def test_stage_markers_are_noop_when_off() -> None:
    # All new markers are guarded behind _fr13_gdn_subop_mab_enabled(); when the
    # flag is OFF the call-site probe block is a pure no-op.  Assert the scan-site
    # callsite-skip block is wholly inside an enabled() guard.
    assert re.search(
        r"if _fr13_gdn_subop_mab_enabled\(\) and not use_fr10_tree:",
        SRC,
    )
    # The existing test_patch_is_purely_additive_no_needle_deleted still passes:
    # the fused_post_conv_prep call + the FR12 needles are untouched.
```

Run: `python -m pytest tests/test_fr13_gdn_subop_mab_wiring.py -q` — expect the
existing 14 + the new ~9 to pass (target ~20+ green; the prior "11/11" baseline
grows). No GPU.

---

## 5. NEXT-GPU-RUN PLAN (worker-env gate + stage-marker checks)

Boot via the locked launcher with `FR13_GDN_SUBOP_MAB=1`
(`scripts/fr13_launch_forked_fa2_tree_server.sh`); drive with
`scripts/fr13_gdn_subop_mab_drive.py` (sets `fr10_decode_mode=tree_mtp`); reduce
with `scripts/fr13_gdn_subop_mab_reduce.py`.

Stage gate (grep the worker log — each marker is one ERROR line):

1. **STAGE (i) worker-env** — `FR13_SUBOP_STAGE=worker-env ... channel=env`
   (or `sidecar`) with `proc_self_has_master=True`. If `channel=NONE` →
   env-to-worker regressed; STOP, do not trust any number (this is the failure
   #2/#3 detector). Independently confirm with
   `cat /proc/<EngineCore-pid>/environ | tr '\0' '\n' | grep FR13_GDN_SUBOP_MAB`.
2. **STAGE (ii) call-site** — if NO `record-written` appears, look for
   `FR13_SUBOP_STAGE=callsite-skip`. Its message names the failing precondition
   (`FR10_ENABLE_TREE_GDN` / `decode_mode` / `tree_parent_set` /
   `num_spec_decodes`). This is the failure-#4 detector: a non-`tree_mtp`
   request or a missing tree env shows up explicitly instead of as silence.
3. **STAGE (iii) engagement** — `FR13_SUBOP_STAGE=engaged ...` should appear once
   per captured verify event; `engage-fail` / `arm-fail` (with the exact reason)
   if a guard or arm raised. A reached-but-failing instrument is now LOUD.
4. **STAGE (iv) capture** — `FR13_SUBOP_STAGE=record-written ... event=0` and the
   JSONL at `FR13_GDN_SUBOP_MAB_DUMP`. The reducer's existing
   `"NO records (hook did not fire / vacuous)"` line
   (`fr13_gdn_subop_mab_reduce.py:57`) is now backed by the stage markers: if it
   fires, exactly one of (i)/(ii)/(iii) markers explains WHY.

Binding gate (class 8): the live B=1 same-seed repeat remains the first gate of
any live campaign; the offline CPU tests in §4 are necessary, not sufficient.
The deliverable verdict (M10-vs-M5 deep-row RAW max_abs per sub-op) is read only
AFTER STAGE (iv) confirms ≥1 record on a `tree_n==10` engaged event.

---

## 6. DEFAULT-OFF / LOCKED-PATH PRESERVATION (proof)

When `FR13_GDN_SUBOP_MAB != "1"` (no env, no sidecar → `_fr13_gdn_subop_mab_enabled()`
returns False):

- EDIT-1: pure function defs — never called unless `enabled()`; zero runtime
  effect at module import beyond two empty module-globals.
- EDIT-2/2b/2c: live ONLY inside `_fr13_gdn_subop_mab`, whose first line is
  `if not _fr13_gdn_subop_mab_enabled(): return` (:1384, unchanged) — the
  function is itself only called from inside `if _fr13_gdn_subop_mab_enabled():`
  at :3936. OFF → the whole helper is unreachable.
- EDIT-3: the entire inserted block is `if _fr13_gdn_subop_mab_enabled() and
  not use_fr10_tree:` / `elif _fr13_gdn_subop_mab_enabled() and use_fr10_tree:`
  — OFF → both conditions are False → no branch taken → byte-identical to the
  locked path (the next line `if use_fr10_tree:` runs exactly as before).
- The conv-site stash (:1900) is unchanged; OFF → `_fr13_gdn_subop_mab_on` is
  False → no stash, no clone.
- No locked-path needle text is deleted; every edit is an additive insertion or
  a like-for-like swap of a swallow-log line. The `[6,6,4,6]` integration
  fingerprint and the FR13_PIPELINE_LOCK default-ON path are unaffected.

The Front B bounds-guard fix (8cdda4c4/d30755c8) — `_guard_rows`, the prior
conv/ssm bank-validity raises — is untouched (it stays inside the new
post-`deep_row` try in EDIT-2; CPU bounds tests §3.3 in the existing file still
cover it).
