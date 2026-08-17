"""FR14 host rung — the two instrument fixes.

`results/fr14_nvfp4_port_20260816/host_dfwd_characterization.md` closes the host
rung as a null (every briefed lever was already shipped) and replaces it with two
measurement fixes, because the 4-bit budget's two largest blocks were being argued
rather than measured:

* §3 — `overhead_other_ms_per_event` (= step_wall − sfwd − dfwd − cfwd) is reported
  as "host glue, sampler, packer, scheduler gap" and is mostly a weight-bound GEMM:
  the sfwd timer brackets `_model_forward` only, and vLLM calls `compute_logits`
  after it returns, so the verifier head is in none of the three spans.
  `FR13_LFWD_GPU_TIMER` brackets it.
* §5.2 — `FR13_DFWD_SPLIT`, the already-written 3-way drafter split timer, was never
  forwarded into the container by either launcher (only the unrelated
  `FR13_DFWD_SPLIT_NEEDLE` was), so its flag file read "0" in 35/35 runroots and it
  never once engaged.

Both flags are default-off and byte-identical when off.
"""

from __future__ import annotations

import importlib.util
import py_compile
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHERS = (
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts" / "fr14_armb_leg3_launch_nomiddleware.sh",
)

# The verifier-head anchor as it appears in the pinned image's
# vllm/v1/worker/gpu_model_runner.py (vllm/vllm-openai@sha256:3dbe092e...,
# vllm 0.19.2rc1.dev134+gfe9c3d6c5): the non-broadcast, non-pooling, last-rank
# branch of execute_model, which is the only branch a TP=1/PP=1 decode takes.
PINNED_ANCHOR = (
    "                sample_hidden_states = hidden_states[logits_indices]\n"
    "                logits = self.model.compute_logits(sample_hidden_states)\n"
    "            else:\n"
)

# A stub carrying BOTH the sfwd patch's anchors (it installs the module block the
# lfwd helpers live in) and the lfwd anchor, each in its real syntactic setting:
# the lfwd anchor is the tail of an `if` branch inside execute_model, and the
# locals the injected call reads must be in scope at that point. Verified against
# the pinned image's gpu_model_runner.py; `test_anchor_matches_the_pinned_image_shape`
# pins the exact bytes so an image bump fails offline instead of at boot.
STUB = '''\
import torch


class GPUModelRunner:
    def execute_model(self, scheduler_output):
        max_num_scheduled_tokens = 32
        num_tokens_unpadded = 32
        num_reqs = 1
        logits_indices = [0]
        cudagraph_mode = None
        defer_kv_connector_finalize = self.speculative_config is not None
        with (
            set_forward_context(
                self.vllm_config,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
        ):
            model_output = self._model_forward(
                input_ids=None,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            hidden_states = model_output
            if not self.broadcast_pp_output:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                logits = None

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
        )
        return logits
'''


def _load_patcher(monkeypatch, stub_path: Path):
    spec = importlib.util.spec_from_file_location("fr10_patcher_lfwd", PATCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "GPU_MODEL_RUNNER_PATH", stub_path)
    return module


@pytest.fixture()
def stub(tmp_path):
    path = tmp_path / "gpu_model_runner.py"
    path.write_text(STUB)
    return path


# --------------------------------------------------------------------------- #
# §3 — FR13_LFWD_GPU_TIMER                                                     #
# --------------------------------------------------------------------------- #


def test_lfwd_patch_brackets_compute_logits_and_is_idempotent(monkeypatch, stub):
    module = _load_patcher(monkeypatch, stub)
    assert module._patch_gpu_model_runner_sfwd_gpu_timer() is True
    assert module._patch_gpu_model_runner_lfwd_gpu_timer() is True
    # Re-running is a no-op: the sentinel is the injected CALL, not a comment.
    assert module._patch_gpu_model_runner_lfwd_gpu_timer() is False

    text = stub.read_text()
    assert text.count("_fr13_lfwd_ev = _fr13_lfwd_begin(") == 1
    assert text.count("_fr13_lfwd_end(_fr13_lfwd_ev)") == 1

    # The span must OPEN before compute_logits and CLOSE after it -- a span that
    # merely surrounds neighbouring statements would measure the wrong thing.
    begin = text.index("_fr13_lfwd_ev = _fr13_lfwd_begin(")
    call = text.index("logits = self.model.compute_logits(sample_hidden_states)")
    end = text.index("_fr13_lfwd_end(_fr13_lfwd_ev)")
    assert begin < call < end

    # ...and it must not have swallowed the gather that feeds it.
    gather = text.index("sample_hidden_states = hidden_states[logits_indices]")
    assert gather < begin


def test_lfwd_patch_output_compiles(monkeypatch, stub, tmp_path):
    module = _load_patcher(monkeypatch, stub)
    module._patch_gpu_model_runner_sfwd_gpu_timer()
    module._patch_gpu_model_runner_lfwd_gpu_timer()
    py_compile.compile(str(stub), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_lfwd_patch_refuses_without_the_sfwd_module_block(monkeypatch, stub):
    """Ordering is a precondition, not a hope: the helpers this patch calls are
    installed by the sfwd patch, so running it alone must fail loud."""
    module = _load_patcher(monkeypatch, stub)
    with pytest.raises(RuntimeError, match="_fr13_lfwd_begin helper missing"):
        module._patch_gpu_model_runner_lfwd_gpu_timer()


def test_lfwd_patch_refuses_an_ambiguous_anchor(monkeypatch, tmp_path):
    """Two verifier-head branches would mean the injected span covers only one of
    them and silently under-reports. Refuse rather than pick."""
    path = tmp_path / "gpu_model_runner.py"
    path.write_text(STUB + "\n\n" + STUB.split("import torch", 1)[1])
    module = _load_patcher(monkeypatch, path)
    module._patch_gpu_model_runner_sfwd_gpu_timer()
    with pytest.raises(RuntimeError, match="expected exactly one verifier"):
        module._patch_gpu_model_runner_lfwd_gpu_timer()


def test_lfwd_patch_is_registered_in_the_patch_steps_after_sfwd():
    text = PATCHER.read_text()
    sfwd = text.index("_patch_gpu_model_runner_sfwd_gpu_timer()),")
    lfwd = text.index("_patch_gpu_model_runner_lfwd_gpu_timer()),")
    assert sfwd < lfwd, "lfwd must be registered after the sfwd module block"


def test_lfwd_helpers_are_default_off_and_pure_decode_gated():
    text = PATCHER.read_text()
    helper = text[text.index("def _fr13_lfwd_begin("):text.index("def _fr13_lfwd_end(")]
    # Strict, default-off env read -- never a truthiness test.
    assert 'environ.get("FR13_LFWD_GPU_TIMER", "0") != "1"' in helper
    # The same pure-decode predicate the sfwd timer uses. Without it the span
    # would average in chunked-prefill head reads over a different row count and
    # stop being comparable to the drafter/committer spans.
    assert "_fr13_sfwd_is_pure_decode(" in helper
    # Recording cuda events inside a graph capture invalidates the capture.
    assert "is_current_stream_capturing()" in helper


def test_lfwd_timer_is_a_span_timer_with_its_own_counter_and_sidecar():
    text = PATCHER.read_text()
    block = text[text.index("def _fr13_lfwd_timer("):text.index("def _fr13_lfwd_begin(")]
    assert '"FR13_LFWD_GPU_TIMER"' in block
    assert '"vllm:fr13_lmhead_gpu_seconds"' in block
    assert '"FR13_LFWD_GPU_TIMER_JSON"' in block
    assert '"lmhead"' in block
    # Same machinery as the drafter/committer twins => same sidecar schema, so
    # the existing reducers and the characterization note read it unchanged.
    assert "_Fr13SpanTimer(" in block


# --------------------------------------------------------------------------- #
# §5.2 — FR13_DFWD_SPLIT reaches the container                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_forwards_the_instrument_flags(launcher):
    text = launcher.read_text()
    # The defect: only the NEEDLE was forwarded, so the flag the patcher's main()
    # reads at pid 1 never existed inside the container.
    assert '-e FR13_DFWD_SPLIT="${FR13_DFWD_SPLIT:-0}"' in text
    assert '-e FR13_LFWD_GPU_TIMER="${FR13_LFWD_GPU_TIMER:-0}"' in text
    assert '-e FR13_LFWD_GPU_TIMER_JSON="${FR13_LFWD_GPU_TIMER_JSON:-}"' in text


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_dfwd_split_json_default_is_not_empty(launcher):
    """`os.environ.get(k, default)` returns "" for a var that is SET-but-empty, so
    forwarding this one as ":-" would clobber _Fr13DfwdSplit.dump()'s own default
    and drop the sidecar into ".<pid>" in the cwd instead of the bind-mounted
    /logs. The sibling *_JSON flags may be empty; this one may not."""
    text = launcher.read_text()
    assert '-e FR13_DFWD_SPLIT_JSON="${FR13_DFWD_SPLIT_JSON:-}"' not in text
    assert (
        '-e FR13_DFWD_SPLIT_JSON="${FR13_DFWD_SPLIT_JSON:-/logs/fr13_dfwd_split.json}"'
        in text
    )
    patcher = PATCHER.read_text()
    assert '"FR13_DFWD_SPLIT_JSON", "/logs/fr13_dfwd_split.json"' in patcher


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_validates_the_instrument_flags_strictly(launcher):
    text = launcher.read_text()
    assert "for _fr13_instrument_flag in FR13_DFWD_SPLIT FR13_LFWD_GPU_TIMER; do" in text
    block = text[text.index("for _fr13_instrument_flag in"):]
    block = block[:block.index("unset _fr13_instrument_flag")]
    assert "0|1) ;;" in block
    assert "must be 0 or 1" in block


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_parses(launcher):
    assert shutil.which("bash"), "bash is required"
    import subprocess

    proc = subprocess.run(
        ["bash", "-n", str(launcher)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_no_comment_line_sits_inside_a_backslash_continuation(launcher):
    """`bash -n` CANNOT catch this and it is silently catastrophic.

    A `#` comment placed between two `\\`-continued lines does not comment itself
    out of the command -- the continuation joins it onto the previous line, the
    `#` starts a real comment there, and the enclosing `docker run` TERMINATES at
    that point. Every remaining `-e` line then runs as its own bogus command.
    This was a live defect in the first draft of this very change, and it parsed
    clean, so the guard is a test rather than a review habit.
    """
    lines = launcher.read_text().splitlines()
    offenders = [
        (i + 1, line)
        for i, line in enumerate(lines)
        if i > 0
        and lines[i - 1].rstrip().endswith("\\")
        and line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "comment inside a line continuation truncates the command: "
        + "; ".join(f"{launcher.name}:{n}: {t.strip()}" for n, t in offenders)
    )


# --------------------------------------------------------------------------- #
# byte-safety when off                                                         #
# --------------------------------------------------------------------------- #


def test_injected_source_is_inert_when_the_flag_is_off(monkeypatch, stub):
    """Default-off must mean *no events created*, not merely "not recorded":
    _fr13_lfwd_begin returns None before touching cuda, and _fr13_lfwd_end
    returns immediately on None."""
    module = _load_patcher(monkeypatch, stub)
    module._patch_gpu_model_runner_sfwd_gpu_timer()
    text = stub.read_text()

    namespace: dict[str, object] = {}
    exec(compile(text, "<fr14-lfwd>", "exec"), namespace)

    monkeypatch.delenv("FR13_LFWD_GPU_TIMER", raising=False)
    begin = namespace["_fr13_lfwd_begin"]
    end = namespace["_fr13_lfwd_end"]
    # Pure-decode arguments, so the ONLY thing that can gate it here is the flag.
    assert begin(32, 32, 1, 32) is None
    assert end(None) is None
    # And an explicit "0" is just as off as an unset var.
    monkeypatch.setenv("FR13_LFWD_GPU_TIMER", "0")
    assert begin(32, 32, 1, 32) is None


def test_flag_off_leaves_the_pure_decode_predicate_unreached(monkeypatch, stub):
    """The flag is checked BEFORE the predicate, so an off arm pays nothing --
    not even the predicate call -- on every step of a serve."""
    module = _load_patcher(monkeypatch, stub)
    module._patch_gpu_model_runner_sfwd_gpu_timer()
    namespace: dict[str, object] = {}
    exec(compile(stub.read_text(), "<fr14-lfwd>", "exec"), namespace)

    calls = []
    real = namespace["_fr13_sfwd_is_pure_decode"]
    namespace["_fr13_sfwd_is_pure_decode"] = lambda *a: (calls.append(a), real(*a))[1]

    monkeypatch.setenv("FR13_LFWD_GPU_TIMER", "0")
    namespace["_fr13_lfwd_begin"](32, 32, 1, 32)
    assert calls == []


def test_helper_source_order_puts_the_env_check_first():
    text = PATCHER.read_text()
    helper = text[text.index("def _fr13_lfwd_begin("):text.index("def _fr13_lfwd_end(")]
    env_at = helper.index('environ.get("FR13_LFWD_GPU_TIMER"')
    pred_at = helper.index("_fr13_sfwd_is_pure_decode(")
    cap_at = helper.index("is_current_stream_capturing()")
    assert env_at < cap_at < pred_at


def test_anchor_matches_the_pinned_image_shape():
    """The anchor is a literal against a pinned image. Keep the exact bytes in the
    test so a future image bump fails HERE, loudly and offline, instead of at
    boot in the middle of a serve."""
    text = PATCHER.read_text()
    for line in PINNED_ANCHOR.splitlines():
        assert re.search(re.escape(line) + r"\\n", text), line
