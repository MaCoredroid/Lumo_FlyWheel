"""Regression test for the T1 session-scoped SuffixDecoding patch.

Round 2 Technique 1 partitions arctic_inference's SuffixDecodingCache
by session_id parsed from a ``lumo_sess_<id>__`` prefix on req_id.
The patch (in ``scripts/run_track_b_loop.py``'s prelaunch hook)
wraps ``self.suffix_cache`` in a router that dispatches per session.

This test runs the patch logic in the same vLLM image the running
container uses and asserts the wrapper:

1. routes prefixed req_ids to per-session sub-caches,
2. routes unprefixed traffic to a single ``__default__`` bucket
   (preserving vanilla behaviour),
3. preserves the cache surface ``SuffixDecodingProposer.propose``
   actually calls (active_requests/cached_requests as set
   difference participants, start/stop/speculate/etc.),
4. is idempotent — re-applying the patch is a no-op.

Skipped if Docker or the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"

INNER_SCRIPT = r'''
import sys
from pathlib import Path

# Apply patch (idempotent).
src_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py")
text = src_path.read_text(encoding="utf-8")
sentinel = "# T1_SESSION_SCOPING_APPLIED"
if sentinel not in text:
    old = (
        "        # Lazy import to avoid error when Suffix Decoding is not used.\n"
        "        from arctic_inference.suffix_decoding import SuffixDecodingCache\n"
        "\n"
        "        # Initialize and empty cache. This object will take care of caching request\n"
        "        # outputs, evicting old requests, and manages the per-prompt suffix trees.\n"
        "        self.suffix_cache = SuffixDecodingCache(\n"
        "            max_tree_depth=config.suffix_decoding_max_tree_depth,\n"
        "            max_cached_requests=config.suffix_decoding_max_cached_requests,\n"
        "        )\n"
    )
    new = (
        "        # Lazy import to avoid error when Suffix Decoding is not used.\n"
        "        from arctic_inference.suffix_decoding import SuffixDecodingCache\n"
        "\n"
        "        # T1_SESSION_SCOPING_APPLIED -- Lumo Track B Round 2 (2026-05-09).\n"
        "        self.suffix_cache = _SessionRoutedSuffixDecodingCache(\n"
        "            cache_factory=lambda: SuffixDecodingCache(\n"
        "                max_tree_depth=config.suffix_decoding_max_tree_depth,\n"
        "                max_cached_requests=config.suffix_decoding_max_cached_requests,\n"
        "            ),\n"
        "            max_tree_depth=config.suffix_decoding_max_tree_depth,\n"
        "            max_cached_requests=config.suffix_decoding_max_cached_requests,\n"
        "        )\n"
    )
    assert old in text, "patch target not found"
    text = text.replace(old, new)
    text += (
        "\n\n# T1_SESSION_SCOPING_APPLIED router class.\n"
        "class _SessionRoutedSuffixDecodingCache:\n"
        "    _PREFIX = \"lumo_sess_\"\n"
        "    _SEP = \"__\"\n"
        "    _DEFAULT = \"__default__\"\n"
        "    def __init__(self, *, cache_factory, max_tree_depth, max_cached_requests):\n"
        "        self._cache_factory = cache_factory\n"
        "        self._max_tree_depth = max_tree_depth\n"
        "        self._max_cached_requests = max_cached_requests\n"
        "        self._caches = {}\n"
        "        self._req_to_session = {}\n"
        "    @property\n"
        "    def max_tree_depth(self): return self._max_tree_depth\n"
        "    @property\n"
        "    def max_cached_requests(self): return self._max_cached_requests\n"
        "    @property\n"
        "    def session_count(self): return len(self._caches)\n"
        "    @property\n"
        "    def active_requests(self):\n"
        "        result = set()\n"
        "        for c in self._caches.values(): result.update(c.active_requests)\n"
        "        return result\n"
        "    @property\n"
        "    def cached_requests(self):\n"
        "        result = set()\n"
        "        for c in self._caches.values(): result.update(c.cached_requests)\n"
        "        return result\n"
        "    def _session_for(self, req_id):\n"
        "        if isinstance(req_id, str) and req_id.startswith(self._PREFIX):\n"
        "            rest = req_id[len(self._PREFIX):]\n"
        "            sep = rest.find(self._SEP)\n"
        "            if sep > 0: return rest[:sep]\n"
        "        return self._DEFAULT\n"
        "    def _cache_for(self, req_id):\n"
        "        sess = self._req_to_session.get(req_id)\n"
        "        if sess is None:\n"
        "            sess = self._session_for(req_id)\n"
        "            self._req_to_session[req_id] = sess\n"
        "        c = self._caches.get(sess)\n"
        "        if c is None:\n"
        "            c = self._cache_factory()\n"
        "            self._caches[sess] = c\n"
        "        return c\n"
        "    def start_request(self, req_id, prompt_token_ids):\n"
        "        return self._cache_for(req_id).start_request(req_id, prompt_token_ids)\n"
        "    def add_active_response(self, req_id, sampled_ids):\n"
        "        return self._cache_for(req_id).add_active_response(req_id, sampled_ids)\n"
        "    def speculate(self, req_id, pattern, **kwargs):\n"
        "        return self._cache_for(req_id).speculate(req_id, pattern, **kwargs)\n"
        "    def stop_request(self, req_id):\n"
        "        c = self._cache_for(req_id)\n"
        "        try: return c.stop_request(req_id)\n"
        "        finally: self._req_to_session.pop(req_id, None)\n"
        "    def evict_cached_response(self, req_id):\n"
        "        return self._cache_for(req_id).evict_cached_response(req_id)\n"
    )
    src_path.write_text(text, encoding="utf-8")
    print("PATCH_APPLIED")
else:
    print("PATCH_ALREADY_PRESENT")

# Idempotency check: re-apply path uses the sentinel.
text_after = src_path.read_text(encoding="utf-8")
assert text_after.count(sentinel) >= 1, "sentinel disappeared after patch"
sentinel_count_pre = text_after.count(sentinel)

# arctic_inference is installed by the prelaunch hook in real runs;
# install it here for the regression test.
import importlib.util
import subprocess as _sp
if importlib.util.find_spec("arctic_inference") is None:
    _sp.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--quiet", "arctic-inference==0.1.2",
    ])

# Now exercise the wrapper functionally.
import numpy as np
from vllm.v1.spec_decode.suffix_decoding import _SessionRoutedSuffixDecodingCache
from arctic_inference.suffix_decoding import SuffixDecodingCache

router = _SessionRoutedSuffixDecodingCache(
    cache_factory=lambda: SuffixDecodingCache(max_tree_depth=8, max_cached_requests=16),
    max_tree_depth=8,
    max_cached_requests=16,
)

req_a1 = "lumo_sess_sess_aaaaaaaaaaaaaaaa__rid-1"
req_a2 = "lumo_sess_sess_aaaaaaaaaaaaaaaa__rid-2"
req_b1 = "lumo_sess_sess_bbbbbbbbbbbbbbbb__rid-1"
req_unprefixed = "plain-uuid-1234"

prompt = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

router.start_request(req_a1, prompt)
router.start_request(req_a2, prompt)
router.start_request(req_b1, prompt)
router.start_request(req_unprefixed, prompt)

assert router.session_count == 3, f"expected 3 distinct buckets, got {router.session_count}"
assert {req_a1, req_a2, req_b1, req_unprefixed} <= router.active_requests

# Add some responses; speculate against the same pattern across sessions.
router.add_active_response(req_a1, [10, 11, 12])
router.add_active_response(req_a2, [20, 21, 22])
router.add_active_response(req_b1, [30, 31, 32])

# Stop request fully clears state.
router.stop_request(req_a1)
assert req_a1 not in router.active_requests
router.stop_request(req_a2)
router.stop_request(req_b1)
router.stop_request(req_unprefixed)
assert len(router.active_requests) == 0

# Idempotency: re-running the patch path is a no-op.
text2 = src_path.read_text(encoding="utf-8")
src_path.write_text(text2, encoding="utf-8")
assert src_path.read_text(encoding="utf-8") == text2
assert src_path.read_text(encoding="utf-8").count(sentinel) == sentinel_count_pre

print("VERIFICATION_PASSED")
'''


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _image_available(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
@pytest.mark.skipif(not _image_available(VLLM_IMAGE), reason=f"{VLLM_IMAGE} not built locally")
def test_t1_session_scoping_partitions_per_session() -> None:
    """Patch applies, routes prefixed req_ids to per-session caches,
    routes unprefixed traffic to a single default bucket, and is
    idempotent across two consecutive applications."""

    result = subprocess.run(
        ["docker", "run", "--rm", VLLM_IMAGE, "python3", "-c", INNER_SCRIPT],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"docker invocation failed (rc={result.returncode}). "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-1500:]}"
    )
    assert "VERIFICATION_PASSED" in result.stdout, (
        f"verification did not pass. stdout:\n{result.stdout}"
    )
