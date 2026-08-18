#!/usr/bin/env python3
"""Validate qrow32 B1 live gates and issue the production credentials.

Two arms are credentialed here, from two different shapes of evidence:

* ``nosplit`` (``issue``/``verify``) binds a live A/B PASS directly.
* ``gqa_pair`` (``issue-gqa-pair``/``verify-gqa-pair``) binds the SEALED byte
  gate artifact plus the live A/B result that gate references by digest. Both
  are supplied by path, never by runroot, so the credential cannot be issued
  against a stale gate: the gate's ``source_commit`` must equal the production
  plumbing commit, which means the gate has to be re-run whenever the patcher
  that serves the candidate changes.

Issuing runs on the HOST, where git resolves the patcher's repository and HEAD.
Verifying runs INSIDE the pinned runtime image, which ships no git binary, so
``verify`` proves the patcher by digest instead -- see
``validate_patch_source_digest``. Both directions are fail-closed; only the
mechanism differs, because only the host has a repository to interrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_live_paged_ab.v2"
SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_production_pass.v2"
# The sealed B1 GQA-pair byte gate and the credential issued from it. The gate
# artifact is the authenticated wrapper (health, traffic audit, container and
# diagnostic bindings); the raw-byte evidence itself lives in the live A/B
# result the gate binds by digest, which is why both are required to issue.
GQA_PAIR_GATE_SCHEMA = "fr13.fixed32.fa2_qrow32_gqa_pair_k64_b1_live_verification.v1"
GQA_PAIR_GATE_SCHEMA_K0 = (
    "fr13.fixed32.fa2_qrow32_gqa_pair_full_vocab_b1_live_verification.v1"
)


def _gate_schema_for(payload):
    """The artifact TYPE name for the shape this artifact was earned in.

    The schema name carried a literal "k64", so a K0 credential would have
    announced itself as a K64 one in its own type string. Both names are
    pinned here and selected by the declared profile, so the name cannot
    disagree with the identity fields beside it.
    """
    profile = payload.get("qualification_profile", "k64_root")
    if profile == "full_vocab":
        return GQA_PAIR_GATE_SCHEMA_K0
    return GQA_PAIR_GATE_SCHEMA



def _draft_vocab_identity_is_consistent(payload):
    """A gate artifact must describe the shape it was actually earned in.

    This replaced a pair of `!= 1` / `!= 65536` literals. Those made the
    validator demand K64 -- so a K0 gate could only pass by MISREPORTING itself
    as K64, which is exactly what happened once. The identity is now checked
    for internal consistency against the profile the artifact declares, so both
    shapes are expressible and neither can masquerade as the other.
    """
    profile = payload.get("qualification_profile", "k64_root")
    root = payload.get("draft_vocab_root")
    k = payload.get("draft_vocab_k")
    if profile == "k64_root":
        return root == 1 and k == 65536
    if profile == "full_vocab":
        return root == 0 and k == 0
    return False
GQA_PAIR_SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_gqa_pair_production_pass.v1"
GQA_PAIR_GATE_TOPOLOGY = "hydra27_fixed32"
# The byte gate ran ONE authenticated real task. That is narrower than the
# exact4 set production serves, and the credential says so rather than
# laundering the difference: what bridges the gap is the selector's per-call
# exact-geometry check, which hard-raises instead of serving any decode whose
# shape is not the one the gate qualified.
GQA_PAIR_GATE_TASK_IDS = ("astropy__astropy-12907",)
GQA_PAIR_GATE_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
GQA_PAIR_SERVED_RETURN = "qrow16 captured graph output unchanged"
GQA_PAIR_CREDENTIAL_BASIS = (
    "raw-byte output/LSE equality between the qrow16 incumbent and GQA-pair "
    "dispatches across all 16 tree layers of the final FULL B1 graph, on one "
    "authenticated real SWE-Verified task at this source commit"
)
GQA_PAIR_REQUIRED_RUNTIME = (
    "Hydra27 fixed32 K64 ROOT=1 B1 physical32 FULL graph"
)
GQA_PAIR_REQUIRED_RUNTIME_K0 = (
    "Hydra27 fixed32 full-vocab K0 ROOT=0 B1 physical32 FULL graph"
)


def _required_runtime_for(payload):
    """The runtime this credential is valid for, in the shape it was earned in.

    A full_vocab credential that says "K64 ROOT=1" in its own required_runtime
    is the same misdescription class as the schema name and the draft_vocab_*
    fields: a reader checking what a credential authorizes would be told the
    wrong workload. Selected by the declared profile, and validated against the
    same selection, so the string cannot drift from the identity beside it.
    """
    if payload.get("qualification_profile", "k64_root") == "full_vocab":
        return GQA_PAIR_REQUIRED_RUNTIME_K0
    return GQA_PAIR_REQUIRED_RUNTIME
GQA_PAIR_PRODUCTION_SCOPE = "qrow32 B1 GQA-pair exact tree attention only"
ARM = "nosplit"
VISIBILITY_ARM = "visibility"
GQA_PAIR_ARM = "gqa_pair"
# FR14 Tier-B. Same GQA-pair traits, context walk split four ways and
# re-reduced by FA2's combine kernel. Gate-only, and structurally so: no byte
# gate can ever qualify it, because its reduction topology differs from the
# reference's by design.
SPLITK_ARM = "gqa_pair_splitk"
SELECTOR_SENTINEL = 1179791668
GQA_PAIR_SELECTOR_SENTINEL = 1179791670
SPLITK_SELECTOR_SENTINEL = 1179791671
SPLITK_NUM_SPLITS = 4
QROW16_REFERENCE_SENTINEL = 1179791667
NUM_SPLITS = 0
PATCH_SOURCE_RELATIVE = "scripts/fr13_patch_fa2_tree_bias.py"
LIVE_ARMS = {
    ARM: {
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": "qrow32 B1 nosplit exact geometry; no fallback",
    },
    "split2": {
        "selector_sentinel": 1179791669,
        "num_splits": 2,
        "split_scratch_allocation": (
            "stock FA2 set_params_splitkv via num_splits=2"
        ),
        "candidate_dispatch": "qrow32 B1 split2 exact geometry; no fallback",
    },
    VISIBILITY_ARM: {
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": (
            "qrow32 B1 fixed32 visibility exact geometry; no fallback"
        ),
    },
    GQA_PAIR_ARM: {
        "selector_sentinel": GQA_PAIR_SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": "qrow32 B1 GQA-pair exact geometry; no fallback",
    },
    SPLITK_ARM: {
        "selector_sentinel": SPLITK_SELECTOR_SENTINEL,
        "num_splits": SPLITK_NUM_SPLITS,
        "split_scratch_allocation": (
            f"stock FA2 set_params_splitkv via num_splits={SPLITK_NUM_SPLITS}"
        ),
        "candidate_dispatch": (
            "qrow32 B1 GQA-pair split-K exact geometry; no fallback"
        ),
    },
}
CANDIDATE_SHA256 = (
    "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
)
CANDIDATE_SIZE = 300_154_616
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "3313e28a5512a1399dbefbfec27f3e84540b827770d138a9db8457e414f64d37"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu": (
        "00f38d6ed68580d0e9c957ed0620d9655b1e4bf24bf16e970e693341bfa60cc1"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu": (
        "223542ecf9bcc8837022aaceeca7468e4a8c866b528c4327c68f924dc4ab344d"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "5829b5c6832ce962fc6864f491901ed68e8a24b165d5cff914e94c38b0de8177"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
SOURCE_CLOSURE_SHA256 = (
    "22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff"
)
VISIBILITY_CANDIDATE_SHA256 = (
    "c5ab32a6ae4e615f1e77a4997db5429152053c549e761fb11d90b33bb3959a79"
)
VISIBILITY_CANDIDATE_SIZE = 300_200_192
VISIBILITY_SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "a84a1c22234f2787cd0c27307e2dbbb5b8bc4859072533070be021710ec7d55e"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu": (
        "a022da77fc020891125aad1c40f82ebeb86a490660ca121b4a346a14c701e04b"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu": (
        "223542ecf9bcc8837022aaceeca7468e4a8c866b528c4327c68f924dc4ab344d"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "5e48444ff68c75dee9227570735b15a4273c064c574d21a4c1953019bb9eb876"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
VISIBILITY_SOURCE_CLOSURE_SHA256 = (
    "a30eca031cd5067133e6278527787c5987635670930e5840ac983f66b088e4fc"
)
SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu",
)
# The GQA-pair B1 codegen emits one translation unit, not the no-split/split2
# pair, so its modified-source set is its own.
GQA_PAIR_SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu",
)
# flash_fwd_kernel.h below is byte-identical to the header in the B4 GQA-pair
# closure that passed the dual byte gate (9c3f9e75...), so the paired
# ((query_row, head_in_pair), column) address layout under test at B1 is the
# same code that was proven arithmetically invariant at B4. flash.h, utils.h and
# flash_api_torch_lib.cpp are byte-identical to the qrow32 B1 closure above.
GQA_PAIR_SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "2c78d7bd17f8406956781b4e7bd70722ba147c651ec4dc0030ed3685a462b634"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu": (
        "edc5e2a9116142a5efd6f4e21ee80c2da3caf823a476595d7a72877e50cfb105"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "4f08741030c46d7e1ef1b88a10d4946f625559fedd7658c3b288e0d7a5d58d13"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
GQA_PAIR_SOURCE_CLOSURE_SHA256 = (
    "172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4"
)
GQA_PAIR_CANDIDATE_SHA256 = (
    "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
)
GQA_PAIR_CANDIDATE_SIZE = 299_815_552
# The FR14 split-K closure. flash_api_torch_lib.cpp, flash.h and utils.h are
# byte-identical to the GQA-pair closure above; flash_api.cpp and
# flash_fwd_kernel.h differ ONLY by the split-K dispatch, its split-scratch
# allocation, and the paired O/LSE tensors learning to address the stock split
# accumulators. The candidate .so is pinned from the qualified build.
SPLITK_SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_splitk_b1_hdim256_bf16_sm80.cu",
)
SPLITK_SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "5799e231fcf0a9c926523e9e77abee876f00f78a84502e3612842529bd06851c"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_splitk_b1_hdim256_bf16_sm80.cu": (
        "c526c4d42b1d1c6601fde785f6881c11dfeab4f3b1fa7167bde3b2ab866dd126"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "f9c313437e972b27081a8bd4ae523aa845bffdaa368e1a36d0c1b979e96d7558"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
SPLITK_SOURCE_CLOSURE_SHA256 = (
    "54287f754b8581d445196edcaa71356acb70860e6f6247ae5dd96094f116865b"
)
SPLITK_CANDIDATE_SHA256 = ""
SPLITK_CANDIDATE_SIZE = 0
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
HEX = frozenset("0123456789abcdef")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload, raw


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase commit")
    return value


def _regular(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return info


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.rstrip("\n")


def _candidate_contract(arm: str) -> dict[str, Any]:
    if arm == GQA_PAIR_ARM:
        if not GQA_PAIR_CANDIDATE_SHA256 or not GQA_PAIR_CANDIDATE_SIZE:
            raise ValueError(
                "qrow32 B1 GQA-pair binary is not pinned: fill "
                "GQA_PAIR_CANDIDATE_SHA256 and GQA_PAIR_CANDIDATE_SIZE from "
                "the build attestation before gating this arm"
            )
        return {
            "sha256": GQA_PAIR_CANDIDATE_SHA256,
            "size": GQA_PAIR_CANDIDATE_SIZE,
            "source_files": GQA_PAIR_SOURCE_FILES,
            "source_closure_sha256": GQA_PAIR_SOURCE_CLOSURE_SHA256,
        }
    if arm == SPLITK_ARM:
        if not SPLITK_CANDIDATE_SHA256 or not SPLITK_CANDIDATE_SIZE:
            raise ValueError(
                "qrow32 B1 GQA-pair split-K binary is not pinned: fill "
                "SPLITK_CANDIDATE_SHA256 and SPLITK_CANDIDATE_SIZE from the "
                "build attestation before gating this arm"
            )
        return {
            "sha256": SPLITK_CANDIDATE_SHA256,
            "size": SPLITK_CANDIDATE_SIZE,
            "source_files": SPLITK_SOURCE_FILES,
            "source_closure_sha256": SPLITK_SOURCE_CLOSURE_SHA256,
        }
    if arm == VISIBILITY_ARM:
        return {
            "sha256": VISIBILITY_CANDIDATE_SHA256,
            "size": VISIBILITY_CANDIDATE_SIZE,
            "source_files": VISIBILITY_SOURCE_FILES,
            "source_closure_sha256": VISIBILITY_SOURCE_CLOSURE_SHA256,
        }
    if arm not in {ARM, "split2"}:
        raise ValueError(
            "qrow32 live arm must be nosplit, split2, visibility, gqa_pair, "
            "or gqa_pair_splitk"
        )
    return {
        "sha256": CANDIDATE_SHA256,
        "size": CANDIDATE_SIZE,
        "source_files": SOURCE_FILES,
        "source_closure_sha256": SOURCE_CLOSURE_SHA256,
    }


def _source_status(arm: str) -> tuple[str, ...]:
    """The modified-source set the codegen produces for this arm."""
    if arm == GQA_PAIR_ARM:
        return GQA_PAIR_SOURCE_STATUS
    if arm == SPLITK_ARM:
        return SPLITK_SOURCE_STATUS
    return SOURCE_STATUS


def splitk_source_contract() -> dict[str, Any]:
    """The split-K source closure, pinnable before the binary exists."""
    return {
        "source_files": SPLITK_SOURCE_FILES,
        "source_closure_sha256": SPLITK_SOURCE_CLOSURE_SHA256,
        "source_status": SPLITK_SOURCE_STATUS,
    }


def gqa_pair_source_contract() -> dict[str, Any]:
    """The GQA-pair source closure, pinnable before the binary exists.

    The codegen fixes the source, so `validate-source` can qualify a
    regenerated tree ahead of the build; only the .so identity waits.
    """
    return {
        "source_files": GQA_PAIR_SOURCE_FILES,
        "source_closure_sha256": GQA_PAIR_SOURCE_CLOSURE_SHA256,
        "source_status": GQA_PAIR_SOURCE_STATUS,
    }


def validate_candidate(
    candidate_so: Path, expected_sha256: str, *, arm: str = ARM
) -> dict[str, Any]:
    contract = _candidate_contract(arm)
    expected_sha256 = _sha256(expected_sha256, "candidate SO")
    if expected_sha256 != contract["sha256"]:
        raise ValueError("candidate SO is not the pinned qrow32 B1 binary")
    info = _regular(candidate_so, "candidate SO")
    if info.st_size != contract["size"] or sha256_file(candidate_so) != contract["sha256"]:
        raise ValueError("candidate SO identity mismatch")
    return {"size": contract["size"], "sha256": contract["sha256"]}


def validate_source_closure(source_root: Path, *, arm: str = ARM) -> dict[str, Any]:
    # The GQA-pair source closure is pinned before its binary exists, so source
    # qualification must not depend on the .so identity.
    if arm == GQA_PAIR_ARM:
        contract = gqa_pair_source_contract()
    elif arm == SPLITK_ARM:
        contract = splitk_source_contract()
    else:
        contract = _candidate_contract(arm)
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("FA2 source root must be a non-symlink directory")
    head = _git(source_root, "rev-parse", "HEAD")
    if head != FA2_HEAD:
        raise ValueError("FA2 source head drifted")
    status = tuple(
        line
        for line in _git(
            source_root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    if status != _source_status(arm):
        raise ValueError("FA2 modified source set drifted")
    records: dict[str, str] = {}
    for relative, expected in contract["source_files"].items():
        path = source_root / relative
        _regular(path, f"FA2 source {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"FA2 source hash drifted: {relative}")
        records[relative] = actual
    closure = {"fa2_head": head, "files": records}
    if _digest(canonical_bytes(closure)) != contract["source_closure_sha256"]:
        raise ValueError("FA2 source closure digest drifted")
    closure["canonical_sha256"] = contract["source_closure_sha256"]
    return closure


def validate_patch_source(
    patch_source: Path, *, expected_source_commit: str
) -> dict[str, str]:
    _regular(patch_source, "patch source")
    expected_source_commit = _commit(expected_source_commit, "source commit")
    repo = Path(_git(patch_source.parent, "rev-parse", "--show-toplevel"))
    head = _git(repo, "rev-parse", "HEAD")
    if head != expected_source_commit:
        raise ValueError("patch source commit drifted")
    relative = patch_source.resolve().relative_to(repo.resolve()).as_posix()
    if relative != PATCH_SOURCE_RELATIVE:
        raise ValueError("patch source path drifted")
    return {"source_commit": head, "patch_source_sha256": sha256_file(patch_source)}


def validate_patch_source_digest(
    patch_source: Path,
    *,
    expected_source_commit: str,
    expected_patch_source_sha256: str,
) -> dict[str, str]:
    """The git-free patch-source identity check, used to VERIFY a credential.

    The pinned runtime image ships no git binary, so a credential cannot be
    re-derived from the repository inside the container. It does not need to
    be. ``issue`` already bound this credential on the host, where git proved
    the patcher lives at the expected path of a repository whose HEAD is the
    expected commit. What the container must establish is narrower and, for the
    risk that matters, stricter: that the patcher bytes about to run ARE the
    bytes that binding was made against. A digest proves exactly that, and it
    proves it about the file rather than about a working tree that git would
    have been asked to describe.

    Fail-closed is preserved: the digest is supplied by the caller (the
    launcher, which validated it against the host tree at issue time) AND is
    cross-checked against the digest recorded in the credential body, so a
    swapped patcher fails whether or not the credential was also swapped.
    """
    _regular(patch_source, "patch source")
    expected_source_commit = _commit(expected_source_commit, "source commit")
    expected_patch_source_sha256 = _sha256(
        expected_patch_source_sha256, "patch source"
    )
    if patch_source.name != PATCH_SOURCE_RELATIVE.rsplit("/", 1)[-1]:
        raise ValueError("patch source path drifted")
    observed = sha256_file(patch_source)
    if observed != expected_patch_source_sha256:
        raise ValueError("patch source digest drifted")
    return {
        "source_commit": expected_source_commit,
        "patch_source_sha256": observed,
    }


def _validate_comparison(value: Any, label: str) -> str:
    if not isinstance(value, dict) or (
        not isinstance(value.get("bytes"), int)
        or value["bytes"] <= 0
        or value.get("raw_byte_mismatches") != 0
    ):
        raise ValueError(f"{label} byte comparison drifted")
    reference = _sha256(value.get("reference_sha256"), f"{label} reference")
    candidate = _sha256(value.get("candidate_sha256"), f"{label} candidate")
    if reference != candidate:
        raise ValueError(f"{label} digest comparison drifted")
    return reference


def validate_live_result(
    payload: dict[str, Any],
    *,
    candidate_sha256: str,
    arm: str,
    source_commit: str | None = None,
    patch_source_sha256: str | None = None,
) -> dict[str, Any]:
    arm_contract = LIVE_ARMS.get(arm)
    if arm_contract is None:
        raise ValueError(
            "qrow32 live arm must be nosplit, split2, visibility, or gqa_pair"
        )
    candidate_contract = _candidate_contract(arm)
    if candidate_sha256 != candidate_contract["sha256"]:
        raise ValueError("qrow32 live candidate is not pinned")
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError(f"qrow32 {arm} live result is not a PASS")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXACT4_TASK_IDS[0]
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("physical_rows") != 32
        or not _draft_vocab_identity_is_consistent(payload)
        or payload.get("runtime_mode") != "FULL"
        or payload.get("candidate_so_sha256") != candidate_contract["sha256"]
        or payload.get("candidate_so_size") != candidate_contract["size"]
        or payload.get("arm") != arm
        or payload.get("selector_sentinel") != arm_contract["selector_sentinel"]
        or payload.get("candidate_num_splits") != arm_contract["num_splits"]
        or payload.get("reference_selector_sentinel") != QROW16_REFERENCE_SENTINEL
        or payload.get("reference_dispatch")
        != "qrow16 incumbent exact geometry; no fallback"
        or payload.get("candidate_dispatch")
        != arm_contract["candidate_dispatch"]
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256")
        != candidate_contract["source_closure_sha256"]
        or payload.get("layer_count") != 16
        or payload.get("fallback_allowed") is not False
        or payload.get("served_return") != "qrow16 captured graph output unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("split_scratch_allocation")
        != arm_contract["split_scratch_allocation"]
    ):
        raise ValueError(f"qrow32 {arm} live provenance drifted")
    live_source_commit = _commit(payload.get("source_commit"), "live source commit")
    live_patch_sha = _sha256(
        payload.get("patch_source_sha256"), "live patch source"
    )
    if source_commit is not None and live_source_commit != source_commit:
        raise ValueError("qrow32 live source commit drifted")
    if patch_source_sha256 is not None and live_patch_sha != patch_source_sha256:
        raise ValueError("qrow32 live patch source drifted")
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != 16:
        raise ValueError(f"qrow32 {arm} live layer set drifted")
    output_digests = []
    lse_digests = []
    names = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_name"), str):
            raise ValueError(f"qrow32 {arm} live layer record drifted")
        names.add(layer["layer_name"])
        output_digests.append(
            _validate_comparison(layer.get("output"), f"layer {index} output")
        )
        lse_digests.append(
            _validate_comparison(layer.get("lse"), f"layer {index} lse")
        )
    expected_names = {
        f"language_model.model.layers.{index}.self_attn.attn"
        for index in range(3, 64, 4)
    }
    if names != expected_names:
        raise ValueError(f"qrow32 {arm} live layer identities drifted")
    if (
        payload.get("output_raw_byte_mismatches") != 0
        or payload.get("lse_raw_byte_mismatches") != 0
    ):
        raise ValueError(f"qrow32 {arm} live aggregate mismatch drifted")
    return {
        "arm": arm,
        "instance_id": payload["instance_id"],
        "source_commit": live_source_commit,
        "patch_source_sha256": live_patch_sha,
        "layers_sha256": _digest(
            canonical_bytes(
                {
                    "names": sorted(names),
                    "output": output_digests,
                    "lse": lse_digests,
                }
            )
        ),
    }


def validate_gqa_pair_gate(
    payload: dict[str, Any], *, source_commit: str
) -> dict[str, Any]:
    """Validate the sealed B1 GQA-pair byte gate artifact.

    This checks the gate's own account of itself. The raw-byte equality is NOT
    asserted here -- the gate artifact does not carry the mismatch counts, it
    carries the digest of the live A/B result that does, and
    ``validate_gqa_pair_binding`` follows that digest and re-validates the
    evidence directly.
    """
    if payload.get("schema") != _gate_schema_for(payload):
        raise ValueError("GQA-pair gate schema drifted")
    if payload.get("status") != "PASS":
        raise ValueError("GQA-pair gate result is not a PASS")
    contract = _candidate_contract(GQA_PAIR_ARM)
    if (
        payload.get("candidate_so_sha256") != contract["sha256"]
        or payload.get("candidate_so_size") != contract["size"]
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256")
        != contract["source_closure_sha256"]
    ):
        raise ValueError("GQA-pair gate candidate identity drifted")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("topology") != GQA_PAIR_GATE_TOPOLOGY
        or payload.get("task_ids") != list(GQA_PAIR_GATE_TASK_IDS)
        or payload.get("subset_sha256") != GQA_PAIR_GATE_SUBSET_SHA256
        or payload.get("batch_size") != 1
        or payload.get("concurrency") != 1
        or payload.get("physical_rows") != 32
        or not _draft_vocab_identity_is_consistent(payload)
    ):
        raise ValueError("GQA-pair gate operating point drifted")
    # The byte gate served the incumbent and measured nothing. Those are
    # properties of THAT run; a gate artifact claiming otherwise has been
    # edited, and it is exactly those two claims a timing verdict would want to
    # borrow illegitimately.
    if (
        payload.get("performance_measurement") is not False
        or payload.get("served_return") != GQA_PAIR_SERVED_RETURN
    ):
        raise ValueError("GQA-pair gate eligibility fields were rewritten")
    gate_commit = _commit(payload.get("source_commit"), "GQA-pair gate source commit")
    if gate_commit != source_commit:
        raise ValueError(
            "GQA-pair gate was not produced at the production plumbing commit"
        )
    summary = {"source_commit": gate_commit}
    for key in (
        "live_result_sha256",
        "layers_sha256",
        "health_sha256",
        "traffic_audit_sha256",
        "block_map_sha256",
        "diagnostic_binding_sha256",
        "patch_source_sha256",
    ):
        summary[key] = _sha256(payload.get(key), f"GQA-pair gate {key}")
    return summary


def validate_gqa_pair_binding(
    *,
    gate: Path,
    expected_gate_sha256: str,
    live_result: Path,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """The GQA-pair credential body, without writing it: a runner pre-flight."""
    if arm != GQA_PAIR_ARM:
        raise ValueError("qrow32 B1 GQA-pair credential arm must be gqa_pair")
    expected_gate_sha256 = _sha256(expected_gate_sha256, "GQA-pair gate")
    validate_candidate(candidate_so, expected_candidate_sha256, arm=GQA_PAIR_ARM)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    gate_payload, gate_raw = load_json(gate)
    if _digest(gate_raw) != expected_gate_sha256:
        raise ValueError("GQA-pair gate raw SHA-256 mismatch")
    gate_summary = validate_gqa_pair_gate(
        gate_payload, source_commit=patch["source_commit"]
    )
    if gate_summary["patch_source_sha256"] != patch["patch_source_sha256"]:
        raise ValueError("GQA-pair gate patch source drifted")
    # Follow the gate's own digest to the raw-byte evidence and re-validate it.
    live_payload, live_raw = load_json(live_result)
    if _digest(live_raw) != gate_summary["live_result_sha256"]:
        raise ValueError("GQA-pair live result is not the one the gate bound")
    live_summary = validate_live_result(
        live_payload,
        candidate_sha256=expected_candidate_sha256,
        arm=GQA_PAIR_ARM,
        source_commit=patch["source_commit"],
        patch_source_sha256=patch["patch_source_sha256"],
    )
    if live_summary["layers_sha256"] != gate_summary["layers_sha256"]:
        raise ValueError("GQA-pair gate layer digest does not bind its evidence")
    contract = _candidate_contract(GQA_PAIR_ARM)
    arm_contract = LIVE_ARMS[GQA_PAIR_ARM]
    body = {
        "schema": GQA_PAIR_SIDECAR_SCHEMA,
        "status": "PASS",
        "arm": GQA_PAIR_ARM,
        "selector_sentinel": arm_contract["selector_sentinel"],
        "num_splits": arm_contract["num_splits"],
        "reference": "qrow16 incumbent exact geometry",
        "reference_selector_sentinel": QROW16_REFERENCE_SENTINEL,
        "candidate_so_size": contract["size"],
        "candidate_so_sha256": contract["sha256"],
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": contract["source_closure_sha256"],
        "source_commit": patch["source_commit"],
        "patch_source_sha256": patch["patch_source_sha256"],
        # The sidecar carries the profile so verify-time can resolve the
        # same schema name the gate used. Without it, verification would
        # default to k64_root and reject a full_vocab credential.
        "qualification_profile": gate_payload.get(
            "qualification_profile", "k64_root"
        ),
        "gate_schema": _gate_schema_for(gate_payload),
        "gate_sha256": expected_gate_sha256,
        "gate_canonical_sha256": _digest(canonical_bytes(gate_payload)),
        "gate_topology": GQA_PAIR_GATE_TOPOLOGY,
        "gate_task_ids": list(GQA_PAIR_GATE_TASK_IDS),
        "gate_subset_sha256": GQA_PAIR_GATE_SUBSET_SHA256,
        "gate_performance_measurement": False,
        "gate_served_return": GQA_PAIR_SERVED_RETURN,
        "gate_health_sha256": gate_summary["health_sha256"],
        "gate_traffic_audit_sha256": gate_summary["traffic_audit_sha256"],
        "gate_block_map_sha256": gate_summary["block_map_sha256"],
        "gate_diagnostic_binding_sha256": gate_summary[
            "diagnostic_binding_sha256"
        ],
        "live_gate_schema": LIVE_SCHEMA,
        "live_result_sha256": gate_summary["live_result_sha256"],
        "live_result_canonical_sha256": _digest(canonical_bytes(live_payload)),
        "instance_id": live_summary["instance_id"],
        "layers_sha256": live_summary["layers_sha256"],
        "layer_count": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "credential_basis": GQA_PAIR_CREDENTIAL_BASIS,
        "required_runtime": _required_runtime_for(gate_payload),
        "production_scope": GQA_PAIR_PRODUCTION_SCOPE,
        # Production serves exact4; the gate qualified one task. The selector's
        # exact-geometry hard-raise is what makes the wider scope safe, not any
        # claim the gate made.
        "production_task_ids": list(EXACT4_TASK_IDS),
        "production_subset_sha256": EXACT4_SUBSET_SHA256,
        "gate_scope_narrower_than_production": True,
        "fallback_allowed": False,
    }
    return body


def issue_gqa_pair_sidecar(
    *,
    gate: Path,
    expected_gate_sha256: str,
    live_result: Path,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    out: Path,
) -> dict[str, Any]:
    body = validate_gqa_pair_binding(
        gate=gate,
        expected_gate_sha256=expected_gate_sha256,
        live_result=live_result,
        candidate_so=candidate_so,
        expected_candidate_sha256=expected_candidate_sha256,
        arm=arm,
        patch_source=patch_source,
        expected_source_commit=expected_source_commit,
    )
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace qrow32 B1 GQA-pair sidecar: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_bytes(sidecar) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return sidecar


def verify_gqa_pair_sidecar(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    expected_patch_source_sha256: str,
) -> dict[str, Any]:
    # Verification runs INSIDE the pinned runtime image, which has no git, so
    # the patch source is proven by digest rather than by repository state.
    if arm != GQA_PAIR_ARM:
        raise ValueError("qrow32 B1 GQA-pair credential arm must be gqa_pair")
    expected_sidecar_sha256 = _sha256(expected_sidecar_sha256, "pass sidecar")
    validate_candidate(candidate_so, expected_candidate_sha256, arm=GQA_PAIR_ARM)
    patch = validate_patch_source_digest(
        patch_source,
        expected_source_commit=expected_source_commit,
        expected_patch_source_sha256=expected_patch_source_sha256,
    )
    payload, raw = load_json(sidecar_path)
    if _digest(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise ValueError("pass sidecar canonical digest mismatch")
    contract = _candidate_contract(GQA_PAIR_ARM)
    arm_contract = LIVE_ARMS[GQA_PAIR_ARM]
    if (
        payload.get("schema") != GQA_PAIR_SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("arm") != GQA_PAIR_ARM
        or payload.get("selector_sentinel") != arm_contract["selector_sentinel"]
        or payload.get("num_splits") != arm_contract["num_splits"]
        or payload.get("reference") != "qrow16 incumbent exact geometry"
        or payload.get("reference_selector_sentinel") != QROW16_REFERENCE_SENTINEL
        or payload.get("candidate_so_size") != contract["size"]
        or payload.get("candidate_so_sha256") != contract["sha256"]
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256")
        != contract["source_closure_sha256"]
        or payload.get("source_commit") != patch["source_commit"]
        or payload.get("patch_source_sha256") != patch["patch_source_sha256"]
        or payload.get("gate_schema") != _gate_schema_for(payload)
        or payload.get("gate_topology") != GQA_PAIR_GATE_TOPOLOGY
        or payload.get("gate_task_ids") != list(GQA_PAIR_GATE_TASK_IDS)
        or payload.get("gate_subset_sha256") != GQA_PAIR_GATE_SUBSET_SHA256
        or payload.get("gate_performance_measurement") is not False
        or payload.get("gate_served_return") != GQA_PAIR_SERVED_RETURN
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("instance_id") != GQA_PAIR_GATE_TASK_IDS[0]
        or payload.get("layer_count") != 16
        or payload.get("output_raw_byte_mismatches") != 0
        or payload.get("lse_raw_byte_mismatches") != 0
        or payload.get("credential_basis") != GQA_PAIR_CREDENTIAL_BASIS
        or payload.get("required_runtime") != _required_runtime_for(payload)
        or payload.get("production_scope") != GQA_PAIR_PRODUCTION_SCOPE
        or payload.get("production_task_ids") != list(EXACT4_TASK_IDS)
        or payload.get("production_subset_sha256") != EXACT4_SUBSET_SHA256
        or payload.get("gate_scope_narrower_than_production") is not True
        or payload.get("fallback_allowed") is not False
    ):
        raise ValueError("pass sidecar contract drifted")
    for key in (
        "gate_sha256",
        "gate_canonical_sha256",
        "gate_health_sha256",
        "gate_traffic_audit_sha256",
        "gate_block_map_sha256",
        "gate_diagnostic_binding_sha256",
        "live_result_sha256",
        "live_result_canonical_sha256",
        "layers_sha256",
    ):
        _sha256(payload.get(key), key)
    payload["canonical_sha256"] = canonical
    return payload


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    out: Path,
) -> dict[str, Any]:
    if arm != ARM:
        raise ValueError("qrow32 production arm must be nosplit")
    expected_live_sha256 = _sha256(expected_live_sha256, "live result")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    live, raw = load_json(live_result)
    if _digest(raw) != expected_live_sha256:
        raise ValueError("live result raw SHA-256 mismatch")
    summary = validate_live_result(
        live,
        candidate_sha256=expected_candidate_sha256,
        arm=arm,
        source_commit=patch["source_commit"],
        patch_source_sha256=patch["patch_source_sha256"],
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "arm": ARM,
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "reference": "qrow16 incumbent exact geometry",
        "reference_selector_sentinel": QROW16_REFERENCE_SENTINEL,
        "candidate_so_size": CANDIDATE_SIZE,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "source_commit": patch["source_commit"],
        "patch_source_sha256": patch["patch_source_sha256"],
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest(canonical_bytes(live)),
        "live_gate_schema": LIVE_SCHEMA,
        "instance_id": summary["instance_id"],
        "layers_sha256": summary["layers_sha256"],
        "required_runtime": "Hydra27 fixed32 K64 ROOT=1 B1 physical32 FULL graph",
        "production_scope": "qrow32 B1 nosplit exact tree attention only",
        "fallback_allowed": False,
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace qrow32 nosplit pass sidecar: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_bytes(sidecar) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return sidecar


def verify_sidecar(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    expected_patch_source_sha256: str,
) -> dict[str, Any]:
    # Verification runs INSIDE the pinned runtime image, which has no git, so
    # the patch source is proven by digest rather than by repository state.
    expected_sidecar_sha256 = _sha256(expected_sidecar_sha256, "pass sidecar")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source_digest(
        patch_source,
        expected_source_commit=expected_source_commit,
        expected_patch_source_sha256=expected_patch_source_sha256,
    )
    payload, raw = load_json(sidecar_path)
    if _digest(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise ValueError("pass sidecar canonical digest mismatch")
    if (
        arm != ARM
        or payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("arm") != ARM
        or payload.get("selector_sentinel") != SELECTOR_SENTINEL
        or payload.get("num_splits") != NUM_SPLITS
        or payload.get("reference") != "qrow16 incumbent exact geometry"
        or payload.get("reference_selector_sentinel") != QROW16_REFERENCE_SENTINEL
        or payload.get("candidate_so_size") != CANDIDATE_SIZE
        or payload.get("candidate_so_sha256") != CANDIDATE_SHA256
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256") != SOURCE_CLOSURE_SHA256
        or payload.get("source_commit") != patch["source_commit"]
        or payload.get("patch_source_sha256") != patch["patch_source_sha256"]
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("required_runtime")
        != "Hydra27 fixed32 K64 ROOT=1 B1 physical32 FULL graph"
        or payload.get("production_scope")
        != "qrow32 B1 nosplit exact tree attention only"
        or payload.get("fallback_allowed") is not False
    ):
        raise ValueError("pass sidecar contract drifted")
    for key in (
        "live_result_sha256",
        "live_result_canonical_sha256",
        "layers_sha256",
    ):
        _sha256(payload.get(key), key)
    payload["canonical_sha256"] = canonical
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("validate-source")
    source.add_argument("--source-root", required=True, type=Path)
    source.add_argument(
        "--arm",
        choices=(ARM, "split2", VISIBILITY_ARM, GQA_PAIR_ARM, SPLITK_ARM),
        default=ARM,
    )
    for name in ("issue", "verify"):
        command = subparsers.add_parser(name)
        if name == "issue":
            command.add_argument("--live-result", required=True, type=Path)
            command.add_argument("--expected-live-sha256", required=True)
            command.add_argument("--out", required=True, type=Path)
        else:
            command.add_argument("--sidecar", required=True, type=Path)
            command.add_argument("--expected-sidecar-sha256", required=True)
            # Verification runs in the git-less runtime image; the launcher
            # supplies the digest it validated against the host tree.
            command.add_argument(
                "--expected-patch-source-sha256", required=True
            )
        command.add_argument("--candidate-so", required=True, type=Path)
        command.add_argument("--expected-candidate-sha256", required=True)
        command.add_argument("--arm", required=True, choices=(ARM,))
        command.add_argument("--patch-source", required=True, type=Path)
        command.add_argument("--expected-source-commit", required=True)
    # The GQA-pair arm keeps its own subcommands rather than overloading the
    # no-split ones: its evidence is a different artifact pair (sealed gate +
    # the live A/B result the gate binds), and the incumbent credential path
    # stays byte-for-byte unchanged.
    for name in ("validate-gqa-pair", "issue-gqa-pair", "verify-gqa-pair"):
        command = subparsers.add_parser(name)
        if name == "verify-gqa-pair":
            command.add_argument("--sidecar", required=True, type=Path)
            command.add_argument("--expected-sidecar-sha256", required=True)
            # Verification runs in the git-less runtime image; the launcher
            # supplies the digest it validated against the host tree.
            command.add_argument(
                "--expected-patch-source-sha256", required=True
            )
        else:
            # By PATH, never by runroot: the credential must be issued against
            # a gate re-run at the production plumbing commit.
            command.add_argument("--gate", required=True, type=Path)
            command.add_argument("--expected-gate-sha256", required=True)
            command.add_argument("--live-result", required=True, type=Path)
            if name == "issue-gqa-pair":
                command.add_argument("--out", required=True, type=Path)
        command.add_argument("--candidate-so", required=True, type=Path)
        command.add_argument("--expected-candidate-sha256", required=True)
        command.add_argument("--arm", required=True, choices=(GQA_PAIR_ARM,))
        command.add_argument("--patch-source", required=True, type=Path)
        command.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        result = validate_source_closure(args.source_root, arm=args.arm)
    elif args.command in ("validate-gqa-pair", "issue-gqa-pair"):
        binding = dict(
            gate=args.gate,
            expected_gate_sha256=args.expected_gate_sha256,
            live_result=args.live_result,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
        )
        if args.command == "validate-gqa-pair":
            result = validate_gqa_pair_binding(**binding)
        else:
            result = issue_gqa_pair_sidecar(out=args.out, **binding)
    elif args.command == "verify-gqa-pair":
        result = verify_gqa_pair_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
            expected_patch_source_sha256=args.expected_patch_source_sha256,
        )
    elif args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
            out=args.out,
        )
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
            expected_patch_source_sha256=args.expected_patch_source_sha256,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
