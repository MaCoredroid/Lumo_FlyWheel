import json
import os
import sys
from pathlib import Path

AGENT_WS = Path(os.environ["AGENT_WS"]).resolve()
VERIFIER_DATA = Path(os.environ["VERIFIER_DATA"]).resolve()
VARIANT_ID = os.environ["VARIANT_ID"]
sys.path.insert(0, str(AGENT_WS))

from deploy.check_release import evaluate_release_alignment  # noqa: E402


def load_gold() -> dict:
    return json.loads((VERIFIER_DATA / VARIANT_ID / "gold_release.json").read_text())


def test_hidden_release_alignment_pack():
    gold = load_gold()
    result = evaluate_release_alignment(AGENT_WS, "staging")
    assert result["ok"], result
    assert result["artifact_manifest"] == gold["proof_contract"]["artifact_manifest"]
    assert result["ordered_checks"] == gold["proof_contract"]["ordered_checks"]


def test_docs_requirements():
    gold = load_gold()
    docs = (AGENT_WS / "docs/releases/staging_rollout.md").read_text().lower()
    for phrase in gold["required_docs_phrases"]:
        assert phrase in docs, phrase
    for earlier, later in gold["docs_order_pairs"]:
        assert docs.index(earlier) < docs.index(later)
