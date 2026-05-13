import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_packet_outputs_exist_and_name_required_sections():
    packet = ROOT / "packet/incident_packet.md"
    findings = ROOT / "packet/findings.json"

    assert packet.is_file(), "write packet/incident_packet.md"
    assert findings.is_file(), "write packet/findings.json"
    text = packet.read_text(encoding="utf-8").lower()
    for section in ("trigger", "guardrail", "follow-up", "ambiguity"):
        assert section in text


def test_findings_shape_is_ranked_and_evidence_backed():
    findings = json.loads((ROOT / "packet/findings.json").read_text(encoding="utf-8"))

    assert isinstance(findings.get("ranked_findings"), list)
    assert findings["ranked_findings"], "ranked_findings must not be empty"
    top = findings["ranked_findings"][0]
    assert top.get("confidence") in {"high", "medium", "low"}
    assert top.get("evidence"), "top finding needs evidence references"
    assert findings.get("unresolved_ambiguity")
