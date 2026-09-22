#!/usr/bin/env python3
"""DERIVED API token-id sidecar for the B4 pilot (2026-09-22). The cohort client v1 omitted return_tokens_as_token_ids, so
the four capture_request.json files carry decoded token STRINGS (preserved untouched). The independent CPU review
p0/monitor/e2-b4-api-inversion-redteam.md (exact pinned tokenizers 0.22.2 + the exact vLLM logprob formatter + the seven
tokenizer_config special ids; all 248320 model ids exhausted at every position) recovered a UNIQUE id for each of the 128
observed strings, keyed by the API response identity without consulting engine ids. This script embeds those certified
vectors, re-verifies them against (a) the raw strings' unique inverse under the checkpoint tokenizer (mapper v2, singleton
decode) and (b) the recorder ledger prefix per mapped request, and writes <run>/e1_api_tokens.derived.json labeled DERIVED
with the report/proof hashes. It never modifies capture_request.json. Exit 2 on any disagreement.
Usage: e1_api_tokens_derived_sidecar.py <run_dir> [--report-path p0/monitor/e2-b4-api-inversion-redteam.md]"""
import argparse, glob, hashlib, json, os, sys
CERTIFIED = {"report": "p0/monitor/e2-b4-api-inversion-redteam.md", "base_map_sha256": "0423a31175fa2895310e53dd325b3b0b0b74f558a41db8f1bcb045d1ccd83a40", "proof_records_sha256": "b4b9cad2bb0c36a6a8f634fdcab8980adeff0ce95aeb4ba0c8763051c1d7b08a",
             "capture_sha256": {"req_0_p072": "ebe977a2189c4d3c6a6b0eca59598c76f7b77ddcca8a132546372e81a4ee5980", "req_1_p017": "38f65eaf163434a5b0b0420cfb88d923275e449ddc5df1c0c4b174c30d2b4afe", "req_2_p085": "a970cd4a6612eb466700697f644b1db997647f4ac6b901525ad1080d7ca9e57e", "req_3_p095": "24f741a9a232290868b48155db928e64c37d08df2ae8f22b2b67ac546a13d8cf"},
             "ids": {"cmpl-a33355be5168a83b": [6813,12333,36349,853,6971,6813,12333,36349,648,6971,6813,12333,36349,788,6971,6813,12333,36349,2135,6971,6813,12333,36349,736,6971,6813,12333,36349,1824,6971,6813,12333],
                     "cmpl-be7c5fc6026dc3ca": [264,2972,14892,22258,332,21853,421,11693,279,220,16,21,13,19,20,85123,198,429,279,220,17,24,13,16,16,15686,13,561,491,795,4774,279],
                     "cmpl-822fbbe30317f121": [62987,5303,506,62987,854,13,561,8057,2099,369,23185,1518,279,8057,8213,26,279,62987,33877,513,23185,1518,62987,8213,13,1061,27028,2107,3274,32524,318,7838],
                     "cmpl-bba5676034b3072d": [8288,2686,1892,424,369,264,2972,588,61384,332,220,19,18295,23014,318,1719,220,21,19,45668,1994,23985,220,16,19,19,20112,33,8,421,8311,310]}}
def fsha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--report-path", default="/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/p0/monitor/e2-b4-api-inversion-redteam.md"); a = ap.parse_args()
    out = {"label": "DERIVED — API token ids recovered from the preserved decoded strings by the certified exact inversion (not requested ids; every future client requests token ids directly)", "certification": dict(CERTIFIED, report_sha256=(fsha(a.report_path) if os.path.exists(a.report_path) else None)), "requests": {}, "disagreements": []}
    v2 = json.load(open(os.path.join(a.run, "e1_api_tokens.v2.json"))) if os.path.exists(os.path.join(a.run, "e1_api_tokens.v2.json")) else {}
    ev = [json.loads(l) for l in open(os.path.join(a.run, "logs", "e1_events.jsonl")) if l.strip()]
    for f in sorted(glob.glob(os.path.join(a.run, "cohort", "req_*", "capture_request.json"))):
        name = os.path.basename(os.path.dirname(f)); j = json.load(open(f)); rid = j["response_id"]; strings = j.get("response_logprobs_tokens") or []
        ids = CERTIFIED["ids"].get(rid); rec = {"response_id": rid, "capture_sha256": fsha(f), "raw_strings_preserved": strings, "n_strings": len(strings), "certified_ids": ids}
        if fsha(f) != CERTIFIED["capture_sha256"].get(name): out["disagreements"].append(f"{name}: capture_request.json sha differs from the certified input")
        if ids is None or len(ids) != len(strings): out["disagreements"].append(f"{name}: no certified vector / length mismatch")
        m = [r for r in v2 if r == rid or r.startswith(rid + "-")]
        if len(m) == 1:
            rec["recorder_id"] = m[0]; rec["mapper_v2_singleton_recovery_equal"] = (v2[m[0]] == ids)
            led = [int(x) for e in ev if e.get("event") == "output_rows" for r in e["rows"] if r["request_id"] == m[0] for x in r["emitted_ids"]]
            rec["ledger_prefix_equal"] = (led[:len(ids)] == ids) if ids else None; rec["ledger_tokens_total"] = len(led)
            if not rec["mapper_v2_singleton_recovery_equal"] or not rec["ledger_prefix_equal"]: out["disagreements"].append(f"{name}: certified ids disagree with the singleton recovery or the ledger prefix")
        else: out["disagreements"].append(f"{name}: recorder id mapping not unique ({m})")
        out["requests"][name] = rec
    out["derived_map_recorder_keyed"] = {out["requests"][n]["recorder_id"]: out["requests"][n]["certified_ids"] for n in out["requests"] if "recorder_id" in out["requests"][n]}
    json.dump(out, open(os.path.join(a.run, "e1_api_tokens.derived.json"), "w"), indent=1, ensure_ascii=False)
    print("derived sidecar:", {n: (r.get("mapper_v2_singleton_recovery_equal"), r.get("ledger_prefix_equal")) for n, r in out["requests"].items()}, "disagreements:", out["disagreements"]); sys.exit(2 if out["disagreements"] else 0)
if __name__ == "__main__":
    main()
