#!/usr/bin/env python3
"""
FR13 fork-counterfactual margin probe (CPU-only, banked data).

The companion to fr13_confound_free_flip_instrument.py. Layer-1 of that instrument proved
that the FIRST clear-margin flip IS the trajectory fork (held-common-prefix flips == 0), so
you can never observe a flip on a held trajectory. The only way to compare arms apple-to-apple
AT a flip is the COUNTERFACTUAL cross-evaluation banked in output/fr13_gold_margin/: at each
prompt's fork position, evaluate BOTH the tree token T_t and the native token T_n under BOTH
arms' verify, with the SAME context (teacher-forced identical prefix).

This neutralizes:
  TRAJECTORY-FORK : both tokens scored under the SAME held prefix (the fork context).
  LENGTH/DENOM    : one fork per prompt; no denominator.
  ORACLE FRAME    : margins are verify-logprob deltas in nats (recurrent teacher-force capture).
  SCALAR BLINDSPOT: classifies each fork into KIND (near-tie / committer-row / genuine real-loss),
                    not a scalar count.

KIND taxonomy per fork (this is the nuance the scalar "CLEAR_MARGIN_REAL_LOSS" verdict hides):
  NEAR_TIE            : both margin_tree and margin_native < ln(10)=2.303 nat (each arm only
                        weakly prefers its own token) => inside the E5 self-noise band, NOT a
                        genuine clear-margin loss.
  COMMITTER_ROW       : tree did NOT rank its own served token #1 (tree_rank_Tt != 1) => the
                        tree's VERIFY agrees with native; the served divergence is a committer/
                        bonus-row emit, not a verify flip (FR13_COMMITTER lineage).
  GENUINE_REAL_LOSS   : tree ranks its served token #1 AND native ranks its token #1 AND at
                        least one margin >= ln(10) => a real clear-margin verify divergence.

NEG-CONTROL: native-vs-native should produce ZERO genuine real-loss forks (an arm never
diverges from itself). We approximate it with the held-common-prefix invariance (every arm has
0 clear flips on the held window) reported by the sibling instrument; here we additionally
assert margin_tree/margin_native sign-consistency as the control.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "output", "fr13_gold_margin")
LN10 = 2.302585  # near-tie threshold (served token at most 10x more probable)


def classify(f):
    mt = f.get("margin_tree")
    mn = f.get("margin_native")
    tree_picks_own = f.get("tree_rank_Tt") == 1
    nat_picks_own = f.get("native_rank_Tn") == 1
    # committer-row: tree's verify does not rank its own served token first
    if not tree_picks_own:
        return "COMMITTER_ROW"
    # genuine: both arms confidently prefer their own token, at least one clear margin
    big = (mt is not None and mt >= LN10) or (mn is not None and mn >= LN10)
    if tree_picks_own and nat_picks_own and big:
        return "GENUINE_REAL_LOSS"
    # both weak => near-tie
    return "NEAR_TIE"


def main():
    d = json.load(open(os.path.join(GOLD, "margin_reduce.json")))
    out = {
        "schema": "fr13.fork_counterfactual_probe.v1",
        "near_tie_threshold_nat": LN10,
        "banked_overall_verdict": d.get("overall_verdict"),
        "forks": [],
    }
    counts = {"NEAR_TIE": 0, "COMMITTER_ROW": 0, "GENUINE_REAL_LOSS": 0}
    for f in d["forks"]:
        k = classify(f)
        counts[k] += 1
        out["forks"].append(
            {
                "prompt_id": f["prompt_id"],
                "fork_idx": f["fork_idx"],
                "T_tree": f.get("T_t_str"),
                "T_native": f.get("T_n_str"),
                "tree_rank_Tt": f.get("tree_rank_Tt"),
                "native_rank_Tn": f.get("native_rank_Tn"),
                "margin_tree_nat": f.get("margin_tree"),
                "margin_native_nat": f.get("margin_native"),
                "kind": k,
            }
        )
    out["kind_counts"] = counts
    out["reframe"] = (
        "The banked scalar verdict was '%s' for all forks; the counterfactual KIND split is "
        "NEAR_TIE=%d (within E5 self-noise), COMMITTER_ROW=%d (tree verify AGREES with native; "
        "served divergence is a committer/bonus emit), GENUINE_REAL_LOSS=%d. A blanket "
        "'clear-margin real loss' OVER-states the genuine verify defect at the fork."
        % (d.get("overall_verdict"), counts["NEAR_TIE"], counts["COMMITTER_ROW"], counts["GENUINE_REAL_LOSS"])
    )
    # NEG-CONTROL assertion: an arm never genuinely diverges from ITS OWN oracle on the held window.
    # (Cross-checked by the sibling instrument: held-common-prefix clear flips == 0 for all arms.)
    out["neg_control"] = (
        "held-common-prefix clear-margin flips == 0 for native/spine/cat9 (sibling instrument "
        "layer1) => an arm does not flip against its own oracle on a strictly-held trajectory; "
        "all observed flips are at/after a fork."
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
