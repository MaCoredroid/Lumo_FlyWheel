#!/usr/bin/env python3
"""Plot a reviewed E1 aggregate without filling missing cells or inventing data.

Consumes the aggregate's certified eligible rates and already-computed paired
intervals. It does not select attempts, compute new confidence intervals, or
change the frozen analysis. Writes PDF/PNG and a source-hash sidecar.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("native-5", "native-11", "tree")
LABELS = ("Native-5", "Native-11", "Cat10")
CONTRASTS = ("tree_minus_native5", "tree_minus_native11", "native11_minus_native5")
DELTA_LABELS = ("Cat10 − Native-5", "Cat10 − Native-11", "Native-11 − Native-5")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregate", type=Path)
    ap.add_argument("--output", type=Path, required=True, help="Output stem")
    a = ap.parse_args()
    data = json.loads(a.aggregate.read_text())
    assert len(data["cells"]) == 18, "Preserve the complete planned cell table"
    assert {r["index"] for r in data["cells"]} == set(range(1, 19))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.6),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    colors = ("#777777", "#999999", "#bbbbbb")
    plotted = {}
    for column, batch in enumerate(("B1", "B4")):
        result = data["by_batch"][batch]
        values = result["per_arm_block_rates"]
        assert set(values) == set(ARMS)
        assert all(len(values[arm]) == 3 for arm in ARMS)
        top, bottom = axes[:, column]
        for block in range(3):
            ys = [values[arm][block] for arm in ARMS]
            for arm, value in zip(ARMS, ys):
                cell = next(r for r in data["cells"]
                            if r["batch"] == batch and r["arm"] == arm
                            and r["block"] == block + 1)
                if value is not None:
                    assert cell["status"] == "VALID" and finite(value) and value > 0
                    assert value == cell["rate"]
            # Missing cells break lines; they are never drawn as zero rates.
            top.plot(range(3), [v if v is not None else math.nan for v in ys],
                     "o-", color=colors[block], linewidth=.7, markersize=3,
                     label=f"Block {block + 1}")
        for x, arm in enumerate(ARMS):
            mean = result["per_arm_mean_over_blocks"][arm]
            if mean is not None:
                assert finite(mean) and all(v is not None for v in values[arm])
                top.plot(x, mean, "D", color="#1d5f91", markersize=4)
        top.set(title=batch, xticks=range(3), xticklabels=LABELS,
                ylabel="API-bound tokens / retained wall s")
        top.set_ylim(bottom=0)
        top.grid(axis="y", linewidth=.4, alpha=.3)
        bottom.axvline(0, color="#777777", linestyle="--", linewidth=.7)
        for y, key in enumerate(CONTRASTS):
            contrast = result["contrasts"][key]
            if "bootstrap_95pct_interval" not in contrast:
                bottom.text(.02, y, "Not estimable", transform=bottom.get_yaxis_transform(),
                            va="center", fontsize=7, color="#777777")
                continue
            mean = contrast["mean_difference"]
            lo, hi = contrast["bootstrap_95pct_interval"]
            assert all(finite(v) for v in (mean, lo, hi)) and lo <= hi
            bottom.hlines(y, lo, hi, color="#1d5f91", linewidth=1.2)
            bottom.plot(mean, y, "D", color="#1d5f91", markersize=4)
        bottom.set(yticks=range(3), yticklabels=DELTA_LABELS,
                   xlabel="Paired rate difference (tokens / wall s)")
        bottom.set_ylim(2.55, -.55)
        plotted[batch] = result
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .97), h_pad=1.7, w_pad=1.5)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    proof = {"source": str(a.aggregate.resolve()),
             "source_sha256": hashlib.sha256(a.aggregate.read_bytes()).hexdigest(),
             "scope": "Top: three paired-boot rates and complete-arm means. Bottom: supplied paired-block 95% bootstrap percentile intervals; three-block uncertainty is coarse. Missing support is not zero.",
             "plotted": plotted}
    a.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
