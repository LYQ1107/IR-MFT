#!/usr/bin/env python3
"""Export a small scientific overview from the audited short-budget JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    lookup = {(row["arm"],row["split"]):row for row in result["evaluations"]}
    arms = ["rgb_original", "rgb_matched", "old_legacy", "new_diffv2ir"]
    labels = ["Original\nRGB", "Fine-tuned\nRGB", "Old IR\nV3", "New IR\nV3"]
    colors = ["#777777", "#298652", "#3572b0", "#dd8534"]
    if any((arm,split) not in lookup for arm in arms for split in ["dev","test"]):
        raise ValueError("All eight complete main evaluations are required")
    fig, axes = plt.subplots(2,2,figsize=(11,7.3),constrained_layout=True)
    for ax, split in zip(axes[0],["dev","test"]):
        values = [lookup[(arm,split)]["metrics"]["HOTA"] for arm in arms]
        ax.bar(labels, values, color=colors, width=.65)
        for x, value in enumerate(values):
            ax.text(x, value+.7, f"{value:.3f}",ha="center",fontsize=9)
        ax.set(title=f"{split.capitalize()} HOTA",ylabel="HOTA (%)",ylim=(0,55))
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    ax = axes[1,0]
    sequences = result["protocol"]["test_sequences"]
    deltas = [result["per_sequence_deltas"]["test_new_minus_old_legacy"][seq]["HOTA"] for seq in sequences]
    positions = np.arange(len(sequences))
    ax.barh(positions,deltas,color=["#298652" if value>=0 else "#bb4949" for value in deltas])
    ax.set_yticks(positions,sequences)
    ax.invert_yaxis()
    ax.axvline(0,color="black",lw=.8)
    ax.set(title="Test: new IR minus old IR",xlabel="HOTA change (percentage points)",xlim=(-12,13))
    for y,value in enumerate(deltas):
        ax.text(value+(.25 if value>=0 else -.25),y,f"{value:+.2f}",va="center",
                ha="left" if value>=0 else "right",fontsize=8)
    ax.grid(axis="x",alpha=.2)
    ax.set_axisbelow(True)
    ax = axes[1,1]
    for arm,label,color in zip(arms[1:],labels[1:],colors[1:]):
        rows = [row for row in result["training"][arm]["history"] if row["phase"] == "V3_finetune"]
        ax.plot(range(1,len(rows)+1),[row["loss"] for row in rows],marker="o",ms=4,
                color=color,label=label.replace("\n"," "))
    ax.set(title="Short-budget joint training loss",xlabel="Completed joint epoch",ylabel="Mean training loss",xticks=[1,2,3])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle("Pseudo-IR comparison: three joint epochs, one seed\nPaired arms also have three auxiliary calibration epochs",fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,
                         source_status=result["status"],scope="Eight main evaluations; historical reference shown separately in tables")))


if __name__ == "__main__":
    main()
