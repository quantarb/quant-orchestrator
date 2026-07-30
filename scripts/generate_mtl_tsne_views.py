"""Generate reproducible 3D t-SNE views from a trained historical MTL run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _save_view(frame: pd.DataFrame, output: Path, title: str, *, target_labels: set[str] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15, 15), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    target_labels = target_labels or set()
    for task, color, marker in (
        ("feature_family", "tab:blue", "o"),
        ("target_family", "black", "*"),
        ("family", "tab:blue", "o"),
        ("sector", "tab:orange", "s"),
        ("subsector", "tab:green", "^"),
        ("industry", "tab:red", "D"),
        ("year", "tab:purple", "P"),
        ("cross_sectional_year", "tab:cyan", "X"),
    ):
        part = frame.loc[frame["task"].eq(task)].copy()
        if part.empty:
            continue
        if task == "family" and target_labels:
            normal = part.loc[~part["label"].isin(target_labels)]
            target = part.loc[part["label"].isin(target_labels)]
            axis.scatter(normal.x, normal.y, normal.z, s=38, color=color, marker=marker, alpha=0.8)
            axis.scatter(target.x, target.y, target.z, s=110, color="black", marker="*", alpha=1.0)
        else:
            axis.scatter(part.x, part.y, part.z, s=28, color=color, marker=marker, alpha=0.85)
        for _, row in part.iterrows():
            axis.text(row.x, row.y, row.z, str(row.label), fontsize=4.0, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.set_zlabel("t-SNE 3")
    axis.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=7, label="feature families"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=10, label="target families"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="tab:purple", markersize=7, label="temporal year tasks"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="tab:cyan", markersize=7, label="cross-sectional year tasks"),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.99))
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Historical MTL")
    parser.add_argument("--prototype", choices=("all", "mean", "min", "max", "rmse", "q25", "q50", "q75"), default="all")
    parser.add_argument("--aggregate-prototypes", action="store_true", help="Average the plotted prototype coordinates into one point per task and label.")
    args = parser.parse_args()

    frame = pd.read_csv(args.coordinates)
    if args.prototype != "all":
        frame = frame.loc[frame["prototype"].eq(args.prototype)].copy()
    if args.aggregate_prototypes:
        frame["label"] = frame["label"].str.replace(r" \[(mean|min|max|rmse|q25|q50|q75)\]$", "", regex=True)
        frame = (
            frame.groupby(["task", "label"], as_index=False)
            .agg(x=("x", "mean"), y=("y", "mean"), z=("z", "mean"), support=("support", "sum"))
        )
        frame["prototype"] = "prototype_mean"
    suffix = "_prototype_mean" if args.aggregate_prototypes else ("" if args.prototype == "all" else f"_{args.prototype}")
    target_labels = {
        str(label) for label in frame.loc[frame["task"].eq("family"), "label"]
        if str(label).startswith("equity.")
    }
    _save_view(
        frame,
        args.output_dir / f"prototype_embeddings_tsne_3d{suffix}.png",
        f"{args.title_prefix} — {'mean of prototype coordinates' if args.aggregate_prototypes else f'{args.prototype} prototype embeddings'}",
        target_labels=target_labels,
    )


if __name__ == "__main__":
    main()
