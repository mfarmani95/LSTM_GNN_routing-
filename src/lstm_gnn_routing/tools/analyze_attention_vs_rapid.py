"""Relate upstream GAT attention diagnostics to GNN-vs-RAPID improvement."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _canonical_gauge_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(8) if digits else text


def _read_attention(attention_dir: Path) -> pd.DataFrame:
    path = attention_dir / "gauge_upstream_attention_metrics.csv"
    df = pd.read_csv(path, dtype={"gauge_id": str})
    df["gauge_id"] = df["gauge_id"].map(_canonical_gauge_id)
    return df


def _read_rapid_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path, dtype={"gauge_id": str})
    metrics["gauge_id"] = metrics["gauge_id"].map(_canonical_gauge_id)
    metrics = metrics[metrics["scale"].astype(str).str.lower().eq("daily")].copy()
    wide = metrics.pivot_table(index="gauge_id", columns="model", values=["kge", "nse", "rmse", "pbias"], aggfunc="first")
    wide.columns = [f"{metric}_{model.lower()}" for metric, model in wide.columns]
    wide = wide.reset_index()
    wide["kge_improvement"] = wide["kge_gnn"] - wide["kge_rapid"]
    wide["nse_improvement"] = wide["nse_gnn"] - wide["nse_rapid"]
    wide["rmse_reduction"] = wide["rmse_rapid"] - wide["rmse_gnn"]
    wide["abs_pbias_reduction"] = wide["pbias_rapid"].abs() - wide["pbias_gnn"].abs()
    return wide


def _add_derived_attention_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for attr in ("travel_time_proxy", "So", "Length_m", "n", "BtmWdth", "TopWdth"):
        weighted = f"attention_weighted_{attr}"
        mean = f"mean_upstream_{attr}"
        if weighted in out and mean in out:
            out[f"attention_weighted_minus_mean_{attr}"] = out[weighted] - out[mean]
    return out


def _save_correlations(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    response_cols = ["kge_improvement", "nse_improvement", "rmse_reduction", "abs_pbias_reduction"]
    predictor_cols = [
        "mean_upstream_attention",
        "top10pct_upstream_attention",
        "attention_near_confluences",
        "attention_on_mainstem_edges",
        "attention_weighted_minus_mean_travel_time_proxy",
        "attention_weighted_minus_mean_So",
        "attention_weighted_minus_mean_BtmWdth",
        "attention_weighted_minus_mean_TopWdth",
    ]
    rows: list[dict[str, float | str | int]] = []
    for predictor in predictor_cols:
        if predictor not in df:
            continue
        for response in response_cols:
            if response not in df:
                continue
            valid = df[[predictor, response]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3:
                continue
            rows.append(
                {
                    "predictor": predictor,
                    "response": response,
                    "n": int(len(valid)),
                    "spearman": float(valid[predictor].corr(valid[response], method="spearman")),
                    "pearson": float(valid[predictor].corr(valid[response], method="pearson")),
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(output_dir / "attention_vs_rapid_improvement_correlations.csv", index=False)
    return corr


def _scatter_with_labels(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    valid = df[[x_col, y_col, "gauge_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(valid[x_col], valid[y_col], s=42, alpha=0.78, color="#2f6f8f", edgecolor="white", linewidth=0.6)
    if len(valid) > 2:
        rho = valid[x_col].corr(valid[y_col], method="spearman")
        title = f"{title}\nSpearman rho={rho:.2f}"
    for _, row in valid.iterrows():
        ax.annotate(str(row["gauge_id"])[-4:], (row[x_col], row[y_col]), fontsize=7, alpha=0.72)
    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.6)


def _plot_attention_vs_improvement(df: pd.DataFrame, output_dir: Path) -> None:
    panels = [
        ("top10pct_upstream_attention", "kge_improvement", "High-attention upstream edges"),
        ("mean_upstream_attention", "kge_improvement", "Mean upstream attention"),
        ("attention_near_confluences", "kge_improvement", "Confluence attention"),
        ("attention_on_mainstem_edges", "kge_improvement", "Mainstem attention"),
        ("attention_weighted_minus_mean_travel_time_proxy", "kge_improvement", "Attention-weighted travel-time shift"),
        ("attention_weighted_minus_mean_So", "kge_improvement", "Attention-weighted slope shift"),
    ]
    panels = [panel for panel in panels if panel[0] in df and panel[1] in df]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax, (x_col, y_col, title) in zip(axes_arr, panels):
        _scatter_with_labels(ax, df, x_col, y_col, title)
    for ax in axes_arr[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Gauge-Level GNN Improvement over RAPID vs Upstream GAT Attention")
    fig.savefig(output_dir / "attention_vs_rapid_kge_improvement.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", required=True, help="Directory containing gauge_upstream_attention_metrics.csv.")
    parser.add_argument("--rapid-metrics-file", required=True, help="rapid_gnn_metrics.csv from analyze_rapid_vs_gnn.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to attention-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention_dir = Path(args.attention_dir).expanduser().resolve()
    rapid_metrics_file = Path(args.rapid_metrics_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else attention_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attention = _add_derived_attention_metrics(_read_attention(attention_dir))
    rapid = _read_rapid_metrics(rapid_metrics_file)
    joined = attention.merge(rapid, on="gauge_id", how="inner")
    joined.to_csv(output_dir / "gauge_attention_rapid_improvement.csv", index=False)
    _save_correlations(joined, output_dir)
    _plot_attention_vs_improvement(joined, output_dir)
    print(f"Saved attention-vs-RAPID diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
