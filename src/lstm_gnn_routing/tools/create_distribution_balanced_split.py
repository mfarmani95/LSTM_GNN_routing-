"""Create distribution-balanced non-overlapping time-block splits.

The split is based on observed gauge streamflow.  Each block receives one split
label, so later rolling samples can be assigned without leaking days across
train/validation/test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_gauge_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        gauge_id = text.split(",")[0].strip()
        if gauge_id.isdigit():
            gauge_id = gauge_id.zfill(8)
        ids.append(gauge_id)
    return ids


def _read_streamflow_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = None
    best_date_count = -1
    for candidate in df.columns:
        parsed = pd.to_datetime(df[candidate], errors="coerce")
        count = int(parsed.notna().sum())
        if candidate in {"datetime", "date", "Date", "DATE", "time", "Time"}:
            count += 1_000_000
        if count > best_date_count:
            date_col = candidate
            best_date_count = count
    if date_col is None:
        raise ValueError(f"Could not find date column in {path}")

    value_col = None
    preferred = ["streamflow", "discharge", "flow", "q", "Q", "00060_Mean", "00060"]
    for candidate in preferred:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        candidates = []
        for c in df.columns:
            if c == date_col:
                continue
            lc = str(c).lower()
            if any(token in lc for token in ["site", "agency", "tz", "code", "qual", "_cd", "approval"]):
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            count = int(converted.notna().sum())
            if count > 0:
                # Prefer variable-looking columns with some dynamic range.
                spread = float(converted.quantile(0.95) - converted.quantile(0.05)) if count > 3 else 0.0
                candidates.append((count, spread, c))
        if not candidates:
            raise ValueError(f"Could not infer streamflow column in {path}")
        candidates.sort()
        value_col = candidates[-1][2]

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=dates).dropna()
    series = series[~series.index.isna()].sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series


def _load_observations(streamflow_dir: Path, gauge_ids: list[str]) -> pd.DataFrame:
    columns = {}
    missing = []
    for gauge_id in gauge_ids:
        path = streamflow_dir / f"{gauge_id}.csv"
        if not path.exists():
            missing.append(gauge_id)
            continue
        columns[gauge_id] = _read_streamflow_csv(path)
    if not columns:
        raise ValueError(f"No streamflow CSVs found in {streamflow_dir}")
    if missing:
        print(f"Warning: missing {len(missing)} gauge CSVs: {', '.join(missing[:10])}")
    return pd.DataFrame(columns).sort_index()


def _assign_splits(
    block_scores: pd.Series,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    n_bins: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    scores = block_scores.to_numpy()
    finite = np.isfinite(scores)
    if finite.sum() < 3:
        raise ValueError("Need at least three finite block scores to create train/validation/test split")

    bins = pd.qcut(block_scores[finite], q=min(n_bins, int(finite.sum())), labels=False, duplicates="drop")
    assignments = pd.Series(index=block_scores.index, dtype=object)

    for _, bin_blocks in pd.Series(block_scores[finite].index, index=bins).groupby(level=0):
        block_ids = np.array(list(bin_blocks.values))
        rng.shuffle(block_ids)
        n = len(block_ids)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * validation_fraction))) if n - n_train > 1 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1
        train_ids = block_ids[:n_train]
        val_ids = block_ids[n_train : n_train + n_val]
        test_ids = block_ids[n_train + n_val :]
        assignments.loc[train_ids] = "train"
        assignments.loc[val_ids] = "validation"
        assignments.loc[test_ids] = "test"

    # Any non-finite blocks are safest in training only if we keep them at all.
    assignments = assignments.fillna("train")
    return assignments


def _plot_cdf(block_table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for split, group in block_table.groupby("split"):
        values = np.sort(group["score"].dropna().to_numpy())
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.plot(values, y, label=f"{split} (n={values.size})", linewidth=2)
    ax.set_xlabel("Block log1p mean observed streamflow")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streamflow-dir", type=Path, default=Path("data/streamflow/daily"))
    parser.add_argument("--basin-file", type=Path, default=Path("data/streamflow/26_basin_ids_ngen.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/distribution_balanced_365day_split_seed619873.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("data/splits/distribution_balanced_365day_split_seed619873_cdf.png"))
    parser.add_argument("--block-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=619873)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    gauge_ids = _read_gauge_ids(args.basin_file)
    obs = _load_observations(args.streamflow_dir, gauge_ids)
    if args.start_date:
        obs = obs.loc[pd.to_datetime(args.start_date) :]
    if args.end_date:
        obs = obs.loc[: pd.to_datetime(args.end_date)]

    obs = obs.dropna(how="all")
    if obs.empty:
        raise ValueError("No observations remain after date filtering")

    start = obs.index.min().normalize()
    end = obs.index.max().normalize()
    block_starts = pd.date_range(start=start, end=end, freq=f"{args.block_days}D")

    rows = []
    for block_id, block_start in enumerate(block_starts):
        block_end = block_start + pd.Timedelta(days=args.block_days - 1)
        block = obs.loc[block_start:block_end]
        if block.empty:
            continue
        score = np.log1p(block).stack().mean()
        rows.append(
            {
                "block_id": block_id,
                "start_date": block_start.date().isoformat(),
                "end_date": min(block_end, end).date().isoformat(),
                "n_days": int(block.shape[0]),
                "n_valid_values": int(block.notna().sum().sum()),
                "score": float(score) if np.isfinite(score) else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    table["split"] = _assign_splits(
        table.set_index("block_id")["score"],
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        n_bins=args.n_bins,
    ).reindex(table["block_id"]).to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    _plot_cdf(table, args.plot_output)

    print(f"Wrote split: {args.output}")
    print(f"Wrote CDF plot: {args.plot_output}")
    print(table["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
