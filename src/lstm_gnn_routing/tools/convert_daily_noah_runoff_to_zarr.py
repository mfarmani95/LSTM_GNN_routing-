from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


FILENAME_RE = re.compile(r"Noah\.dailymean\.(\d{8})\.nc$")
SECONDS_PER_DAY = 86400.0


def _parse_date(path: Path) -> pd.Timestamp | None:
    match = FILENAME_RE.fullmatch(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert daily Noah runoff NetCDF files into yearly Zarr stores."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glob", default="Noah.dailymean.*.nc")
    parser.add_argument("--variables", nargs="+", default=["RUNSF", "RUNSB"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-time", type=int, default=64)
    parser.add_argument("--chunk-y", type=int, default=128)
    parser.add_argument("--chunk-x", type=int, default=128)
    return parser


def _collect_year_groups(input_dir: Path, glob_pattern: str) -> dict[int, list[tuple[pd.Timestamp, Path]]]:
    groups: dict[int, list[tuple[pd.Timestamp, Path]]] = defaultdict(list)
    for path in sorted(input_dir.glob(glob_pattern)):
        if not path.is_file():
            continue
        timestamp = _parse_date(path)
        if timestamp is None:
            continue
        groups[int(timestamp.year)].append((timestamp, path))
    return groups


def _load_year_dataset(
    entries: list[tuple[pd.Timestamp, Path]],
    *,
    variables: list[str],
) -> xr.Dataset:
    if not entries:
        raise ValueError("Cannot build a yearly runoff dataset from zero files")

    times = [timestamp for timestamp, _ in entries]
    first_ds = xr.open_dataset(entries[0][1])
    try:
        ny = int(first_ds.sizes["ny"])
        nx = int(first_ds.sizes["nx"])
        lat = np.asarray(first_ds["lat"].to_numpy(), dtype=np.float32)
        lon = np.asarray(first_ds["lon"].to_numpy(), dtype=np.float32)
    finally:
        first_ds.close()

    arrays = {
        name: np.empty((len(entries), ny, nx), dtype=np.float32)
        for name in variables
    }

    for time_index, (_, path) in enumerate(tqdm(entries, desc=f"load {times[0].year}", dynamic_ncols=True, leave=False)):
        ds = xr.open_dataset(path)
        try:
            for name in variables:
                if name not in ds:
                    raise KeyError(f"Variable '{name}' not found in {path}")
                values = np.asarray(ds[name].to_numpy(), dtype=np.float32)
                missing_value = ds[name].attrs.get("missing_value", ds.attrs.get("missing_value"))
                if missing_value is not None:
                    values = np.where(values == np.float32(missing_value), np.nan, values)
                arrays[name][time_index] = values * np.float32(SECONDS_PER_DAY)
        finally:
            ds.close()

    data_vars = {}
    for name in variables:
        attrs = {
            "long_name": f"{name} daily runoff depth",
            "units": "mm/day",
            "source_units": "mm/s",
            "conversion": "daily_mean_rate_times_86400",
        }
        data_vars[name] = (("time", "y", "x"), arrays[name], attrs)

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": pd.DatetimeIndex(times),
            "y": np.arange(ny, dtype=np.int32),
            "x": np.arange(nx, dtype=np.int32),
            "lat": ("y", lat),
            "lon": ("x", lon),
        },
        attrs={
            "title": "Daily Noah runoff converted from daily-mean mm/s to mm/day",
            "variables": ",".join(variables),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    year_groups = _collect_year_groups(args.input_dir, args.glob)
    if not year_groups:
        raise FileNotFoundError(
            f"No daily Noah runoff files matching {args.glob} were found in {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in sorted(year_groups):
        store_path = args.output_dir / f"{year}.zarr"
        if store_path.exists():
            if not args.overwrite:
                print(f"[skip] {year}: {store_path} exists")
                continue
            if store_path.is_dir():
                import shutil

                shutil.rmtree(store_path)
            else:
                store_path.unlink()

        entries = sorted(year_groups[year], key=lambda item: item[0])
        print(f"[convert] {year}: {len(entries)} daily files -> {store_path}")
        ds = _load_year_dataset(entries, variables=list(args.variables))
        try:
            chunk_time = min(int(args.chunk_time), int(ds.sizes["time"]))
            ds = ds.chunk({"time": max(chunk_time, 1), "y": int(args.chunk_y), "x": int(args.chunk_x)})
            ds.to_zarr(
                store_path,
                mode="w",
                consolidated=True,
                zarr_format=2,
            )
        finally:
            ds.close()


if __name__ == "__main__":
    main()
