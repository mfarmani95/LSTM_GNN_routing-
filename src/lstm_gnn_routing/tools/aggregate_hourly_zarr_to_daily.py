from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

try:
    from numcodecs import Blosc
except ImportError:  # pragma: no cover - environment dependent
    Blosc = None


DEFAULT_AGGREGATIONS: dict[str, dict[str, Any]] = {
    "RAINRATE": {"op": "sum", "factor": 3600.0, "output_name": "RAINRATE_mm_day"},
    "LWDOWN": {"op": "mean", "factor": 1.0, "output_name": "LWDOWN_mean"},
    "SWDOWN": {"op": "mean", "factor": 1.0, "output_name": "SWDOWN_mean"},
    "T2D": {"op": "mean", "factor": 1.0, "output_name": "T2D_mean"},
    "Q2D": {"op": "mean", "factor": 1.0, "output_name": "Q2D_mean"},
    "PSFC": {"op": "mean", "factor": 1.0, "output_name": "PSFC_mean"},
    "U2D": {"op": "mean", "factor": 1.0, "output_name": "U2D_mean"},
    "V2D": {"op": "mean", "factor": 1.0, "output_name": "V2D_mean"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate yearly hourly forcing Zarr stores to yearly daily Zarr stores. "
            "RAINRATE is converted from hourly rate to daily depth by sum(rate * 3600); "
            "other default AORC variables are daily means."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Folder containing yearly hourly Zarr stores.")
    parser.add_argument("--output-root", type=Path, required=True, help="Folder for yearly daily Zarr stores.")
    parser.add_argument("--glob-pattern", default="*.zarr", help="Glob pattern for yearly input stores.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_AGGREGATIONS),
        help="Variables to aggregate. Defaults to the standard AORC forcing variables.",
    )
    parser.add_argument("--time-dim", default="time", help="Input time dimension name.")
    parser.add_argument("--daily-time-chunk", type=int, default=31, help="Output Zarr chunk size along daily time.")
    parser.add_argument("--y-chunk", type=int, default=128, help="Output Zarr chunk size along y.")
    parser.add_argument("--x-chunk", type=int, default=128, help="Output Zarr chunk size along x.")
    parser.add_argument("--compression-level", type=int, default=3, help="Blosc zstd compression level.")
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=(2,),
        default=2,
        help="Zarr format version to write. The converter writes Zarr v2 for numcodecs compatibility.",
    )
    parser.add_argument(
        "--valid-hour-counts",
        nargs="+",
        type=int,
        default=[8760, 8784],
        help="Skip yearly stores whose hourly time dimension is not one of these counts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing daily stores.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing daily stores instead of failing.")
    return parser.parse_args()


def _discover_yearly_zarr_stores(root: Path, glob_pattern: str) -> list[Path]:
    stores = sorted(path for path in root.glob(glob_pattern) if path.suffix == ".zarr")
    return [path for path in stores if path.name.removesuffix(".zarr").isdigit()]


def _cleanup_dataset(ds: xr.Dataset) -> xr.Dataset:
    drop_names: list[str] = []
    if "reference_time" in ds.variables:
        drop_names.append("reference_time")
    if "crs" in ds.data_vars:
        drop_names.append("crs")
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")
    return ds


def _daily_reduce(da: xr.DataArray, spec: Mapping[str, Any], time_dim: str) -> xr.DataArray:
    op = str(spec.get("op", "mean")).lower()
    factor = float(spec.get("factor", 1.0))
    scaled = da * factor
    resampler = scaled.resample({time_dim: "1D"})
    if op == "sum":
        return resampler.sum(skipna=True)
    if op == "mean":
        return resampler.mean(skipna=True)
    if op == "min":
        return resampler.min(skipna=True)
    if op == "max":
        return resampler.max(skipna=True)
    raise ValueError(f"Unsupported daily aggregation op '{op}'")


def _aggregate_store(store_path: Path, args: argparse.Namespace) -> tuple[xr.Dataset, xr.Dataset]:
    ds = xr.open_zarr(store_path, consolidated=True)
    ds = _cleanup_dataset(ds)
    missing = [name for name in args.variables if name not in ds.data_vars]
    if missing:
        ds.close()
        raise KeyError(f"Variables not found in {store_path}: {missing}")

    if args.time_dim not in ds.dims:
        ds.close()
        raise KeyError(f"Time dimension '{args.time_dim}' not found in {store_path}")

    count = int(ds.sizes[args.time_dim])
    if count not in set(int(value) for value in args.valid_hour_counts):
        ds.close()
        raise ValueError(
            f"{store_path.name} has {count} hourly steps; expected one of {list(args.valid_hour_counts)}"
        )

    daily_vars: dict[str, xr.DataArray] = {}
    for name in args.variables:
        spec = DEFAULT_AGGREGATIONS.get(str(name), {"op": "mean", "factor": 1.0, "output_name": f"{name}_mean"})
        out_name = str(spec.get("output_name", name))
        daily_vars[out_name] = _daily_reduce(ds[name], spec, args.time_dim)
    daily = xr.Dataset(daily_vars)
    for coord_name in (args.time_dim, "y", "x", "lat", "lon"):
        if coord_name in daily.coords:
            continue
        if coord_name in ds.coords and coord_name != args.time_dim:
            daily = daily.assign_coords({coord_name: ds.coords[coord_name]})
    return daily, ds


def _build_encoding(ds: xr.Dataset, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    if Blosc is None:
        raise ImportError("aggregate_hourly_zarr_to_daily.py requires numcodecs to write compressed Zarr stores")
    compressor = Blosc(cname="zstd", clevel=int(args.compression_level), shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict[str, object]] = {}
    for name in ds.data_vars:
        chunks = tuple(
            int(args.daily_time_chunk) if dim == args.time_dim else
            int(args.y_chunk) if dim == "y" else
            int(args.x_chunk) if dim == "x" else
            int(ds.sizes[dim])
            for dim in ds[name].dims
        )
        encoding[name] = {"compressor": compressor, "chunks": chunks}
    return encoding


def _resolve_zarr_version_kwargs(zarr_version: int) -> dict[str, int]:
    to_zarr_signature = inspect.signature(xr.Dataset.to_zarr)
    if "zarr_format" in to_zarr_signature.parameters:
        return {"zarr_format": int(zarr_version)}
    return {"zarr_version": int(zarr_version)}


def _prepare_store_path(store_path: Path, *, overwrite: bool, skip_existing: bool) -> bool:
    if not store_path.exists():
        store_path.parent.mkdir(parents=True, exist_ok=True)
        return True
    if skip_existing:
        print(f"[skip] {store_path.name}: output already exists")
        return False
    if not overwrite:
        raise FileExistsError(f"Output Zarr store already exists: {store_path}")
    if store_path.is_dir():
        shutil.rmtree(store_path)
    else:
        store_path.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return True


def main() -> None:
    args = _parse_args()
    args.input_root = Path(args.input_root)
    args.output_root = Path(args.output_root)

    stores = _discover_yearly_zarr_stores(args.input_root, str(args.glob_pattern))
    if not stores:
        raise FileNotFoundError(f"No yearly Zarr stores matched {args.glob_pattern} in {args.input_root}")

    converted = 0
    skipped = 0
    zarr_version_kwargs = _resolve_zarr_version_kwargs(int(args.zarr_version))
    for store_path in stores:
        year = store_path.name.removesuffix(".zarr")
        output_path = args.output_root / f"{year}.zarr"
        if not _prepare_store_path(output_path, overwrite=bool(args.overwrite), skip_existing=bool(args.skip_existing)):
            skipped += 1
            continue
        try:
            daily, source = _aggregate_store(store_path, args)
        except ValueError as exc:
            print(f"[skip] {store_path.name}: {exc}")
            skipped += 1
            continue
        print(f"[aggregate] {store_path.name} -> {output_path}")
        try:
            daily = daily.chunk({args.time_dim: int(args.daily_time_chunk), "y": int(args.y_chunk), "x": int(args.x_chunk)})
            daily.to_zarr(
                store=output_path,
                mode="w",
                consolidated=True,
                encoding=_build_encoding(daily, args),
                **zarr_version_kwargs,
            )
        finally:
            daily.close()
            source.close()
        converted += 1

    print(f"Done. Converted {converted} daily yearly stores, skipped {skipped}.")


if __name__ == "__main__":
    main()
