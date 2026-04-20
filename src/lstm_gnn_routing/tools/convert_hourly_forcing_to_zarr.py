from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import xarray as xr
from numcodecs import Blosc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert hourly Routing forcing NetCDF files arranged in year folders "
            "to Zarr stores. In yearly mode, only complete years are converted."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root forcing folder containing year subfolders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output folder for generated Zarr stores.")
    parser.add_argument(
        "--mode",
        choices=("yearly", "single"),
        default="yearly",
        help="Write one Zarr store per year, or one combined store for all years.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.LDASIN_DOMAIN1",
        help="Glob pattern for hourly forcing files inside each year folder.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["RAINRATE", "LWDOWN", "SWDOWN", "T2D", "Q2D", "PSFC", "U2D", "V2D"],
        help="Variables to keep in the output Zarr store. Use '--variables all' to convert every data variable.",
    )
    parser.add_argument("--time-dim", default="time", help="Time dimension name in the forcing files.")
    parser.add_argument("--y-dim", default="y", help="Y dimension name in the forcing files.")
    parser.add_argument("--x-dim", default="x", help="X dimension name in the forcing files.")
    parser.add_argument(
        "--batch-hours",
        type=int,
        default=24 * 7,
        help="How many hourly files to append in one batch while writing Zarr.",
    )
    parser.add_argument("--time-chunk", type=int, default=24 * 7, help="Zarr chunk size along time.")
    parser.add_argument("--y-chunk", type=int, default=128, help="Zarr chunk size along y.")
    parser.add_argument("--x-chunk", type=int, default=128, help="Zarr chunk size along x.")
    parser.add_argument("--compression-level", type=int, default=3, help="Blosc zstd compression level.")
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=(2,),
        default=2,
        help="Zarr format version to write. The current converter uses numcodecs Blosc and writes Zarr v2.",
    )
    parser.add_argument(
        "--valid-year-counts",
        nargs="+",
        type=int,
        default=[8760, 8784],
        help="Valid hourly file counts for a complete year in yearly mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output Zarr store if it already exists.",
    )
    return parser.parse_args()


def _discover_year_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def _chunked(items: Sequence[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _cleanup_dataset(ds: xr.Dataset) -> xr.Dataset:
    drop_names: list[str] = []
    if "reference_time" in ds.variables:
        drop_names.append("reference_time")
    if "crs" in ds.data_vars:
        drop_names.append("crs")
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")
    return ds


def _resolve_variables_from_file(file_path: Path, requested_variables: Sequence[str]) -> list[str]:
    requested = [str(name) for name in requested_variables]
    if len(requested) != 1 or requested[0].lower() != "all":
        return requested

    ds = xr.open_dataset(file_path)
    try:
        ds = _cleanup_dataset(ds)
        variables = list(ds.data_vars)
    finally:
        ds.close()

    if not variables:
        raise RuntimeError(f"No data variables found in {file_path} after cleanup.")
    return variables


def _open_hourly_batch(files: Sequence[Path], variables: Sequence[str], time_dim: str) -> xr.Dataset:
    datasets = []
    for path in files:
        ds = xr.open_dataset(path)
        ds = _cleanup_dataset(ds)
        missing = [name for name in variables if name not in ds.data_vars]
        if missing:
            ds.close()
            raise KeyError(f"Variables not found in {path}: {missing}")
        datasets.append(ds[list(variables)])

    combined = xr.concat(
        datasets,
        dim=time_dim,
        coords="minimal",
        compat="override",
        join="outer",
    )
    if time_dim in combined.coords:
        combined = combined.sortby(time_dim)
    return combined


def _build_encoding(ds: xr.Dataset, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    compressor = Blosc(cname="zstd", clevel=int(args.compression_level), shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict[str, object]] = {}
    for name in ds.data_vars:
        chunks = tuple(
            int(args.time_chunk) if dim == args.time_dim else
            int(args.y_chunk) if dim == args.y_dim else
            int(args.x_chunk) if dim == args.x_dim else
            int(ds.sizes[dim])
            for dim in ds[name].dims
        )
        encoding[name] = {"compressor": compressor, "chunks": chunks}
    return encoding


def _prepare_store_path(store_path: Path, overwrite: bool) -> None:
    if store_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output Zarr store already exists: {store_path}")
        if store_path.is_dir():
            shutil.rmtree(store_path)
        else:
            store_path.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_zarr_version_kwargs(zarr_version: int) -> dict[str, int]:
    to_zarr_signature = inspect.signature(xr.Dataset.to_zarr)
    if "zarr_format" in to_zarr_signature.parameters:
        return {"zarr_format": int(zarr_version)}
    return {"zarr_version": int(zarr_version)}


def _write_store_from_files(
    files: Sequence[Path],
    store_path: Path,
    args: argparse.Namespace,
    *,
    variables: Sequence[str],
) -> None:
    _prepare_store_path(store_path, overwrite=bool(args.overwrite))

    first = True
    zarr_version_kwargs = _resolve_zarr_version_kwargs(int(args.zarr_version))
    for batch_files in _chunked(list(files), int(args.batch_hours)):
        ds = _open_hourly_batch(batch_files, variables, args.time_dim)
        try:
            ds = ds.chunk({args.time_dim: int(args.time_chunk), args.y_dim: int(args.y_chunk), args.x_dim: int(args.x_chunk)})
            write_kwargs = dict(
                store=store_path,
                mode="w" if first else "a",
                append_dim=None if first else args.time_dim,
                consolidated=True,
                **zarr_version_kwargs,
            )
            if first:
                write_kwargs["encoding"] = _build_encoding(ds, args)
            ds.to_zarr(
                **write_kwargs,
            )
        finally:
            ds.close()
        first = False


def _convert_yearly(args: argparse.Namespace) -> None:
    year_dirs = _discover_year_dirs(args.input_root)
    if not year_dirs:
        raise FileNotFoundError(f"No year subfolders found under {args.input_root}")

    converted = 0
    skipped = 0
    valid_counts = set(int(value) for value in args.valid_year_counts)

    for year_dir in year_dirs:
        files = sorted(path for path in year_dir.glob(args.glob_pattern) if path.is_file())
        count = len(files)
        if count == 0:
            print(f"[skip] {year_dir.name}: no files matched '{args.glob_pattern}'")
            skipped += 1
            continue
        if count not in valid_counts:
            print(
                f"[skip] {year_dir.name}: file count {count} is incomplete "
                f"(expected one of {sorted(valid_counts)})"
            )
            skipped += 1
            continue

        variables = _resolve_variables_from_file(files[0], args.variables)
        store_path = args.output_root / f"{year_dir.name}.zarr"
        print(f"[convert] {year_dir.name}: {count} hourly files, {len(variables)} variables -> {store_path}")
        _write_store_from_files(files, store_path, args, variables=variables)
        converted += 1

    print(f"Done. Converted {converted} yearly stores, skipped {skipped}.")


def _convert_single(args: argparse.Namespace) -> None:
    year_dirs = _discover_year_dirs(args.input_root)
    if not year_dirs:
        raise FileNotFoundError(f"No year subfolders found under {args.input_root}")

    all_files: list[Path] = []
    for year_dir in year_dirs:
        files = sorted(path for path in year_dir.glob(args.glob_pattern) if path.is_file())
        if files:
            all_files.extend(files)

    if not all_files:
        raise FileNotFoundError(f"No forcing files matched '{args.glob_pattern}' under {args.input_root}")

    variables = _resolve_variables_from_file(all_files[0], args.variables)
    store_path = args.output_root / "forcing_all_years.zarr"
    print(f"[convert] single store: {len(all_files)} hourly files, {len(variables)} variables -> {store_path}")
    _write_store_from_files(all_files, store_path, args, variables=variables)
    print("Done. Converted one combined Zarr store.")


def main() -> None:
    args = _parse_args()
    args.input_root = Path(args.input_root)
    args.output_root = Path(args.output_root)

    if args.mode == "yearly":
        _convert_yearly(args)
    elif args.mode == "single":
        _convert_single(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
