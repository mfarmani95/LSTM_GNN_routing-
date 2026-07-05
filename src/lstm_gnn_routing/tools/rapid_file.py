from __future__ import annotations

import os
from pathlib import Path

import xarray as xr


DEFAULT_RAPID_CANDIDATES = [
    Path("/xdisk/tyferre/farmani/Graph_Routing/RAPID_13nd.nc"),
    Path("/xdisk/tyferre/farmani/Graph_Routing/rapid.nc"),
    Path("/xdisk/tyferre/farmani/Graph_Routing/RAPID.nc"),
]


def _has_qout(path: Path, qout_var: str) -> bool:
    try:
        with xr.open_dataset(path) as ds:
            return qout_var in ds.data_vars
    except Exception:
        return False


def detect_rapid_file(
    rapid_file: str | Path | None = None,
    *,
    qout_var: str = "Qout",
    search_root: str | Path = "/xdisk/tyferre/farmani/Graph_Routing",
) -> Path:
    """Resolve a RAPID NetCDF file.

    Priority:
      1. Explicit function argument.
      2. RAPID_FILE environment variable.
      3. Known project defaults.
      4. Files under search_root whose names contain "rapid".

    When possible we prefer candidates containing the requested Qout variable.
    """

    explicit = rapid_file or os.environ.get("RAPID_FILE")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"RAPID file was specified but does not exist: {path}")
        return path

    candidates: list[Path] = []
    for path in DEFAULT_RAPID_CANDIDATES:
        if path.is_file():
            candidates.append(path)

    root = Path(search_root).expanduser()
    if root.exists():
        for pattern in ("*rapid*.nc", "*RAPID*.nc", "*Rapid*.nc"):
            candidates.extend(root.glob(pattern))
            candidates.extend((root / "LSTM_GNN_routing-").glob(pattern) if (root / "LSTM_GNN_routing-").exists() else [])

    unique: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if not unique:
        raise FileNotFoundError(
            "Could not detect a RAPID NetCDF file. Provide --rapid-file or set RAPID_FILE=/path/to/RAPID.nc."
        )

    qout_matches = [path for path in unique if _has_qout(path, qout_var)]
    if qout_matches:
        return qout_matches[0]
    return unique[0]
