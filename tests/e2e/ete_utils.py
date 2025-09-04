# kd_validate_runner.py
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.copernicus.kd_index import KDIndex

# Use your existing project types
from src.copernicus.tile_catalog import TileCatalog

# -------- I/O (edge only) --------


def load_layers(dynamic_path: str, static_path: str) -> Tuple[xr.Dataset, xr.Dataset]:
    with xr.open_dataset(dynamic_path) as ds_dyn:
        dyn = ds_dyn.load()
    with xr.open_dataset(static_path) as ds_sta:
        sta = ds_sta.load()
    return dyn, sta


# -------- Build runtime (uses your classes) --------


@dataclass(frozen=True)
class Runtime:
    catalog: TileCatalog
    kd: KDIndex
    lat0: float  # mean sea-lat for the KD metric


def build_runtime_from_datasets(
    dynamic_ds: xr.Dataset, static_ds: xr.Dataset, mask_name: str
) -> Runtime:
    mask_da = static_ds[mask_name]
    catalog = TileCatalog.from_dataset(dynamic_ds, mask=mask_da)
    kd = KDIndex(catalog)  # KDIndex uses the same sea centers
    _, sea_lat = catalog.sea_tile_coords()
    lat0 = float(np.nanmean(sea_lat))
    return Runtime(catalog=catalog, kd=kd, lat0=lat0)


# -------- Small, pure helpers --------


def kd_transform(
    lon: np.ndarray, lat: np.ndarray, *, lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    scale = cos(radians(lat0))
    return lon * scale, lat


def tile_center(catalog: TileCatalog, tile_id: int) -> Tuple[float, float]:
    return catalog.sea_cell_coords(tile_id)


def tile_ij(catalog: TileCatalog, tile_id: int) -> Tuple[int, int]:
    j, i = catalog.sea_cell_ids(tile_id)  # your API returns (j,i)
    return int(i), int(j)


# -------- Neighbor logic (pure) --------


def neighbor_direction(a_id: int, b_id: int, catalog: TileCatalog) -> Tuple[bool, str]:
    if a_id == b_id:
        return True, "same"
    ai, aj = tile_ij(catalog, a_id)
    bi, bj = tile_ij(catalog, b_id)
    if abs(ai - bi) > 1 or abs(aj - bj) > 1:
        return False, ""
    lons = catalog.grid.lons
    lats = catalog.grid.lats
    dlat = np.sign(lats[bi] - lats[ai])  # +1 north, -1 south
    dlon = np.sign(lons[bj] - lons[aj])  # +1 east,  -1 west
    if dlat > 0 and dlon == 0:
        return True, "N"
    if dlat < 0 and dlon == 0:
        return True, "S"
    if dlat == 0 and dlon > 0:
        return True, "E"
    if dlat == 0 and dlon < 0:
        return True, "W"
    if dlat > 0 and dlon > 0:
        return True, "NE"
    if dlat > 0 and dlon < 0:
        return True, "NW"
    if dlat < 0 and dlon > 0:
        return True, "SE"
    if dlat < 0 and dlon < 0:
        return True, "SW"
    return False, ""


# -------- Brute force (same layer + metric as KD) --------


def bruteforce_single(
    haul_lon: float, haul_lat: float, *, catalog: TileCatalog, lat0: float
) -> Tuple[int, float]:
    sea_lon, sea_lat = catalog.sea_tile_coords()  # aligned with tile_id
    xs, ys = kd_transform(np.asarray(sea_lon), np.asarray(sea_lat), lat0=lat0)
    xh, yh = kd_transform(np.array([haul_lon]), np.array([haul_lat]), lat0=lat0)
    d2 = (xs - xh[0]) ** 2 + (ys - yh[0]) ** 2
    k = int(np.argmin(d2))  # this index IS the tile_id in your design
    return k, float(d2[k])


def bruteforce_batch(
    lons: Iterable[float], lats: Iterable[float], *, catalog: TileCatalog, lat0: float
) -> np.ndarray:
    ids = []
    for lo, la in zip(lons, lats):
        k, _ = bruteforce_single(float(lo), float(la), catalog=catalog, lat0=lat0)
        ids.append(k)
    return np.asarray(ids, dtype=np.int64)


# -------- Sampling (pure) --------


def sample_hauls(df: pd.DataFrame, n: int, *, seed: int = 0) -> pd.DataFrame:
    base = df.reset_index(drop=False)  # keep original index
    if n >= len(base):
        return base
    return base.sample(n=n, random_state=seed).reset_index(drop=True)


# -------- Row + validation (pure + DI) --------


def build_result_row(
    haul_row: pd.Series, kd_id: int, ref_id: int, catalog: TileCatalog
) -> Dict[str, object]:
    kd_lon, kd_lat = tile_center(catalog, kd_id)
    ref_lon, ref_lat = tile_center(catalog, ref_id)
    is_nb, direction = neighbor_direction(kd_id, ref_id, catalog)
    match_type = (
        "same" if kd_id == ref_id else ("neighbor" if is_nb else "non-neighbor")
    )
    if match_type == "same":
        direction = "same"
    return {
        "haul_index": haul_row.get("index", haul_row.name),
        "haul_lon": float(haul_row["lon"]),
        "haul_lat": float(haul_row["lat"]),
        "kd_tile_id": int(kd_id),
        "kd_tile_lon": kd_lon,
        "kd_tile_lat": kd_lat,
        "ref_tile_id": int(ref_id),
        "ref_tile_lon": ref_lon,
        "ref_tile_lat": ref_lat,
        "match_type": match_type,
        "neighbor_direction": direction or "",
    }


def validate_sample(
    hauls_sample: pd.DataFrame, rt: Runtime
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    kd_ids = rt.kd.query_many(
        hauls_sample["lon"].to_numpy(), hauls_sample["lat"].to_numpy()
    )
    ref_ids = bruteforce_batch(
        hauls_sample["lon"], hauls_sample["lat"], catalog=rt.catalog, lat0=rt.lat0
    )

    rows: List[Dict[str, object]] = []
    for i in range(len(hauls_sample)):
        rows.append(
            build_result_row(
                hauls_sample.iloc[i], int(kd_ids[i]), int(ref_ids[i]), rt.catalog
            )
        )
    details = pd.DataFrame(rows)

    same = (details["match_type"] == "same").mean() * 100.0
    nb = (details["match_type"] == "neighbor").mean() * 100.0
    non = (details["match_type"] == "non-neighbor").mean() * 100.0
    return details, {"same_pct": same, "neighbor_pct": nb, "non_neighbor_pct": non}
