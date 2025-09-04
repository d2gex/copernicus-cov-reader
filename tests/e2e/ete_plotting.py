# ete_plotting.py
from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.copernicus.tile_catalog import TileCatalog

# ---------- pure helpers ----------


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Compute cell edges from 1D center coordinates (monotonic)."""
    centers = np.asarray(centers)
    d = np.diff(centers)
    left = centers[0] - 0.5 * d[0]
    right = centers[-1] + 0.5 * d[-1]
    mids = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate([[left], mids, [right]])


def _edge_index_for_value(edges: np.ndarray, value: float) -> int:
    """
    Return index i such that edges[i] <= value < edges[i+1], clamped to [0, len(edges)-2].
    """
    i = int(np.searchsorted(edges, value, side="right") - 1)
    if i < 0:
        return 0
    last = len(edges) - 2
    if i > last:
        return last
    return i


def _rect_from_center(
    cx: float, cy: float, lon_edges: np.ndarray, lat_edges: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Given a cell center, compute the rectangle [x,y,w,h] using edges via searchsorted.
    """
    j = _edge_index_for_value(lon_edges, cx)
    i = _edge_index_for_value(lat_edges, cy)
    x = float(lon_edges[j])
    y = float(lat_edges[i])
    w = float(lon_edges[j + 1] - lon_edges[j])
    h = float(lat_edges[i + 1] - lat_edges[i])
    return x, y, w, h


def _unique_tile_ids(details: pd.DataFrame, col: str) -> np.ndarray:
    vals = details[col].to_numpy()
    return np.unique(vals.astype(np.int64))


def _solution_meta(kind: str) -> Tuple[str, str]:
    if kind == "kd":
        return "kd_tile_id", "KDIndex — haul→tile"
    if kind == "ref":
        return "ref_tile_id", "Brute force (KD metric) — haul→tile"
    raise ValueError("kind must be 'kd' or 'ref'")


# ---------- drawing primitives ----------


def _draw_tiles(
    ax: plt.Axes,
    catalog: TileCatalog,
    tile_ids: Iterable[int],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
) -> None:
    for tid in tile_ids:
        cx, cy = catalog.sea_cell_coords(int(tid))  # center coordinates
        x, y, w, h = _rect_from_center(cx, cy, lon_edges, lat_edges)
        ax.add_patch(Rectangle((x, y), w, h, fill=False))
        ax.text(cx, cy, str(int(tid)), ha="center", va="center")


def _scatter_hauls(ax: plt.Axes, details: pd.DataFrame) -> None:
    ax.scatter(details["haul_lon"].to_numpy(), details["haul_lat"].to_numpy(), s=12)


# ---------- public API ----------


def plot_solution_map(
    details: pd.DataFrame,
    catalog: TileCatalog,
    *,
    kind: str,  # "kd" or "ref"
    title: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draws tiles involved in this solution (KD or reference), labels by tile_id, and overlays hauls.
    """
    col, default_title = _solution_meta(kind)
    used_ids = _unique_tile_ids(details, col)

    # Grid centers and edges from the catalog's grid
    lons = np.asarray(catalog.grid.lons)
    lats = np.asarray(catalog.grid.lats)
    lon_edges = _edges_from_centers(lons)
    lat_edges = _edges_from_centers(lats)

    fig, ax = plt.subplots()
    _draw_tiles(ax, catalog, used_ids, lon_edges, lat_edges)
    _scatter_hauls(ax, details)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or default_title)
    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def plot_both_maps(
    details: pd.DataFrame, catalog: TileCatalog
) -> Tuple[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes]]:
    """
    Creates two separate figures: KDIndex and Brute force.
    """
    kd = plot_solution_map(details, catalog, kind="kd", title="KDIndex — haul→tile")
    ref = plot_solution_map(
        details, catalog, kind="ref", title="Brute force (KD metric) — haul→tile"
    )
    return kd, ref
