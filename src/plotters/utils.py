import os
from typing import Literal, Tuple, Union

import fiona
import geopandas as gpd
import pandas as pd

Kind = Literal["geopandas", "csv"]
TARGET_CRS = "EPSG:4326"  # Required by Copernicus Marine subset API


def read_coast_to_geoms(
    path: str, target_crs: str = TARGET_CRS
) -> Tuple[Kind, Union[gpd.GeoDataFrame, pd.DataFrame]]:
    """
    Read a coastline as geometries.

    Supports:
      - Shapefile / GeoJSON / GeoPackage: returns ("geopandas", GeoDataFrame in EPSG:4326)
      - CSV (lon,lat): returns ("csv", pandas.DataFrame with lon,lat)

    Returns:
        (kind, obj)
          kind="geopandas" -> obj: gpd.GeoDataFrame (EPSG:4326)
          kind="csv"       -> obj: pd.DataFrame with columns lon, lat
    """
    ext = os.path.splitext(path)[1].lower()

    # Fail fast for unsupported formats
    if ext not in {".shp", ".geojson", ".json", ".gpkg", ".csv", ".txt"}:
        raise ValueError(f"Unsupported coastline file extension: {ext}")

    # Vector formats
    if ext in {".shp", ".geojson", ".json", ".gpkg"}:
        with fiona.open(path, "r") as src:
            gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
        if gdf.crs and gdf.crs.to_epsg() != target_crs:
            gdf = gdf.to_crs(target_crs)
        return "geopandas", gdf

    # CSV formats
    with open(path, "r") as f:
        df = pd.read_csv(f)
    return "csv", df
