from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src import bounding_box
from src.plotters import utils


class CoastHaulsPlotter:
    """
    Plot hauls + coarse bbox + coastline, and report any hauls outside the bbox.

    Dependencies are injected:
      - hauls_df: DataFrame with columns lon, lat
      - bbox: BoundingBox from BoundingBoxCalculator
      - coast_path: path to coast file (shp/geojson/gpkg or CSV lon,lat)
    """

    def __init__(
        self, hauls_df: pd.DataFrame, bbox: bounding_box.BoundingBox, coast_path: str
    ):
        self.hauls_df = hauls_df
        self.bbox = bbox
        self.coast_path = coast_path

    def find_orphans(self) -> pd.DataFrame:
        """Return hauls that fall outside the provided bounding box."""
        mnx, mxx, mny, mxy = self.bbox
        mask = ~(
            (self.hauls_df["lon"] >= mnx)
            & (self.hauls_df["lon"] <= mxx)
            & (self.hauls_df["lat"] >= mny)
            & (self.hauls_df["lat"] <= mxy)
        )
        return self.hauls_df.loc[mask].copy()

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Hauls + Coarse BBox + Coastline",
        point_size: float = 6.0,
        fig_size: Tuple[float, float] = (9, 9),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw the plot on the provided axes (or create new ones) and return (fig, ax).
        Does not call savefig() or show().
        """
        kind, coast = utils.read_coast_to_geoms(self.coast_path)

        # Setup axes/figure
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
            created_fig = True
        else:
            fig = ax.figure

        mnx, mxx, mny, mxy = self.bbox

        # Coastline first (so points/bbox sit on top)
        if kind == "geopandas":
            coast.boundary.plot(ax=ax, linewidth=0.8)
        else:  # CSV polyline
            ax.plot(coast["lon"], coast["lat"], linewidth=0.8)

        # Hauls
        ax.scatter(self.hauls_df["lon"], self.hauls_df["lat"], s=point_size)

        # BBox rectangle
        rect = Rectangle(
            (mnx, mny), (mxx - mnx), (mxy - mny), fill=False, linewidth=1.5
        )
        ax.add_patch(rect)

        # Axes/labels
        ax.set_title(title)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        # Tight view with margin
        pad_x = max(0.1, 0.05 * (mxx - mnx))
        pad_y = max(0.1, 0.05 * (mxy - mny))
        ax.set_xlim(mnx - pad_x, mxx + pad_x)
        ax.set_ylim(mny - pad_y, mxy + pad_y)

        if created_fig:
            fig.tight_layout()

        return fig, ax
