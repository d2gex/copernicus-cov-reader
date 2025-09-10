import numpy as np
import pandas as pd

from src.bounding_box import bounding_box


class BoundingBoxCalculator:
    """
    Compute a coarse bounding box for a set of haul coordinates, with optional padding.
    """

    def __init__(self, df: pd.DataFrame, pad_deg: float = 0.0):
        self.df = df
        self.pad = float(pad_deg)

    def run(self) -> bounding_box.BoundingBox:
        # Expect columns 'lon' and 'lat'
        lons = self.df["lon"].to_numpy()
        lats = self.df["lat"].to_numpy()

        min_lon = float(np.min(lons)) - self.pad
        max_lon = float(np.max(lons)) + self.pad
        min_lat = float(np.min(lats)) - self.pad
        max_lat = float(np.max(lats)) + self.pad

        return bounding_box.BoundingBox(
            min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat
        )
