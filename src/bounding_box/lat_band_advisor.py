import numpy as np

from src.bounding_box.bounding_box import BoundingBox


class LatBandAdvisor:
    @staticmethod
    def recommend_n_bands(
        bbox: BoundingBox,
        cell_deg: float = 0.083,
        max_err_km2: float = 3.0,
    ) -> int:
        """
        Choose the number of latitude bands so that the per-cell area error
        from using a single cos(lat0) within a band is <= max_err_km2.

        Uses a conservative bound with sin(phi_worst) at bbox.max_lat.
        """
        square_deg_to_km = 111.32**2  # km^2 per degree^2
        square_degree_cell = float(cell_deg) ** 2
        phi_worst = np.deg2rad(bbox.max_lat)
        s = np.sin(phi_worst)
        if s <= 0:
            return 1
        # half-band in radians
        h_rad = max_err_km2 / (square_deg_to_km * square_degree_cell * s)
        if h_rad <= 0:
            return 1
        h_deg = np.degrees(h_rad)
        band_height = 2.0 * h_deg
        lat_span = bbox.max_lat - bbox.min_lat
        n = int(np.ceil(lat_span / band_height))
        return max(n, 1)
