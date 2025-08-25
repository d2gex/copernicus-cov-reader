from unittest.mock import Mock

import numpy as np
import pytest


# --- 3×3 rectilinear grid (all sea) ---
@pytest.fixture
def grid_3x3():
    lons = np.array([0.0, 1.0, 2.0], dtype=float)
    lats = np.array([10.0, 11.0, 12.0], dtype=float)
    g = Mock()
    g.lons = lons
    g.lats = lats
    g.nx = lons.size
    g.ny = lats.size
    return g


@pytest.fixture
def mask_all_sea():
    return np.ones((3, 3), dtype=bool)


# --- 2×2 high-lat grid for the cosine counterexample ---
@pytest.fixture
def grid_counterexample():
    # lon: [0.5, 1.0], lat: [60.0, 60.45]
    lons = np.array([0.5, 1.0], dtype=float)
    lats = np.array([60.0, 60.45], dtype=float)
    g = Mock()
    g.lons = lons
    g.lats = lats
    g.nx = lons.size
    g.ny = lats.size
    return g


@pytest.fixture
def mask_counterexample():
    # True cells row-major: (j=0,i=1)=east first → tile_id 0; (j=1,i=0)=north → tile_id 1
    return np.array([[False, True], [True, False]], dtype=bool)
