from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src import bb_calculator, config, hauls_cleaner
from tests.e2e.ete_plotting import plot_solution_map
from tests.e2e.ete_utils import (
    Runtime,
    build_runtime_from_datasets,
    load_layers,
    sample_hauls,
    validate_sample,
)


def run(
    dynamic_path: str,
    static_path: str,
    hauls_df: pd.DataFrame,
    mask_name: str,
    *,
    sample_n: int,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float], Runtime]:
    """
    1) load layers
    2) build Runtime (TileCatalog + KDIndex + lat0)
    3) sample hauls
    4) validate KD vs brute-force

    Returns (details_df, summary_pct, runtime) so you can debug step-by-step.
    """
    dyn, sta = load_layers(dynamic_path, static_path)
    rt = build_runtime_from_datasets(dyn, sta, mask_name)
    hauls_sample = sample_hauls(hauls_df, n=sample_n, seed=seed)
    details, summary = validate_sample(hauls_sample, rt)
    return details, summary, rt


# ---------- Small reporting helpers (pure/side-effect free except prints) ----------


def mismatches(details: pd.DataFrame) -> pd.DataFrame:
    """Rows where KD != brute-force (neighbors + non-neighbors)."""
    return details[details["match_type"] != "same"].copy()


def print_summary(summary: Dict[str, float]) -> None:
    same = summary["same_pct"]
    nb = summary["neighbor_pct"]
    non = summary["non_neighbor_pct"]
    total = same + nb + non
    print(f"Match summary (% of sampled hauls) [total={total:.1f}%]:")
    print(f"  same       : {same:.2f}%")
    print(f"  neighbor   : {nb:.2f}%")
    print(f"  non-neighbor: {non:.2f}%")


def print_mismatches(details: pd.DataFrame, *, limit: int = 50) -> None:
    """
    Prints KD vs reference for mismatched rows, including neighbor direction and
    both tiles' centers (lon/lat).
    """
    mm = mismatches(details)
    if mm.empty:
        print("No mismatches 🎉")
        return
    cols = [
        "haul_index",
        "haul_lon",
        "haul_lat",
        "kd_tile_id",
        "kd_tile_lon",
        "kd_tile_lat",
        "ref_tile_id",
        "ref_tile_lon",
        "ref_tile_lat",
        "match_type",
        "neighbor_direction",
    ]
    head = mm.loc[:, cols]
    if len(head) > limit:
        head = head.iloc[:limit]
        print(f"Showing first {limit} of {len(mm)} mismatches:")
    else:
        print(f"Mismatches: {len(mm)}")
    # Compact numeric display
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(head.to_string(index=False))


# ---------- Optional convenience entrypoint (keeps I/O explicit) ----------


def main(
    dynamic_path: str,
    static_path: str,
    hauls_df: pd.DataFrame,
    mask_name: str,
    *,
    sample_n: int,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float], Runtime]:
    """
    Convenience wrapper around run() that also prints a short report.
    Returns the same tuple as run().
    """
    details, summary, rt = run(
        dynamic_path, static_path, hauls_df, mask_name, sample_n=sample_n, seed=seed
    )
    print_summary(summary)
    print_mismatches(details)
    return details, summary, rt


if __name__ == "__main__":
    # with open(config.INPUT_PATH / "tallas_2023_nueva.csv", "r") as f:
    df = pd.read_csv(
        config.INPUT_PATH / "clean_db_tallas.csv", encoding_errors="ignore"
    )

    c_bb = hauls_cleaner.HaulsCleaner(df)

    unique_hauls_df = c_bb.run(
        columns=["Idlance", "lon", "lat", "dia", "largada_time", "virada_time"]
    )
    bbox_cal = bb_calculator.BoundingBoxCalculator(unique_hauls_df, pad_deg=0.08)
    bbox = bbox_cal.run()

    details, summary, rt = main(
        config.OUTPUT_PATH
        / "multiple_depth_layer"
        / "multiple_depth_layer_2020-01-01_2020-01-31.nc",
        config.OUTPUT_PATH / "multiple_depth_layer" / "multiple_depth_static_layer.nc",
        unique_hauls_df,
        "mask_thetao",
        sample_n=50,
        seed=None,
    )

    # # Or individually:
    fig, ax = plot_solution_map(details, rt.catalog, kind="kd")  # KDIndex map
    fig2, ax2 = plot_solution_map(details, rt.catalog, kind="ref")  # Brute-force map
    plt.show()
