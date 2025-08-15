from matplotlib import pyplot as plt

from src import bb_calculator, config, hauls_cleaner
from src.plotters import coarse_bb_plotter as plotter

if __name__ == "__main__":
    c_bb = hauls_cleaner.HaulsCleaner(config.INPUT_PATH / "clean_db_tallas.csv")

    c_bb.load()
    unique_hauls_df = c_bb.run(
        columns=["Idlance", "lon", "lat", "dia", "largada_time", "virada_time"]
    )
    bbox_cal = bb_calculator.BoundingBoxCalculator(unique_hauls_df, pad_deg=0.08)
    bbox = bbox_cal.run()
    hauls_plotter = plotter.CoastHaulsPlotter(
        unique_hauls_df, bbox, config.INPUT_PATH / "galicia_coast_4326.shp"
    )
    hauls_plotter.plot()
    plt.show()
