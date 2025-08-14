from src import bb_calculator, config, hauls_cleaner

if __name__ == "__main__":
    c_bb = hauls_cleaner.HaulsCleaner(config.INPUT_PATH / "clean_db_tallas.csv")
    c_bb.load()
    unique_hauls_df = c_bb.run(
        columns=["Idlance", "lon", "lat", "dia", "largada_time", "virada_time"]
    )
    bbox_cal = bb_calculator.BoundingBoxCalculator(unique_hauls_df, pad_deg=0.08)
    bbox = bbox_cal.run()
    print(bbox)
