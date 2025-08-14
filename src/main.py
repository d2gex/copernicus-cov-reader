from src import coarse_bb, config

if __name__ == "__main__":
    c_bb = coarse_bb.HaulsCleaner(config.INPUT_PATH / "clean_db_tallas.csv")
    c_bb.load()
    unique_hauls_df = c_bb.run(
        columns=["Idlance", "lon", "lat", "dia", "largada_time", "virada_time"]
    )
    print(unique_hauls_df.head(3))
