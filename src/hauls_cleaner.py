from typing import List, Optional

import pandas as pd


class HaulsCleaner:
    """
    Load and clean haul data from a CSV file.
    """

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def run(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        sel_columns = self.df.columns if columns is None else columns
        return (
            (self.df[sel_columns])
            .drop_duplicates(subset="Idlance", keep="first")
            .reset_index(drop=True)
        )
