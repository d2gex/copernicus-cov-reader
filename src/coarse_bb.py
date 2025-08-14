from typing import List, Optional

import pandas as pd


class HaulsCleaner:
    """
    Load and clean haul data from a CSV file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        with open(self.file_path, "r") as f:
            self.df = pd.read_csv(f)

    def run(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        sel_columns = self.df.columns if columns is None else columns
        return (
            (self.df[sel_columns])
            .drop_duplicates(subset="Idlance", keep="first")
            .reset_index(drop=True)
        )
