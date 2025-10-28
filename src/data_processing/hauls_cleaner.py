from __future__ import annotations

import numpy as np
import pandas as pd


class HaulDbBuilder:
    def __init__(self, hauls_db: pd.DataFrame, to_fix_hauls: pd.DataFrame):
        self.hauls_db = hauls_db.copy()
        self.to_fix_hauls = to_fix_hauls

    # --- step 1: keep only rows that have at least one long and one lat ---
    @staticmethod
    def keep_rows_with_any_lon_lat(
        df: pd.DataFrame, long_fields: list[str], lat_fields: list[str]
    ) -> pd.DataFrame:
        any_long = df[long_fields].notna().any(axis=1)
        any_lat = df[lat_fields].notna().any(axis=1)
        df = df.loc[any_long & any_lat].copy()
        return df

    # --- primitives ---
    @staticmethod
    def sanitize_numeric(s: pd.Series) -> pd.Series:
        s = (
            s.astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("\u00a0", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def fallback(preferred: pd.Series, alternate: pd.Series) -> pd.Series:
        return preferred.where(preferred.notna(), alternate)

    @staticmethod
    def as_decimal_lat(raw: pd.Series) -> pd.Series:
        t = np.trunc(raw / 100000.0)
        mm = (raw - (100000.0 * t)) / 1000.0
        return t + (mm / 60.0)

    @staticmethod
    def as_decimal_lon(raw: pd.Series) -> pd.Series:
        t = np.trunc(raw / 100000.0)
        mm = (raw - (100000.0 * t)) / 1000.0
        return -(t + (mm / 60.0))

    def _correct_hauls(self, df: pd.DataFrame) -> pd.DataFrame:
        blocklist = self.to_fix_hauls[self.to_fix_hauls["lon_corrected"].isna()][
            "haul_id"
        ]
        whitelist = self.to_fix_hauls[~self.to_fix_hauls["lon_corrected"].isna()]

        # Remove
        df = df[~df["Idlance"].isin(blocklist)]
        # Correct hauls
        df["lon"] = (
            df["Idlance"]
            .map(whitelist.set_index("haul_id")["lon_corrected"])
            .fillna(df["lon"])
        )
        return df

    # --- NEW: sanitize `dia` into dd/mm/YYYY, drop invalid, and return failing rows ---
    @staticmethod
    def _sanitize_dia(
        df: pd.DataFrame, dia_col: str = "dia", out_col: str = "date"
    ) -> pd.DataFrame:
        s = df[dia_col].astype(str).str.strip()

        # Capture the first date token allowing 1–2 digits for day/month
        # e.g. "5/8/1999", "05/08/1999", "5/8/1999 0:00:00"
        pat = r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})"
        parts = s.str.extract(pat)

        valid = parts.notna().all(axis=1)

        # Keep only valid rows and build zero-padded dd/mm/YYYY
        df = df.loc[valid].copy()
        parts = parts.loc[valid]
        day = parts[0].astype(str).str.zfill(2)
        mon = parts[1].astype(str).str.zfill(2)
        year = parts[2].astype(str)

        df[out_col] = (day + "/" + mon + "/" + year).values
        return df

    @staticmethod
    def _result_view(df: pd.DataFrame, columns_map: dict[str, str]) -> pd.DataFrame:
        out = pd.DataFrame(
            {out_name: df[src_name] for out_name, src_name in columns_map.items()}
        )
        return out

    # --- step 2+3: pick start_* else end_* and convert to EPSG:4326 ---
    def _build_lon_lat(
        self,
        df: pd.DataFrame,
        start_long: str,
        start_lat: str,
        end_long: str,
        end_lat: str,
        out_lon: str,
        out_lat: str,
    ) -> pd.DataFrame:
        lon_raw = self.fallback(df[start_long], df[end_long])
        lat_raw = self.fallback(df[start_lat], df[end_lat])

        lon_raw = self.sanitize_numeric(lon_raw)
        lat_raw = self.sanitize_numeric(lat_raw)

        df[out_lon] = self.as_decimal_lon(lon_raw)
        df[out_lat] = self.as_decimal_lat(lat_raw)

        if df[out_lon].isna().any() or df[out_lat].isna().any():
            raise ValueError("Failed to compute lon/lat for some rows (NaNs present).")
        return df

    def run(
        self,
        columns_map: dict[str, str],
        start_long: str = "LON inicio",
        start_lat: str = "LAT inicio",
        end_long: str = "LON final",
        end_lat: str = "LAT final",
        out_lon: str = "lon",
        out_lat: str = "lat",
    ) -> pd.DataFrame:
        # 0) Eliminate duplicates
        df = self.hauls_db.drop_duplicates(
            subset=["Idlance"], keep="first"
        ).reset_index(drop=True)

        # 1) only keep rows with at least one long and one lat
        df = self.keep_rows_with_any_lon_lat(
            df,
            long_fields=[start_long, end_long],
            lat_fields=[start_lat, end_lat],
        )
        # 2) build lon/lat in EPSG:4326
        df = self._build_lon_lat(
            df, start_long, start_lat, end_long, end_lat, out_lon, out_lat
        )

        # 3) select columns and correct hauls
        df = df[["Idlance", "dia", "lon", "lat", "PROFMax", "PROFMin"]]

        # 4) Remove and correct hauls
        df = self._correct_hauls(df)

        # 5) Calculate max depth per haul
        df["depth"] = df[["PROFMax", "PROFMin"]].max(axis=1)
        df = df.loc[~df["depth"].isna()]

        # 6) Sanitize `dia` into dd/mm/YYYY, drop invalid
        df = self._sanitize_dia(df)

        # 7) Return a view of final dataframe with selected columns
        df = self._result_view(df, columns_map)

        return df
