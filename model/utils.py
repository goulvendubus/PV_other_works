"""
@author: Goulven Dubus
@date: 2026-06-15
@version: 1.0

Utils for pv forecasting and monitoring.
Functions with @T.Salola are adapted from pv_forecaster.py by T.Salola, with modifications to fit into the PVPlant class.
"""


##_________________Imports_________________##
from __future__ import annotations  
import datetime

import pandas as pd
from pathlib import Path
import cfgrib
import xarray as xr



##_________________Functions_________________##
def print_full(x: pd.DataFrame):
    """
    @T.Salola: Prints a dataframe without leaving any columns or rows out. 
    Useful for debugging.
    """

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1400)
    pd.set_option('display.float_format', '{:10,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


##________________________Data pre-processing functions________________________________
def load_cams_csv(filepath: str | Path) -> pd.DataFrame:
    """Handles I/O only — reads the raw CAMS file."""
    col_names = None
    with open(filepath) as f:
        for line in f:
            if line.startswith("# Observation period"):
                col_names = line.strip().lstrip("# ").split(";")
                break

    return pd.read_csv(
        filepath,
        sep=";",
        comment="#",
        names=col_names,
        index_col=0,
        parse_dates=True,
    )

def preprocess_cams_df(df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """Handles transformation only — no I/O dependency."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index)
    if output_path is not None:
        write_df_to_csv(df, output_path, index=True, sep=";")
    return df


def load_and_preprocess_cams_csv(
    filepath: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Convenience wrapper — composes the two above."""
    return preprocess_cams_df(load_cams_csv(filepath), output_path=output_path)

def dist(lat: float, long: float, lat2: float, lon2: float) -> float:
    """
    Calculates the distance in kilometers between two points on Earth specified by their latitude and longitude.
    Uses the Haversine formula to compute the great-circle distance.
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in kilometers

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def compute_solar_zenith_and_dni(lat: float, lon: float, Ps: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the solar zenith angle and direct normal irradiance (DNI) for given
    latitude, longitude, times, and solar constant Ps.
 
    Handles both tz-aware and tz-naive inputs by localising to UTC when needed.
 
    Returns a DataFrame indexed by the (UTC) timestamps with columns:
        'Dates'           – copy of the index
        'apparent_zenith' – solar zenith angle in degrees
        'computed_dni'    – DNI estimated via the simple Kasten formula (W/m²)
    """
    import pvlib
    import numpy as np
 
    # Ensure times is a UTC-aware DatetimeIndex so pvlib is happy
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
 
    df["Dates"] = df.index
 
    solar_pos       = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    apparent_zenith = solar_pos["apparent_zenith"].values   # degrees
 
    computed_dni = np.where(apparent_zenith < 90, (df["FMI - GHI"] - df["FMI - DHI"]) / np.cos(np.radians(apparent_zenith)), 0.0)
 
    df["apparent_zenith"] = apparent_zenith
    df["computed_dni"]    = computed_dni
    return df

def align_to_common_day(df, ref_date="2021-01-01"):
    df = df.copy()
    ref_date = pd.Timestamp(ref_date)

    # Use Dates column if it exists, otherwise fall back to index
    if "Dates" in df.columns:
        time_series = df["Dates"]
    else:
        time_series = df.index

    df["aligned_time"] = pd.to_datetime(time_series).map(
        lambda ts: pd.Timestamp(
            year=ref_date.year,
            month=ref_date.month,
            day=ref_date.day,
            hour=ts.hour,
            minute=ts.minute,
            second=ts.second
        )
    )

    return df


##_________________csv utility functions____________________

def round_csv_data(filepath:str, decimals: int, columns:list[str]):
    import pandas as pd
    df = pd.read_csv(filepath)
    for col in columns:
        df[col] = df[col].round(decimals)
    df.to_csv(filepath, index=False)

def grib_to_csv(input_path, output_path=None) -> pd.DataFrame:
    """
    Used to create a pandas DataFrame from as .grib file
    """
    ds = cfgrib.open_dataset(input_path)
    df = ds.to_dataframe().reset_index()
    if output_path:
        df.to_csv(output_path, index=False)
    return df

def netcdf_to_csv(input_path, output_path=None) -> pd.DataFrame:
    """
    Used to create a pandas DataFrame from as .netcdf file
    """
    ds = xr.open_dataset(input_path)
    df = ds.to_dataframe().reset_index()
    if output_path:
        df.to_csv(output_path, index=False)
    return df

def csv_transform(source: str | pd.DataFrame, output_path=None) -> pd.DataFrame:
    """
    Used to create a pandas DataFrame from as .csv file
    """
    df = pd.read_csv(source,parse_dates= ["Dates"]) if isinstance(source, (str, Path)) else source.copy()
    # ... transform logic
    if output_path:
        df.to_csv(output_path, index=False)
    return df

def write_df_to_csv(df: pd.DataFrame, output_path: str | Path, *, index: bool = False, sep: str = ",",
                    encoding: str = "utf-8", overwrite: bool = True ) -> Path:
    output_path = Path(output_path)
    if not overwrite and output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index, sep=sep, encoding=encoding)
    return output_path


def format_df_for_pvfc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats a DataFrame to be compatible with the PVForecasting class.
 
    Renames FMI station columns to the names expected by the forecaster,
    clamps all irradiance (and POA, if present) columns to >= 0, and
    returns the result.  The function is intentionally permissive about
    missing columns so that it works for stations that lack G_POA (e.g.
    Kuopio).
 
    Requires a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
 
    df = df.copy()
    df = df.rename(columns={
        "FMI - GHI":         "ghi",
        "FMI - DNI":         "dni",
        "FMI - DHI":         "dhi",
        "FMI - MODULE_TEMP": "module_temp",
        "FMI - G_POA":       "poa_ref_cor",
    })
 
    # Clamp irradiance columns to zero; skip any that are absent (e.g. poa_ref_cor
    # for stations without a tilted pyranometer).
    for col in ["ghi", "dni", "dhi", "poa_ref_cor"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
 
    return df





