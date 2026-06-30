from matplotlib import dates

from model.cache_utils import *
from model.utils import *
from datetime import datetime, timedelta
import pandas as pd
import io
import os
import cdsapi
import xarray as xr
import cfgrib
from pathlib import Path
from typing import List, Tuple


# Canonical station name -> data folder slug, shared by all three download functions
# so each station's FMI, CAMS radiation, and CAMS environmental files end up
# together under data/<slug>/
STATION_SLUGS = {
    "Helsinki Kumpula": "helsinki",
    "Kuopio Savilahti": "kuopio",
    "Sodenkylä Tähtelä": "sodankyla",
    "Parainen Utö": "uto",
}


def _resolve_station_dir(station: str, base_path: str = "") -> str:
    """Resolves the data/<slug> directory for a station, creating it if needed."""
    import os

    slug = None
    for name, candidate_slug in STATION_SLUGS.items():
        if name.lower() in station.lower() or station.lower() == candidate_slug:
            slug = candidate_slug
            break
    if slug is None:
        slug = station.lower().replace(" ", "_")

    if base_path == "":
        base_path = os.path.join(os.getcwd(), "data")

    station_dir = os.path.join(base_path, slug)
    os.makedirs(station_dir, exist_ok=True)
    return station_dir

def _parse_cams_csv_body(data_str: str) -> pd.DataFrame:
    """
    Parses the data body of a CAMS csv_expert/csv export.

    CAMS writes no real header row for the data section (only commented '#' metadata
    lines above it, which are stripped before this is called), and its first column
    is a single field containing a "start/end" period rather than one timestamp, e.g.:
        2025-01-01T00:00:00.0/2025-01-01T00:01:00.0;0.0000;0.0000;...

    Reading this with header=0 makes pandas treat the first data row as column names
    (producing bogus columns like "0.0000.1") and trying to parse the whole period
    string as a single date produces NaT for nearly every row. This instead reads
    with header=None and splits the period column into period_start/period_end.
    """
    if not data_str.strip():
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(data_str), sep=";", header=None)

    period = df[0].astype(str).str.split("/", n=1, expand=True)
    period.columns = ["period_start", "period_end"]
    period["period_start"] = pd.to_datetime(period["period_start"], errors="coerce", utc=True)
    period["period_end"] = pd.to_datetime(period["period_end"], errors="coerce", utc=True)

    df = pd.concat([period, df.drop(columns=[0])], axis=1)
    df = df.dropna(subset=["period_start"])
    return df

def find_contiguous_day_ranges(days: List[datetime.date]) -> List[Tuple[datetime.date, datetime.date]]:
    days = sorted(days)
    ranges = []
    range_start = days[0]
    prev = days[0]
    for d in days[1:]:
        if (d - prev).days > 1:
            ranges.append((range_start, prev))
            range_start = d
        prev = d
    ranges.append((range_start, prev))
    return ranges



##_________________data download functions____________________
def download_cams_solar_radiation_data_v2(station:str, start: Tuple[int, int, int], end: Tuple[int, int, int], timestep: str = "1minute", time_reference: str = "universal_time",
                                       format: str = "csv_expert", path: str = "")-> Path:
    """
    Allows to fetch data from the ECMWF data store.
    To do so, it is necessary to have created the file ".cdsapirc" : "cd ~/.cdsapirc" containing:
        url: https://ads.atmosphere.copernicus.eu/api
        key: 8d4605f4-f9e8-4292-9408-076625134bd8
    and then in virtual python environment: 
        $ pip install "cdsapi>=0.7.7"
    --------------
    Parameters
        station: str, name of the station for which to fetch data
        start: tuple (yyyy,m,d), start date of the data to fetch
        end: tuple (yyyy,m,d), end date of the data to fetch
        timestep: str, time step of the data, in ["1minute", "15minute", "1hour", "1day", "1month"] (default: "1minute")
        time_reference: str, time reference of the data (default: "universal_time")
        format: str, format of the data (default: "csv_expert")
        path: str, path to save the data (default: "")
    --------------
    """
    #--- Imports ---
    import cdsapi
    import os
    import io
    import tempfile

 
    #--- Parameters---
    plant_dict = {"Helsinki": (24.961305, 60.203071, 47),
                    "Kuopio": (27.633311, 62.89256, 87),
                    "Sodankyla": (26.650, 67.367, 184),
                    "Parainen Uto": (21.21297, 59.46554, 10)}
    
    if station in plant_dict:
        location = plant_dict[station] 
    else:
        raise ValueError(f"Unknown station: {station}")

    cams_columns = ["Dates",
                    "TOA", "CAMS - clear_sky_GHI", "CAMS - clear_sky_BHI", "CAMS - clear_sky_DHI", "CAMS - clear_sky_BNI", "CAMS - GHI", "CAMS - BHI", "CAMS - DHI", "CAMS - BNI", "CAMS - reliability", "CAMS - sza", "CAMS - summer_winter_split", "CAMS - tco3",
    "CAMS - tcwv",
    "CAMS - AOD_BC",
    "CAMS - AOD_DU",
    "CAMS - AOD_SS",
    "CAMS - AOD_OR",
    "CAMS - AOD_SU",
    "CAMS - AOD_NI",
    "CAMS - AOD_AM",
    "CAMS - AOD_SO",
    "CAMS - alpha",
    "CAMS - snow_probability",
    "CAMS - fiso",
    "CAMS - fvol",
    "CAMS - fgeo",
    "CAMS - albedo",
    "CAMS - cloud_optical_depth",
    "CAMS - cloud_coverage",
    "CAMS - cloud_type",
    "CAMS - GHI_no_corr",
    "CAMS - BHI_no_corr",
    "CAMS - DHI_no_corr",
    "CAMS - BNI_no_corr"]
    drop_columns = ["CAMS - summer_winter_split",
                        "CAMS - tco3",
                        "CAMS - tcwv",
                    "CAMS - AOD_BC",
                    "CAMS - AOD_DU",
                    "CAMS - AOD_SS",
                    "CAMS - AOD_OR",
                    "CAMS - AOD_SU",
                    "CAMS - AOD_NI",
                    "CAMS - AOD_AM",
                    "CAMS - AOD_SO",
                    "CAMS - alpha",
                    "CAMS - snow_probability",
                    "CAMS - fiso",
                    "CAMS - fvol",
                    "CAMS - fgeo",
                    "CAMS - cloud_optical_depth",
                    "CAMS - cloud_type",
                    "CAMS - GHI_no_corr",
                    "CAMS - BHI_no_corr",
                    "CAMS - DHI_no_corr",
                    "CAMS - BNI_no_corr",
                ]
    
    #--- File path ---
    station_dir = _resolve_station_dir(station, path)
    filename = os.path.join(station_dir, f"cams_radiation_data.csv")
    file_exists = os.path.exists(filename)                                      # check if data file already exists for this station

    start_date = datetime(start[0], start[1], start[2])
    end_date = datetime(end[0], end[1], end[2])

    
    #--- Load existing data if file exists, and find header/comment block ---
    loaded_data = None
    covered_days = set()

    if file_exists:
        print(f"Found existing file: {filename}")
        df_existing = pd.read_csv(filename, sep=",", comment="#", header=0)
        covered_days = {pd.Timestamp(t).date() for t in df_existing["Dates"].values}
        loaded_data = df_existing


    # --- Determine missing days ---
    full_days = pd.date_range(start=start_date, end=end_date, freq="D").date
    missing_days = [d for d in full_days if d not in covered_days]

    if not missing_days:
        print("No missing data — file is already up to date.")
        return Path(filename)
    

    #--- Fetch missing data from API ---
    client = cdsapi.Client()
    dataset = "cams-solar-radiation-timeseries"

    for range_start, range_end in find_contiguous_day_ranges(missing_days):
        print(f"Fetching {range_start} → {range_end} at {location}...")
        request = {
            "sky_type": "observed_cloud",
            "location": {"longitude": location[0], "latitude": location[1]},
            "altitude": location[2],
            "date": f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            "time_step": "1minute",
            "time_reference": "universal_time",
            "data_format": "csv_expert"
            }
        result = client.retrieve(dataset, request)
    
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
            tmp_path = tmp.name
        try:
            result.download(tmp_path)
            with open(tmp_path, "r") as f:
                lines = [line for line in f if not line.startswith("#")]
            df = _parse_cams_csv_body("".join(lines))
            df = df.drop(columns=["period_end"])
            df.columns = cams_columns
            df = df.drop(drop_columns, axis=1)
            loaded_data = pd.concat([loaded_data, df], ignore_index=True).drop_duplicates(subset="Dates").sort_values("Dates").reset_index(drop=True) if loaded_data is not None else df
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
        with open(filename, "w") as f:
            write_df = loaded_data.copy()
            write_df.to_csv(f, sep=",", mode="a", index=False, header=True)

    print(f"Updated: {filename} ({len(missing_days)} new days added, {len(loaded_data)} total rows)")
    return filename



def  download_cams_environmental_data(
    station: str,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    time: str = "00:00",
    leadtime_hour: List[str] | None = None,
    data_format: str = "netcdf",
    path: str = "",
    parameters: List[str] | None = None,
) -> str:
    """
    Download CAMS environmental forecast data with incremental updates.
    """
 
    import json
 
    import cdsapi
    import xarray as xr
 
    # -----------------------------
    # Defaults
    # -----------------------------
    if leadtime_hour is None:
        leadtime_hour = ["0"]
 
    if parameters is None:
        parameters = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ]
 
    # -----------------------------
    # Validation
    # -----------------------------
    if time not in {"00:00", "12:00"}:
        raise ValueError("time must be '00:00' or '12:00'")
 
    if not all(h.isdigit() and 0 <= int(h) <= 120 for h in leadtime_hour):
        raise ValueError("leadtime_hour must be between 0–120")
 
    if data_format not in {"netcdf", "grib"}:
        raise ValueError("data_format must be 'netcdf' or 'grib'")
 
    requested_leadtimes = {int(h) for h in leadtime_hour}
 
    # -----------------------------
    # Station → location mapping
    # -----------------------------
    station_locations = {
        "helsinki": (24.961305, 60.203071, 47),
        "kuopio": (27.633311, 62.89256, 87),
        "sodankyla": (26.650, 67.367, 184),
        "uto": (21.21297, 59.46554, 10),
    }
 
    # Resolve slug using your shared mapping
    slug = None
    station_lower = station.lower()
 
    for name, s in STATION_SLUGS.items():
        if name.lower() in station_lower or s == station_lower:
            slug = s
            break
 
    if slug is None:
        raise ValueError(f"Unknown station: {station}")
 
    if slug not in station_locations:
        raise ValueError(f"No coordinates for station slug: {slug}")
 
    location = station_locations[slug]
 
    # -----------------------------
    # File path
    # -----------------------------
    station_dir = _resolve_station_dir(station, path)
 
    ext = "nc" if data_format == "netcdf" else "grib"
    filename = os.path.join(station_dir, f"cams_env_data.{ext}")
 
    # Sidecar file tracking, per calendar day, which leadtime hours have
    # actually been fetched into `filename`. This is what lets us tell
    # "day exists" apart from "day exists with the leadtime hours we need".
    coverage_file = filename + ".coverage.json"
 
    start_date = pd.Timestamp(*start).date()
    end_date = pd.Timestamp(*end).date()
 
    # -----------------------------
    # Existing dataset + coverage log
    # -----------------------------
    existing_ds = None
    coverage: dict = {}
 
    if os.path.exists(coverage_file):
        with open(coverage_file) as f:
            coverage = json.load(f)
 
    if os.path.exists(filename):
        if data_format == "grib":
            raise ValueError("GRIB incremental updates are not supported.")
 
        print(f"Found existing file: {filename}")
 
        existing_ds = xr.load_dataset(filename)
 
        time_dim = (
            "forecast_reference_time"
            if "forecast_reference_time" in existing_ds.coords
            else "time"
        )
 
    # A day only counts as "covered" if the coverage log shows every
    # leadtime hour we're asking for now was already fetched for it.
    # (If you're upgrading from a version without this log, every day will
    # look uncovered on the first run and get refetched once — after that,
    # the log keeps things accurate and avoids redundant downloads.)
    covered_days = {
        pd.Timestamp(day_str).date()
        for day_str, hours in coverage.items()
        if requested_leadtimes.issubset(set(hours))
    }
 
    # -----------------------------
    # Missing days
    # -----------------------------
    full_days = pd.date_range(start=start_date, end=end_date, freq="D").date
 
    missing_days = [d for d in full_days if d not in covered_days]
 
    if not missing_days:
        print("No missing data.")
        return filename
 
    # Drop stale rows for any day we're about to refetch (e.g. a day that
    # previously only had a subset of leadtime hours), so the freshly
    # fetched, complete data replaces them instead of losing to
    # drop_duplicates further down.
    if existing_ds is not None:
        missing_set = set(missing_days)
        keep_mask = [
            pd.Timestamp(t).date() not in missing_set
            for t in existing_ds[time_dim].values
        ]
        existing_ds = existing_ds.isel({time_dim: keep_mask})
 
    ranges = find_contiguous_day_ranges(missing_days)
 
    print(f"Fetching {len(ranges)} missing range(s)...")
 
    # -----------------------------
    # Download
    # -----------------------------
    client = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels-timeseries"
 
    new_datasets = []
 
    for r_start, r_end in ranges:
        print(f"  Fetching {r_start} → {r_end}")
 
        request = {"variable": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
                   "location": {"longitude": 20.06421, "latitude": 64.00876},
                   "date": ["2024-01-01/2024-07-31"],
                   "data_format": "csv"
            }
 
        tmp_file = os.path.join(
            station_dir, f"_tmp_{r_start}_{r_end}.{ext}"
        )
 
        client.retrieve(dataset, request).download(tmp_file)
 
        ds = xr.load_dataset(tmp_file)
        os.remove(tmp_file)
 
        new_datasets.append(ds)
 
    if not new_datasets:
        print("No new data returned from API.")
        return filename
 
    # -----------------------------
    # Merge
    # -----------------------------
    combined_new = (
        xr.concat(new_datasets, dim="forecast_reference_time")
        if len(new_datasets) > 1
        else new_datasets[0]
    )
 
    time_dim = (
        "forecast_reference_time"
        if "forecast_reference_time" in combined_new.coords
        else "time"
    )
 
    if existing_ds is not None:
        combined_ds = xr.concat([existing_ds, combined_new], dim=time_dim)
        combined_ds = combined_ds.drop_duplicates(dim=time_dim).sortby(time_dim)
        existing_ds.close()
    else:
        combined_ds = combined_new
 
    # -----------------------------
    # Save NetCDF
    # -----------------------------
    tmp_final = filename + ".tmp"
 
    combined_ds.to_netcdf(tmp_final)
    combined_ds.close()
 
    os.replace(tmp_final, filename)
 
    # -----------------------------
    # Update coverage log
    # -----------------------------
    for d in missing_days:
        day_str = d.isoformat()
        coverage[day_str] = sorted(set(coverage.get(day_str, [])) | requested_leadtimes)
 
    with open(coverage_file, "w") as f:
        json.dump(coverage, f, indent=2)
 
    # -----------------------------
    # Convert → CSV
    # -----------------------------
    csv_file = filename.replace(".nc", ".csv")
 
    df = netcdf_to_csv(filename, csv_file)
 

    df["Dates"] = pd.to_datetime(df["valid_time"], errors="coerce", utc=True)
    df["Dates"] = df["Dates"].dt.tz_convert(None)  # Drop timezone info for consistency with other files
 
    drop_cols = [
        c for c in ("forecast_reference_time", "forecast_period", "valid_time")
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)
 
    # Keep Dates as the first column, matching the previous layout.
    df = df[["Dates"] + [c for c in df.columns if c != "Dates"]]
 
    df.to_csv(csv_file, index=False)
 
    os.remove(filename)
 
    print(f"Updated: {csv_file}")
 
    return csv_file





def download_fmi_station_data(station: str, start: tuple | pd.Timestamp, end: tuple | pd.Timestamp = None,
                        parameters = ("fmisid,"
                                        "stationname,"
                                        "utctime,"
                                        "GLOB_PT1M_AVG,"
                                        "DIFF_PT1M_AVG,"
                                        "DIR_PT1M_AVG,"
                                        "GLOBA_PT1M_AVG(:31),"
                                        "TTECH_PT1M_AVG(:31),"
                                        "TTECH_PT1M_AVG(:32),"
                                        "TTECH_PT1M_AVG(:33),"
                                        "TA_PT1M_AVG,"
                                        "RH_PT1M_AVG,"
                                        "WS_PT10M_AVG,"
                                        "WD_PT10M_AVG,"
                                        "SND_PT1M_INSTANT"),
                        path = "", format: str = "csv"):
    """
    Fetches radiation, temperature and wind data for the required PV station and dates.
    Maintains one persistent file per station and only fetches data for missing date ranges.
 
    For each missing time range, this:
      1. Downloads the FMI station data for that range (GHI, DHI, DNI sensor reading, etc).
      2. Downloads the solar zenith angle for that same range via pvlib.
      3. Computes DNI from GHI, DHI and zenith using pvlib.irradiance.dni, and uses it
         to fill in the sensor's DNI column wherever the sensor reading is missing
         (this is the normal case for stations like Kuopio, which has no DNI sensor
         at all and reports DIR_PT1M_AVG as empty for every row).
 
    This computation only ever runs on freshly fetched chunks, so previously cached
    rows already on disk are left as-is and are not recomputed on every call.
 
    Returns the path to the station's data file.
    """
    import os
    import io
    import requests
    from model.utils import compute_solar_zenith_and_dni
 
    station_ids = {
        "Helsinki": "101004",
        "Turku":    "100949",
        "Kuopio":   "101586",
        "Sodankyla":"101932",
        "Uto":      "100908"
    }
 
    Ps = 1.353e3  # W/m², solar constant at 1 AU
 
    # (latitude, longitude) used for pvlib solar-position computation
    station_coords = {
        "Helsinki":  (60.2058, 24.9610),
        "Turku":     (60.5157, 22.2681),
        "Kuopio":    (62.8921, 27.6325),
        "Sodankyla": (67.3670, 26.6331),
        "Uto":       (59.7804, 21.3696),
    }
 
    station_id   = station_ids.get(station, station)
    station_name = next((name for name, cid in station_ids.items() if cid == station_id), station)
 
    if isinstance(start, pd.Timestamp):
        start = start.to_pydatetime()
    elif isinstance(start, tuple):
        start = datetime(start[0], start[1], start[2])
 
    if isinstance(end, pd.Timestamp):
        end = end.to_pydatetime()
    elif isinstance(end, tuple):
        end = datetime(end[0], end[1], end[2])
    elif end is None:
        end = start + timedelta(days=1)
 
    station_dir = _resolve_station_dir(station_name, path)
 
    format = format.lower()
    if format not in ["txt", "csv"]:
        raise ValueError("format must be either 'txt' or 'csv'")
 
    filename = os.path.join(station_dir, f"fmi_station_data.{format}")
 
    col_names = [
        "fmisid", "stationname", "Dates",
        "FMI - GHI", "FMI - DHI", "FMI - DNI",
        "FMI - G_POA", "FMI - ROOF_TEMP", "FMI - NE_MODULE_TEMP",
        "FMI - SW_MODULE_TEMP", "FMI - 2M_AIR_TEMP", "FMI - AIR_HUM",
        "FMI - 10M_WS", "FMI - 10M_WD", "FMI - SNOW_DEPTH"
    ]
 
    # ------------------------------------------------------------------ #
    #  Load existing data                                                  #
    # ------------------------------------------------------------------ #
    existing_df    = None
    existing_dates = set()
 
    if os.path.exists(filename):
        print(f"Found existing file: {filename}")
        if format == "csv":
            existing_df    = pd.read_csv(filename, parse_dates=["Dates"])
            existing_dates = set(existing_df["Dates"])
        else:
            existing_df = pd.read_csv(
                filename, sep=";", names=col_names, comment="#", header=None
            )
            existing_df["Dates"] = pd.to_datetime(existing_df["Dates"], format="%Y%m%dT%H%M%S")
            existing_dates = set(existing_df["Dates"])
 
    # ------------------------------------------------------------------ #
    #  Decide what needs to happen                                       #
    # ------------------------------------------------------------------ #
    full_index         = pd.date_range(start=start, end=end, freq="1min")
    missing_timestamps = [ts for ts in full_index if ts not in existing_dates]
 
    # DNI fill is needed if Ps was provided and the existing file still has
    # NaN DNI (e.g. it was cached before Ps support was added).
    dni_needs_fill = (
        Ps is not None
        and station_name in station_coords
        and existing_df is not None
        and "FMI - DNI" in existing_df.columns
        and existing_df["FMI - DNI"].isna().any()
    )
 
    if not missing_timestamps and not dni_needs_fill:
        # Nothing to fetch and DNI is already filled — truly up to date.
        print("No missing data — file is already up to date.")
        return filename
 
    # ------------------------------------------------------------------ #
    #  Fetch missing ranges (only if there are any)                        #
    # ------------------------------------------------------------------ #
    new_df = pd.DataFrame()   # stays empty when only a DNI fill is needed
 
    if missing_timestamps:
        missing_ranges = find_contiguous_day_ranges(sorted(missing_timestamps))
        print(f"Fetching {len(missing_ranges)} missing range(s) for {station_name}...")
 
        base_url  = "http://smartmet.fmi.fi/timeseries"
        producer  = "observations_fmi"
        new_frames = []
 
        for range_start, range_end in missing_ranges:
            print(f"  Fetching {range_start} → {range_end}")
 
            # Build URL as a plain string so commas and (:31)-style sensor
            # suffixes in `parameters` are NOT URL-encoded by requests.
            url = (
                f"{base_url}?"
                f"producer={producer}"
                f"&format=ascii"
                f"&precision=double"
                f"&separator=;"
                f"&starttime={range_start.strftime('%Y%m%dT%H%M%S')}"
                f"&endtime={range_end.strftime('%Y%m%dT%H%M%S')}"
                f"&tz=UTC"
                f"&timestep=1"
                f"&fmisid={station_id}"
                f"&param={parameters}"
            )
 
            response = requests.get(url)
            response.raise_for_status()
 
            if not response.text.strip():
                print(f"  No data returned for {range_start} → {range_end}, skipping.")
                continue
 
            df_new = pd.read_csv(
                io.StringIO(response.text), sep=";", names=col_names, header=None
            )
            df_new["Dates"] = pd.to_datetime(df_new["Dates"], format="%Y%m%dT%H%M%S")
            new_frames.append(df_new)
 
        if new_frames:
            new_df = pd.concat(new_frames, ignore_index=True)
        else:
            print("API returned no new data.")
            if not dni_needs_fill:
                return filename
 
    elif dni_needs_fill:
        print("No new data to fetch — re-opening existing file to apply DNI fill.")
 
    # ------------------------------------------------------------------ #
    #  Merge with existing data, sort, deduplicate                         #
    # ------------------------------------------------------------------ #
    if existing_df is not None:
        if format == "csv" and {"fmisid", "stationname"}.issubset(existing_df.columns):
            existing_df = existing_df.drop(columns=["fmisid", "stationname"])
        combined_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if not new_df.empty
            else existing_df.copy()
        )
    else:
        combined_df = new_df.copy()
 
    combined_df = (
        combined_df
        .drop_duplicates(subset=["Dates"])
        .sort_values("Dates")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ #
    #  Solar zenith computation and DNI fill                               #
    # ------------------------------------------------------------------ #
    if Ps is not None and station_name in station_coords:
        lat, lon = station_coords[station_name]
 
        df_zenith    = compute_solar_zenith_and_dni(lat, lon, Ps, combined_df)
        computed_dni = df_zenith["computed_dni"].values
 
        dni_col = "FMI - DNI"
 
        if not combined_df[dni_col].notna().any():
            # Entirely NaN (e.g. Kuopio): fill every row.
            print(f"  DNI column is entirely NaN — filling all values from solar zenith formula.")
            combined_df[dni_col] = computed_dni
 
        else:
            # Partial gaps: fill only NaN rows where the formula result is
            # within the IQR-based logical range of existing measured values.
            existing_vals = combined_df[dni_col].dropna()
            q1, q3 = existing_vals.quantile(0.25), existing_vals.quantile(0.75)
            iqr    = q3 - q1
            lower  = max(0.0, q1 - 1.5 * iqr)
            upper  = q3 + 1.5 * iqr
 
            mask_nan     = combined_df[dni_col].isna()
            mask_logical = (computed_dni >= lower) & (computed_dni <= upper)
            fill_mask    = mask_nan & mask_logical
 
            n_filled  = int(fill_mask.sum())
            n_skipped = int(mask_nan.sum()) - n_filled
            if n_filled > 0:
                combined_df.loc[fill_mask, dni_col] = computed_dni[fill_mask]
                print(f"  Filled {n_filled} partial DNI gap(s) within logical bounds "
                      f"[{lower:.1f}, {upper:.1f}] W/m².")
            if n_skipped > 0:
                print(f"  Skipped {n_skipped} gap(s) outside logical bounds — left as NaN.")
 
    elif Ps is None:
        print("  Ps not provided — skipping solar zenith DNI fill.")
    else:
        print(f"  No coordinates known for '{station_name}' — skipping solar zenith DNI fill.")
    
    
    # ------------------------------------------------------------------ #
    #  Mean module temperature                                             #
    # ------------------------------------------------------------------ #
    combined_df["FMI - MODULE_TEMP"] = combined_df[
        ["FMI - SW_MODULE_TEMP", "FMI - NE_MODULE_TEMP"]
    ].mean(axis=1, skipna=True)

    # ------------------------------------------------------------------ #
    #  Write back to file                                                  #
    # ------------------------------------------------------------------ #
    header = f"""#
    # FMI's {station_name} solar site meteorological and ancillary PV data
    # contact: Anders.Lindfors@fmi.fi 
    # data protocol: if used in scientific studies, co-authorship shall be offered to FMI 
    # 
    # references: 
    # Böök et al. (2020): https://doi.org/10.1016/j.solener.2020.04.068
    # Böök & Lindfors (2020): https://doi.org/10.1016/j.solener.2020.10.024
    #
    # fmisid: station id
    # stationname: name of station 
    # utctime in format: %Y%m%dT%H%M%S
    #
    # 1 minute values
    #   GLOB_PT1M_AVG [W/m2]:        global radiation on horizontal surface
    #   DIFF_PT1M_AVG [W/m2]:        diffuse radiation on horizontal surface
    #   DIR_PT1M_AVG [W/m2]:         direct normal irradiance (perpendicular surface), 
    #                                not available for Kuopio
    #   GLOBA_PT1M_AVG(:31) [W/m2]:  global radiation on inclined PV plane-of-array
    #   TTECH_PT1M_AVG(:31) [deg C]: air temperature on roof, vicinity of PV arrays
    #   TTECH_PT1M_AVG(:32) [deg C]: PV module temperature, NE corner
    #   TTECH_PT1M_AVG(:32) [deg C]: PV module temperature, SW corner
    #   TA_PT1M_AVG [deg C]:         2m air temperature at closeby weather station
    #   RH_PT1M_AVG [%]:             Relative humidity at closeby weather station
    # 10 minute values: 
    #   WS_PT10M_AVG [m/s]: 10m wind speed at closeby weather station 
    #   WD_PT10M_AVG [deg]: 10m wind direction at closeby weather station 
    #
    # Note: time stamp denotes the end of the observation time window
    #       for both 1 min and 10 min values
    #
    """ + parameters + '\n'
 
    if format == "txt":
        with open(filename, "w") as f:
            f.write(header)
            combined_df.to_csv(f, sep=";", index=False, header=False,
                               date_format="%Y%m%dT%H%M%S")
    else:
        save_df = combined_df.drop(columns=["fmisid", "stationname"], errors="ignore")
        save_df.to_csv(filename, index=False)
 
    n_new = len(new_df)
    print(f"Updated: {filename} ({n_new} new rows added, {len(combined_df)} total rows)")
    return filename





def download_fmi_power_data(station: str, start: tuple | pd.Timestamp, end: tuple | pd.Timestamp = None,
                        path = "", format: str = "csv"):
    """
    Fetches PV power output data for the required station and dates.
    Maintains one persistent file per station and only fetches data for missing date ranges.
    Returns the path to the station's data file.
 
    Data is served as daily files (data_YYYYMMDD.txt) on the havemi.fmi.fi server.
    For Sodankyla, power measurements are split across two inverter subdirectories
    (pv350_11, pv350_12): both are fetched per day and their power columns are summed.
    """
    import os
    import io
    import requests
 
    station_ids = {
        "Helsinki": "101004",
        "Turku": "100949",
        "Kuopio": "101586",
        "Sodankyla": "101932",
        "Uto": "100908"
    }
 
    station_id = station_ids.get(station, station)
    station_slug = STATION_SLUGS.get(station, station.lower().replace(" ", "_"))
    station_name = next((name for name, candidate_id in station_ids.items() if candidate_id == station_id), station)
 
    if isinstance(start, pd.Timestamp):
        start = start.to_pydatetime()
    elif isinstance(start, tuple):
        start = datetime(start[0], start[1], start[2])
 
    if isinstance(end, pd.Timestamp):
        end = end.to_pydatetime()
    elif isinstance(end, tuple):
        end = datetime(end[0], end[1], end[2])
    elif end is None:
        end = start  # Default to a single day if end is not provided
 
    # --- Resolve file path: data/<station_slug>/fmi_power_data.<format> ---
    station_dir = _resolve_station_dir(station_name, path)
 
    format = format.lower()
    if format not in ["txt", "csv"]:
        raise ValueError("format must be either 'txt' or 'csv'")
 
    # Single persistent file per station (fixed name, per the data/ folder architecture)
    filename = os.path.join(station_dir, f"fmi_power_data.{format}")
 
    col_names = ["Dates", "Count",
                 "GridPower[W]", "In12Power[W]", "In1Power[W]", "In2Power[W]",
                 "GridPower_max[W]", "In1Power_max[W]", "In2Power_max[W]",
                 "GridPower_std[W]", "In1Power_std[W]", "In2Power_std[W]"]
 
    # Power columns summed across inverters for Sodankyla
    power_cols = [c for c in col_names if c not in ("Dates", "Count")]
 
    # --- Load existing data if file exists ---
    existing_df = None
    existing_dates = set()
 
    if os.path.exists(filename):
        print(f"Found existing file: {filename}")
        if format == "csv":
            existing_df = pd.read_csv(filename, parse_dates=["Dates"])
        else:  # txt: semicolon-separated, no comment header
            existing_df = pd.read_csv(filename, sep=";", parse_dates=["Dates"])
        existing_dates = set(existing_df["Dates"].dt.date)
 
    # --- Determine which date ranges are missing ---
    # Check at day granularity: each source file covers exactly one calendar day
    full_days = pd.date_range(start=start, end=end, freq="D").date
    missing_days = [d for d in full_days if d not in existing_dates]
 
    if not missing_days:
        print("No missing data — file is already up to date.")
        return filename
 
    missing_ranges = find_contiguous_day_ranges(missing_days)
    print(f"Fetching {len(missing_ranges)} missing range(s) for {station_name}...")
 
    # --- Set up base URL(s) ---
    # Sodankyla power data is split across two inverter subdirectories
    if station_id == "101932":
        base_urls = [
            "http://havemi.fmi.fi/sodankyla/suo/pv350_11/",
            "http://havemi.fmi.fi/sodankyla/suo/pv350_12/",
        ]
    else:
        base_urls = [f"http://havemi.fmi.fi/pvsolar/{station_slug}/data/"]
 
    new_frames = []
 
    for range_start, range_end in missing_ranges:
        print(f"  Fetching {range_start} → {range_end}")
 
        day = range_start
        while day <= range_end:
            str_date = day.strftime("%Y%m%d")
            print("str_date : ", str_date)                      # for debugging
            #fname = f"sofar_{station_slug}_{str_date}.txt"
            #print("file name : ", fname)                      # for debugging
            
            day_frames = []

            for base_url in base_urls:
                response = None

                filenames = [
                    f"sofar_{station_slug}_{str_date}.txt",
                    f"trio20_{station_slug}_{str_date}.txt",
                ]

                for fname in filenames:
                    url = f"{base_url}{fname}"
                    try:
                        r = requests.get(url, timeout=15)
                        print(f"    GET {url} → {r.status_code}")

                        if r.status_code == 200 and r.text.strip():
                            response = r
                            print(f"Using {fname}")
                            break

                    except requests.RequestException as e:
                        print(f"    ERROR for {url}: {e}")

                if response is None:
                    print(f"No valid file for {str_date} at {base_url}")
                    continue

                df_new = pd.read_csv(io.StringIO(response.text), sep=",", parse_dates=["DATE"])
                df_new = df_new.rename(columns={"DATE": "Dates"})
                day_frames.append(df_new)

 
            if day_frames:
                if len(day_frames) == 1:
                    new_frames.append(day_frames[0])
                else:
                    # Sodankyla: align on Dates index, sum power columns across both inverters
                    merged = day_frames[0].set_index("Dates")
                    for df in day_frames[1:]:
                        other = df.set_index("Dates")
                        for col in power_cols:
                            if col in merged.columns and col in other.columns:
                                merged[col] = merged[col].add(other[col], fill_value=0)
                    new_frames.append(merged.reset_index())
 
            day += timedelta(days=1)
 
    if not new_frames:
        print("No new data retrieved.")
        return filename
 
    new_df = pd.concat(new_frames, ignore_index=True)
 
    # --- Merge with existing data, sort, deduplicate ---
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
 
    combined_df = (
        combined_df
        .drop_duplicates(subset=["Dates"])
        .sort_values("Dates")
        .reset_index(drop=True)
    )
 
    # --- Write back to file ---
    if format == "txt":
        combined_df.to_csv(filename, sep=";", index=False)
    else:  # csv
        combined_df.to_csv(filename, index=False)
 
    print(f"Updated: {filename} ({len(new_df)} new rows added, {len(combined_df)} total rows)")
    return filename