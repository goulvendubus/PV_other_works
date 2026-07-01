import pandas as pd
import numpy as np
from model.download import download_fmi_power_data, download_fmi_station_data
from model.utils import csv_transform, format_df_for_pvfc, align_to_common_day
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fmi_pv_forecaster as pvfc

##___Parameters___
Ps = 1.353e3
Stations = {"Kuopio": (62.892, 27.634), "Kumpula": (60.203071, 24.961305),}

station_name = "Kuopio"
latitude, longitude = Stations[station_name]

pvfc.set_location(latitude=latitude, longitude=longitude)
pvfc.set_angles(15, 217)
pvfc.set_module_elevation(95)
pvfc.set_nominal_power_kw(20.28)
pvfc.set_default_albedo(0.15)





day = pd.Timestamp("2023-10-12", tz="Europe/Helsinki")




##_______________________________Direct power output data_____________________________________________________
download_fmi_power_data(station_name, start=day)

df = csv_transform(f"data/{station_name.lower()}/fmi_power_data.csv")
df = df[df["Dates"].dt.strftime("%Y-%m-%d") == "2023-10-12"]

df_aligned = align_to_common_day(df)
df_aligned = df_aligned.sort_values("aligned_time")




##_______________________________Clearsky Model_____________________________________________________
""" Use clearsky estimate model from fmi_pv_forecaster to get the clearsky estimate for each day and align them to a common day for comparison. """

# pollen day: 
day_naive = day.tz_localize(None)
df_clearsky_mod_day  = pvfc.get_clearsky_estimate_for_interval(day_naive, day_naive + pd.Timedelta(days=1), timestep=1)
df_clearsky_mod_day["Dates"] = df_clearsky_mod_day.index.tz_convert(None)  # strip tz to match other dfs
df_clearsky_mod_day = df_clearsky_mod_day.reset_index(drop=True)
df_clearsky_mod_day_aligned = align_to_common_day(df_clearsky_mod_day)
df_clearsky_mod_day_aligned = df_clearsky_mod_day_aligned.sort_values("aligned_time")


##_______________________________Environmental data download and processing_____________________________________________________
"""
Download and process environmental data from FMI station data for the specified days.
The data is filtered to keep only rows where every column that has data is non-null, and negative values are removed. 
The relevant columns are renamed for consistency.
The process_radiation_df function from fmi_pv_forecaster is used to process the radiation data, and the results are aligned to a common day for comparison.
"""
download_fmi_station_data(station_name, start=day.tz_localize(None))

df_env = csv_transform(f"data/{station_name.lower()}/fmi_station_data.csv")
df_env["Dates"] = pd.to_datetime(df_env["Dates"])
print("df_env après import\n", df_env.head())

# Keep only rows where every column that has data is non-null.
required_cols  = ["Dates", "FMI - GHI", "FMI - DHI", "FMI - DNI", "FMI - 2M_AIR_TEMP"]
cols_with_data = [c for c in required_cols if df_env[c].notna().any()]
df_env = df_env[df_env[cols_with_data].notnull().all(axis=1)]

df_env.rename(columns={"FMI - GHI": "ghi", "FMI - DHI": "dhi",
                        "FMI - DNI": "dni", "FMI - 2M_AIR_TEMP": "T"}, inplace=True)

for col in ["ghi", "dhi", "dni", "T"]:
    df_env[col] = pd.to_numeric(df_env[col], errors="coerce")

irr_cols = [c for c in ["ghi", "dhi", "dni"] if df_env[c].notna().any()]

df_env = df_env.dropna(subset=irr_cols)

bounds_mask = pd.Series(True, index=df_env.index)
for col in irr_cols:
    bounds_mask &= (df_env[col] >= 0)
df_env = df_env[bounds_mask]

df_env = df_env[df_env["ghi"] > 0]
# DNI is filled by the download function; keep only daytime rows (dni > 0)
if "dni" in irr_cols:
    df_env = df_env[df_env["dni"] > 0]

df_env = df_env.replace([np.inf, -np.inf], np.nan)
df_env = df_env.dropna(subset=irr_cols)
df_env = df_env.set_index("Dates")

# Date filtering
df_env = df_env[df_env.index.normalize() == pd.Timestamp("2023-10-12")]

print("df_env après suppression des valeurs manquantes et négatives\n", df_env.head())
print("Irradiance columns with data:", irr_cols)
print("Any NaN:",      df_env[irr_cols].isna().any())
print("Any negative:", (df_env[irr_cols] < 0).any())
print("Any zero rows:", ((df_env[irr_cols] == 0).all(axis=1)).sum())

cols_needed = ["ghi", "dhi", "dni", "module_temp"]

df_allsky_mod = align_to_common_day(pvfc.process_radiation_df(format_df_for_pvfc(df_env)[cols_needed]))

##______________________________Plotting_____________________________________________________
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_aligned["aligned_time"],
    y=df_aligned["In12Power[W]"],
    mode='lines',
    name=f'{day.date()} - Measured output (pollen day)'
))
fig.add_trace(go.Scatter(
    x=df_clearsky_mod_day_aligned["aligned_time"],
    y=df_clearsky_mod_day_aligned["output"],
    mode='lines',
    name=f'{day.date()} - Clearsky FMI output modelization (pollen day)'
))
fig.add_trace(go.Scatter(
    x=df_allsky_mod["aligned_time"],
    y=df_allsky_mod["output"],
    mode='lines',
    name=f'{day.date()} - Allsky FMI output modelization (pollen day)'
))

fig.update_layout(
    title=f"Power comparison (midnight to midnight) for {station_name}",
    xaxis_title="Time of day",
    yaxis_title="Power [W]",
    xaxis=dict(
        tickformat="%H:%M",
        dtick=2 * 60 * 60 * 1000  # 2 hours in milliseconds
    )
)

fig.show()
fig.write_html(f"plots/{station_name.lower()}_power_comparison_{day.date()}.html")
fig.write_image(f"plots/{station_name.lower()}_power_comparison_{day.date()}.png", scale=2, width=1200, height=800)




## Plotting difference between pollen and no pollen days
df_merged = pd.merge(
    df_aligned[["aligned_time", "In12Power[W]"]],
    df_allsky_mod[["aligned_time", "output"]],
    on="aligned_time",
    how="inner",
    suffixes=("_pollen", "_no_pollen")
)

# Compute difference
df_merged["diff"] = (
    (df_merged["output"] - df_merged["In12Power[W]"]) / df_merged["output"]
)


fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_merged["aligned_time"],
    y=df_merged["diff"],
    mode='lines',
    name= f"{day.date()} (pollen day)"
))
fig2.update_layout(
    title="Relative error (model - measured) / model",
    xaxis_title="Time of day",
    yaxis_title="Relative error",
    xaxis=dict(
        tickformat="%H:%M",
        dtick=2 * 60 * 60 * 1000
    )
)
fig2.show()
fig2.write_html(f"plots/{station_name.lower()}_relative_error_{day.date()}.html")
fig2.write_image(f"plots/{station_name.lower()}_relative_error_{day.date()}.png", scale=2, width=1200, height=800)