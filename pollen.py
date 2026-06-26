import pandas as pd
import numpy as np
from model.download import download_fmi_power_data, download_fmi_station_data
from model.utils import csv_transform
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fmi_pv_forecaster as pvfc

##___Parameters___
Ps = 1.353e3


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


pvfc.set_location(latitude=62.892, longitude=27.634)
pvfc.set_angles(15, 217)
pvfc.set_module_elevation(95)
pvfc.set_nominal_power_kw(20.28)
pvfc.set_default_albedo(0.15)





pollen_day = pd.Timestamp("2021-05-12", tz="Europe/Helsinki")
no_pollen_day1 = pd.Timestamp("2021-08-06", tz="Europe/Helsinki")
no_pollen_day2 = pd.Timestamp("2021-04-21", tz="Europe/Helsinki")
no_pollen_day3 = pd.Timestamp("2021-04-24", tz="Europe/Helsinki")




##_______________________________Direct power output data_____________________________________________________
download_fmi_power_data("Kuopio", start=pollen_day)
download_fmi_power_data("Kuopio", start=no_pollen_day1  )
download_fmi_power_data("Kuopio", start=no_pollen_day2  )
download_fmi_power_data("Kuopio", start=no_pollen_day3  )

df = csv_transform("data/kuopio/fmi_power_data.csv")
df_no_pollen1 = df[df["Dates"].str.match("2021-08-06")]
df_no_pollen2 = df[df["Dates"].str.match("2021-04-21")]
df_no_pollen3 = df[df["Dates"].str.match("2021-04-24")]
df_pollen = df[df["Dates"].str.match("2021-05-12")]

df_pollen["Dates"] = pd.to_datetime(df_pollen["Dates"])
df_no_pollen1["Dates"] = pd.to_datetime(df_no_pollen1["Dates"])
df_no_pollen2["Dates"] = pd.to_datetime(df_no_pollen2["Dates"])
df_no_pollen3["Dates"] = pd.to_datetime(df_no_pollen3["Dates"])

df_pollen_aligned = align_to_common_day(df_pollen)
df_no_pollen1_aligned = align_to_common_day(df_no_pollen1)
df_no_pollen2_aligned = align_to_common_day(df_no_pollen2)
df_no_pollen3_aligned = align_to_common_day(df_no_pollen3)
df_pollen_aligned = df_pollen_aligned.sort_values("aligned_time")
df_no_pollen1_aligned = df_no_pollen1_aligned.sort_values("aligned_time")
df_no_pollen2_aligned = df_no_pollen2_aligned.sort_values("aligned_time")
df_no_pollen3_aligned = df_no_pollen3_aligned.sort_values("aligned_time")



##_______________________________Environmental data_____________________________________________________



# --- clearsky
pollen_day_naive = pollen_day.tz_localize(None)
df_clearsky_mod = pvfc.get_clearsky_estimate_for_interval(pollen_day_naive, pollen_day_naive + pd.Timedelta(days=1), timestep=1)

df_clearsky_mod["Dates"] = df_clearsky_mod.index.tz_convert(None)  # strip tz to match other dfs
df_clearsky_mod = df_clearsky_mod.reset_index(drop=True)

df_clearsky_mod_aligned = align_to_common_day(df_clearsky_mod)
df_clearsky_mod_aligned = df_clearsky_mod_aligned.sort_values("aligned_time")

# --- allsky
# df_env_pollen = pvfc.meps_loader.collect_fmi_opendata(latitude=62.892, longitude=27.634, start_time=pollen_day, end_time=pollen_day + pd.Timedelta(days=1))
# df_env_no_pollen1 = pvfc.meps_loader.collect_fmi_opendata(latitude=62.892, longitude=27.634, start_time=no_pollen_day1, end_time=no_pollen_day1 + pd.Timedelta(days=1))
# df_env_no_pollen2 = pvfc.meps_loader.collect_fmi_opendata(latitude=62.892, longitude=27.634, start_time=no_pollen_day2, end_time=no_pollen_day2 + pd.Timedelta(days=1))
# df_env_no_pollen3 = pvfc.meps_loader.collect_fmi_opendata(latitude=62.892, longitude=27.634, start_time=no_pollen_day3, end_time=no_pollen_day3 + pd.Timedelta(days=1))

# df_mod_pollen = pvfc.process_radiation_df(df_env_pollen)
# df_mod_no_pollen1 = pvfc.process_radiation_df(df_env_no_pollen1)
# df_mod_no_pollen2 = pvfc.process_radiation_df(df_env_no_pollen2)
# df_mod_no_pollen3 = pvfc.process_radiation_df(df_env_no_pollen3)

# df_mod_pollen_aligned = align_to_common_day(df_mod_pollen)
# df_mod_no_pollen1_aligned = align_to_common_day(df_mod_no_pollen1)
# df_mod_no_pollen2_aligned = align_to_common_day(df_mod_no_pollen2)
# df_mod_no_pollen3_aligned = align_to_common_day(df_mod_no_pollen3)


download_fmi_station_data("Kuopio", start=pollen_day.tz_localize(None))
download_fmi_station_data("Kuopio", start=no_pollen_day1.tz_localize(None))
download_fmi_station_data("Kuopio", start=no_pollen_day2.tz_localize(None))
download_fmi_station_data("Kuopio", start=no_pollen_day3.tz_localize(None))

df_env = csv_transform("data/kuopio/fmi_station_data.csv")
df_env["Dates"] = pd.to_datetime(df_env["Dates"])
print("df_env après import", df_env.head())

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
df_env_pollen = df_env[df_env.index.normalize() == pd.Timestamp("2021-05-12")]
df_env_no_pollen1 = df_env[df_env.index.normalize() == pd.Timestamp("2021-08-06")]
df_env_no_pollen2 = df_env[df_env.index.normalize() == pd.Timestamp("2021-04-21")]
df_env_no_pollen3 = df_env[df_env.index.normalize() == pd.Timestamp("2021-04-24")]

print("df_env après suppression des valeurs manquantes et négatives", df_env.head())
print("Irradiance columns with data:", irr_cols)
print("Any NaN:",      df_env[irr_cols].isna().any())
print("Any negative:", (df_env[irr_cols] < 0).any())
print("Any zero rows:", ((df_env[irr_cols] == 0).all(axis=1)).sum())

cols_needed = ["ghi", "dhi", "dni"]

df_allsky_pollen_mod = align_to_common_day(
    pvfc.process_radiation_df(df_env_pollen[cols_needed])
)

df_allsky_pollen_mod = align_to_common_day(pvfc.process_radiation_df(df_env_pollen))
df_allsky_no_pollen1_mod = align_to_common_day(pvfc.process_radiation_df(df_env_no_pollen1))
df_allsky_no_pollen2_mod = align_to_common_day(pvfc.process_radiation_df(df_env_no_pollen2))
df_allsky_no_pollen3_mod = align_to_common_day(pvfc.process_radiation_df(df_env_no_pollen3))






##______________________________Plotting_____________________________________________________
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df_pollen_aligned["aligned_time"],
#     y=df_pollen_aligned["In12Power[W]"],
#     mode='lines',
#     name='Pollen day (2021-05-12)'
# ))
# fig.add_trace(go.Scatter(
#     x=df_clearsky_mod_aligned["aligned_time"],
#     y=df_clearsky_mod_aligned["output"],
#     mode='lines',
#     name=f'Clearsky FMI output modelization: {pollen_day.date()} (pollen day)'
# ))
# fig.add_trace(go.Scatter(
#     x=df_no_pollen1_aligned["aligned_time"],
#     y=df_no_pollen1_aligned["In12Power[W]"],
#     mode='lines',
#     name=f'No pollen day: {no_pollen_day1.date()} (no pollen day)'
# ))
# fig.add_trace(go.Scatter(
#     x=df_no_pollen2_aligned["aligned_time"],
#     y=df_no_pollen2_aligned["In12Power[W]"],
#     mode='lines',
#     name=f'No pollen day: {no_pollen_day2.date()} (no pollen day)'
# ))
# fig.add_trace(go.Scatter(
#     x=df_no_pollen3_aligned["aligned_time"],
#     y=df_no_pollen3_aligned["In12Power[W]"],
#     mode='lines',
#     name=f'Measured output: {no_pollen_day3.date()} (no pollen day)'
# ))

# # fig.add_trace(go.Scatter(
# #     x=df_allsky_pollen_mod["aligned_time"],
# #     y=df_allsky_pollen_mod["output"],
# #     mode='lines',
# #     name=f'FMI allsky output modelization: {pollen_day.date()} (pollen day)'
# # ))
# # fig.add_trace(go.Scatter(
# #     x=df_allsky_no_pollen1_mod["aligned_time"],
# #     y=df_allsky_no_pollen1_mod["output"],
# #     mode='lines',
# #     name=f'FMI allsky output modelization: {no_pollen_day1.date()} (no pollen day)'
# # ))
# # fig.add_trace(go.Scatter(
# #     x=df_allsky_no_pollen2_mod["aligned_time"],
# #     y=df_allsky_no_pollen2_mod["output"],
# #     mode='lines',
# #     name=f'FMI allsky output modelization: {no_pollen_day2.date()} (no pollen day)'
# # ))
# # fig.add_trace(go.Scatter(
# #     x=df_allsky_no_pollen3_mod["aligned_time"],
# #     y=df_allsky_no_pollen3_mod["output"],
# #     mode='lines',
# #     name=f'FMI allsky output modelization: {no_pollen_day3.date()} (no pollen day)'
# # ))
# fig.update_layout(
#     title="Power comparison (midnight to midnight)",
#     xaxis_title="Time of day",
#     yaxis_title="Power [W]",
#     xaxis=dict(
#         tickformat="%H:%M",
#         dtick=2 * 60 * 60 * 1000  # 2 hours in milliseconds
#     )
# )

# fig.show()





# ## Plotting difference between pollen and no pollen days


# df_merged = pd.merge(
#     df_pollen_aligned[["aligned_time", "In12Power[W]"]],
#     df_no_pollen1_aligned[["aligned_time", "In12Power[W]"]],
#     on="aligned_time",
#     how="inner",
#     suffixes=("_pollen", "_no_pollen")
# )

# # Compute difference
# df_merged["diff"] = (
#     df_merged["In12Power[W]_no_pollen"]
#     - df_merged["In12Power[W]_pollen"]
# )


# fig2 = go.Figure()
# fig2.add_trace(go.Scatter(
#     x=df_merged["aligned_time"],
#     y=df_merged["diff"],
#     mode='lines',
#     name=f'Difference (No pollen ({no_pollen_day1.date()}) - Pollen ({pollen_day.date()}))'
# ))
# fig2.update_layout(
#     title="Power difference (midnight to midnight)",
#     xaxis_title="Time of day",
#     yaxis_title="Power difference [W]",
#     xaxis=dict(
#         tickformat="%H:%M",
#         dtick=2 * 60 * 60 * 1000
#     )
# )
# fig2.show()


