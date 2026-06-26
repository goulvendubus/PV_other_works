from model.utils import *

import pandas as pd
from datetime import datetime

import pytest
from model.download import download_fmi_station_data, download_fmi_power_data



@pytest.mark.model
def test_download_fmi_station_data(tmp_path):
    result = download_fmi_station_data(
        "Turku",
        (2025, 2, 1),
        (2025, 2, 2),
        path=tmp_path
    )
    assert result is not None

@pytest.mark.model
def test_download_fmi_power_data(tmp_path):
    result = download_fmi_power_data(
        "Turku",
        (2025, 2, 1),
        (2025, 2, 2),
        path=tmp_path
    )
    assert result is not None

#download_cams_solar_radiation_data_v2("Kuopio", (2025, 2, 1), (2025, 2, 20), format="csv_expert")
#download_cams_environmental_data("Kuopio", (2025, 2, 1), (2025, 2, 10), leadtime_hour=["0", "2", "4", "6", "8", "10", "12"])

# pvfc.set_location(latitude=60.188, longitude=24.941)
# pvfc.set_angles(15, 135)
# pvfc.set_module_elevation(47)
# pvfc.set_nominal_power_kw(1000)
# pvfc.set_default_albedo(0.2)
# df1 = pvfc.get_fmi_forecast_for_interval(datetime(2024, 1, 1), datetime(2024, 1, 31))
# df2 = pvfc.get_default_fmi_forecast(10)
# df3 = pvfc.get_default_clearsky_forecast(10)

# print(df1.head())
# print("\nDefault forecast:")
# print(df2.head(10))
# print("\nClearsky forecast:")
# print(df3.head(10))


# import cdsapi

# dataset = "reanalysis-era5-land-timeseries"
# request = {
#     "variable": ["2m_temperature"],
#     "location": {"longitude": 0, "latitude": 0},
#     "date": ["2020-01-02/2020-06-13"],
#     "data_format": "csv"
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
