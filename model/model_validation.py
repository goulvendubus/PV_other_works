"""
File with comaprison between fmi and pvlib models for solar forecasting.
used on FMI and CAMS solar radiation data

"""



import pandas as pd
import xarray as xr
import cfgrib
from pathlib import Path
import os
from .download import *
from .utils import *



start_date = (2025, 2, 1)
end_date = (2025, 2, 20)
