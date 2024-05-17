"""
Functions adapted from Hannah Bloomfield's code for S2S4E for European demand model. 

First testing with daily reanalysis data for the UK.

Before moving on to see whether decadal predictions can be used for this.
"""

import glob

import numpy as np
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
import shapely.geometry as sgeom
import pandas as pd
import xarray as xr


# For CLEARHEADS, Hannah has already preprocessed the T2M data to
# be at the NUTS0 level.
# So we will just load and use this data instead of going through the faff
# of finding T2M data on JASMIN and processing it ourselves.
def load_clearheads(
    filename: str,
    directory: str = "/home/users/benhutch/CLEARHEADS_EU_Power_Data",
    trend_level: float = -9999.0,
    index: str = "time_in_hours_from_first_jan_1950",
    columns: str = "NUTS",
    time_units: str = "h",
    start_date: str = "1950-01-01",
    nuts_keys_name: str = "NUTS_keys",
    trend_levels_name: str = "trend_levels",
) -> pd.DataFrame:
    """
    Load the CLEARHEADS data into a pandas DataFrame.

    Parameters
    ----------

    filename: str
        The filename for the CLEARHEADS data.

    directory: str
        The directory where the CLEARHEADS data are stored.

    trend_level: float
        The value to replace the trend level with.

    index: str
        The name of the index column.

    columns: str
        The name of the columns.

    time_units: str
        The units of the time.

    start_date: str
        The start date of the time.

    nuts_keys_name: str
        The name of the NUTS keys variable.

    trend_levels_name: str
        The name of the trend levels variable.

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data.

    """

    # Find the file
    file = glob.glob(f"{directory}/{filename}")

    # If there are no files raise an error
    if len(file) == 0:
        raise FileNotFoundError(f"No file found for {filename}")
    elif len(file) > 1:
        raise ValueError(f"Multiple files found for {filename}")

    # Load the data
    ds = xr.open_dataset(file[0])

    # Assert that NUTS_keys can be extracted from the dataset
    assert (
        nuts_keys_name in ds.variables
    ), f"Cannot find {nuts_keys_name} in the dataset variables."

    # Extract the NUTS keys
    nuts_keys = ds[nuts_keys_name].values

    # Print the NUTS keys
    print(f"NUTS keys: {nuts_keys}")

    # If the trend level is not -9999.0
    if trend_level != -9999.0:
        print(f"Extracting data with trend level {trend_level}")
        
        # Extract the data
        trend_levels = ds[trend_levels_name].values

        # Print the trend levels
        print(f"Trend levels: {trend_levels}")

        # Find the index of the trend level
        trend_idx = np.where(trend_levels == trend_level)[0][0]

        # Extract the data
        ds = ds.isel(trend=trend_idx)

    # Convert the dataset to a DataFrame
    df = ds.to_dataframe()

    # If the trend level is not -9999.0
    if trend_level != -9999.0:
        # Pivot the dataframe
        df = df.reset_index().pivot(
            index=index, 
            columns=columns,
            values="detrended_data",
        )
    else:
        # Pivot the dataframe
        df = df.reset_index().pivot(
            index=index, 
            columns=columns,
            values="timeseries_data",
        )

    # Add the nuts keys as the columns
    df.columns = nuts_keys

    # Convert the index to a datetime
    df.index = pd.to_datetime(df.index, unit=time_units, origin=start_date)

    return df

#  Calculate the heating degree days and cooling degree days
def calc_hdd_cdd(
    df: pd.DataFrame,
    hdd_base: float = 15.5,
    cdd_base: float = 22.0,
    temp_suffix: str = "t2m",
    hdd_suffix: str = "hdd",
    cdd_suffix: str = "cdd",
    index_name: str = "time_in_hours_from_first_jan_1950",
) -> pd.DataFrame:
    """
    Calculate the heating degree days and cooling degree days.

    Parameters
    ----------

    df: pd.DataFrame
        The CLEARHEADS data.

    hdd_base: float
        The base temperature for the heating degree days.

    cdd_base: float
        The base temperature for the cooling degree days.

    temp_suffix: str
        The suffix for the temperature.

    hdd_suffix: str
        The suffix for the heating degree days.

    cdd_suffix: str
        The suffix for the cooling degree days.

    index_name: str
        The name of the index column.

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data with the heating degree days and cooling degree days.

    """

    # if the data is not already in daily format, resample to daily
    if df.index.freq != "D":
        print("Resampling to daily")
        
        # Resample the data
        df = df.resample("D").mean()

    # add the temperature suffix to the columns
    df.columns = [f"{col}_{temp_suffix}" for col in df.columns]

    # Loop over the columns
    for col in df.columns:
        # set up the column names
        hdd_col = f"{col}_{hdd_suffix}"
        cdd_col = f"{col}_{cdd_suffix}"

        # Calculate the heating degree days
        df[hdd_col] = df[col].apply(lambda x: max(0, hdd_base - x))

        # Calculate the cooling degree days
        df[cdd_col] = df[col].apply(lambda x: max(0, x - cdd_base))

    return df
