"""
Functions adapted from Hannah Bloomfield's code for S2S4E for European demand model. 

First testing with daily reanalysis data for the UK.

Before moving on to see whether decadal predictions can be used for this.
"""

import glob
import os
import sys
import re

import numpy as np
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
import shapely.geometry as sgeom
import pandas as pd
import xarray as xr
from tqdm import tqdm

# import dictionaries from unseen_dictionaries.py
import unseen_dictionaries as udicts

# Specific imports from functions
from functions import preprocess, set_integer_time_axis


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


# Write a function to load in the decadal prediction data for a given initialisation year
def load_dcpp_data(
    model_variable: str,
    model: str,
    init_years: list[int],
    experiment: str = "dcppA-hindcast",
    frequency: str = "day",
    engine: str = "netcdf4",
    parallel: bool = True,
    grid: dict = udicts.eu_grid,
    csv_fpath: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
) -> xr.Dataset:
    """
    Load the decadal prediction data for a given initialisation year.
    Subsets the data to the European domain.

    Parameters
    ----------

    model_variable: str
        The variable to load from the model.

    model: str
        The model to load the data from.

    init_years: int
        The initialisation years to load the data for.

    experiment: str
        The experiment to load the data from.

    frequency: str
        The frequency of the data.

    engine: str
        The engine to use to load the data.

    parallel: bool
        Whether to load the data in parallel.

    grid: dict
        The dictionary of the grid.

    csv_fpath: str
        The file path for the CSV file.

    Returns
    -------

    ds: xr.Dataset
        The loaded decadal prediction data.

    """

    # Try extracting the lat and lon bounds
    try:
        lon1, lon2 = grid["lon1"], grid["lon2"]
        lat1, lat2 = grid["lat1"], grid["lat2"]
    except KeyError:
        raise KeyError("Cannot extract lat and lon bounds from grid dictionary.")

    # Check that the csv file exists
    if not os.path.exists(csv_fpath):
        raise FileNotFoundError(f"Cannot find the file {csv_fpath}")

    # Load in the csv file
    csv_data = pd.read_csv(csv_fpath)

    # Extract the path for the given model, experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == model_variable)
        & (csv_data["frequency"] == frequency),
        "path",
    ].values[0]

    # Assert that theb model path exists
    assert os.path.exists(model_path), f"Cannot find the model path {model_path}"

    # Assert that the model path is not empty
    assert os.listdir(model_path), f"Model path {model_path} is empty"

    # print the model path
    print(f"Model path: {model_path}")

    # Extract the root of the model path
    model_path_root = model_path.split("/")[1]

    # print the model path root
    print(f"Model path root: {model_path_root}")

    agg_files = []

    # Depending on the model path root, load the data differently
    if model_path_root == "gws":
        print("Loading data from JASMIN GWS")

        # Loop over the initialisation years
        for init_year in init_years:
            # glob the files in the directory containing the initialisation year
            files = glob.glob(os.path.join(model_path, f"*{init_year}*"))

            # # print the len of the files
            # print(f"Number of files: {len(files)}")

            # Assert that there are files
            assert len(files) > 0, f"No files found for {init_year} in {model_path}"

            # Append the files to the aggregated files list
            agg_files.extend(files)
    elif model_path_root == "badc":
        print("Loading data from BADC")

        # Loop over the initialisation years
        for init_year in init_years:
            # Form the path to the data
            year_path = f"{model_path}/s{init_year}-r*i?p?f?/{frequency}/{model_variable}/g?/files/d????????/*.nc"

            # glob the files in the directory containing the initialisation year
            files = glob.glob(year_path)

            # # print the len of the files
            # print(f"Number of files: {len(files)} for {init_year} in {model_path}")

            # Assert that there are files
            assert len(files) > 0, f"No files found for {init_year} in {model_path}"

            # Append the files to the aggregated files list
            agg_files.extend(files)
    else:
        raise ValueError(f"Model path root {model_path_root} not recognised.")

    # Extract the variants
    variants = [
        re.split(r'_s....-', agg_files.split("/")[-1].split("_g")[0])[1] for agg_files in files
    ]

    # Print the unique variants
    print(f"Unique variants: {set(variants)}")

    # print the shape of the agg_files
    print(f"Shape of agg_files: {len(agg_files)}")

    # Set up the init_year list
    init_year_list = []

    # Loop over the initialisation years
    for init_year in tqdm(init_years, desc="Loading data"):
        # member list for the xr objects
        member_list = []

        # Load the data by looping over the unique variants
        for variant in set(variants):
            # Find the variant files
            variant_files = [file for file in agg_files if f"s{init_year}-{variant}" in file]

            # Open all leads for the specified variant
            member_ds = xr.open_mfdataset(
                variant_files,
                combine="nested",
                concat_dim="time",
                preprocess=lambda ds: preprocess(ds),  # define preprocess function
                parallel=parallel,
                engine=engine,
                coords="minimal",  # explicitly set coords to minimal
                data_vars="minimal",  # explicitly set data_vars to minimal
                compat="override",  # override the default behaviour
            ).squeeze()  # remove any dimensions of length 1

            # if variant_label = set(variants)[0]
            if init_year == init_years[0] and variant == list(set(variants))[0]:
                # Set new integer time axis
                member_ds = set_integer_time_axis(
                    xro=member_ds,
                    frequency=frequency,
                    first_month_attr=True,
                )
            else:
                # Set new integer time axis
                member_ds = set_integer_time_axis(
                    xro=member_ds,
                    frequency=frequency,
                    first_month_attr=False,
                )

            # Append the member dataset to the list
            member_list.append(member_ds)

        # Concatenate the member list by the member dimension
        ds = xr.concat(member_list, dim="member")

        # Append the dataset to the init_year list
        init_year_list.append(ds)
    # Concatenate the init_year list by the init_year dimension
    ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

    # Set up the members
    ds["member"] = np.arange(1, ds.sizes["member"] + 1)
    ds['init'] = init_years

    # print the dataset
    print(ds)

    # # print that we are exiting the script
    # print("Exiting the script.")
    # print("--------------------")
    # sys.exit()


    # extract the data for the variable
    ds = ds[model_variable]

    # Chunk the data
    ds = ds.chunk({"time": "auto", "lat": "auto", "lon": "auto", "member": "auto"})

    return ds


# Define a function to calculate the spatial mean of a masked dataset
# and convert to a dataframe
def calc_spatial_mean(
    ds: xr.Dataset,
    country: str,
    variable: str,
    variable_name: str = None,
    convert_kelv_to_cel: bool = True,
) -> pd.DataFrame:
    """
    Calculate the spatial mean of a masked dataset and convert to a DataFrame.

    Parameters
    ----------

    ds: xr.Dataset
        The dataset to calculate the spatial mean from.

    country: str
        The country to calculate the spatial mean for.

    variable: str
        The variable to calculate the spatial mean for.

    variable_name: str
        The name of the variable in the dataset.

    convert_kelv_to_cel: bool
        Whether to convert the data from Kelvin to Celsius.

    Returns
    -------

    df: pd.DataFrame
        The DataFrame of the spatial mean.

    """

    # set up data as none
    data = None

    # if the type of ds is xarray.DataArray
    if isinstance(ds, xr.DataArray):
        # # raise an error
        # raise ValueError("Input data must be a Dataset.")

        # extract the values
        data = ds.values
    else:
        ds = ds[variable]

        # # Convert to a numpy array
        # data = ds.values

    # if the shape has length 3
    if isinstance(data, np.ndarray) and len(data.shape) == 3:
        print("Processing observed data")

        # Take the mean over the lat and lon dimensions
        data_mean = np.nanmean(data, axis=(1, 2))

        # Set up the time index
        time_index = ds.time.values

        # Set up the DataFrame
        df = pd.DataFrame(data_mean, index=time_index, columns=[f"{country}_{variable}"])

        # If convert_kelv_to_cel is True
        if convert_kelv_to_cel:
            # Convert the data from Kelvin to Celsius
            df[f"{country}_{variable}"] = df[f"{country}_{variable}"] - 273.15
    elif isinstance(ds, xr.DataArray):
        print("Processing model data")

        # assert that variable name is not none
        assert variable_name is not None, "Variable name must be provided."

        # Take the mean over the lat and lon dimensions
        ds_mean = ds.mean(dim=["lat", "lon"])

        # Set the name for the ds
        ds.name = f"{country}_{variable_name}"

        # convert to a dataframe
        df = ds_mean.to_dataframe()

        # Rename the column at index 0
        df.columns = [f"{country}_{variable_name}"]

        # Reset the index
        df = df.reset_index()

        # if variable_name is tas or t2m
        if variable_name == "tas" or variable_name == "t2m":
            # Convert the data from Kelvin to Celsius
            df[f"{country}_{variable_name}"] = df[f"{country}_{variable_name}"] - 273.15
    else:
        raise ValueError("Data shape not recognised.")

    return df

# # Define a function to perform the bias correction
# def calc_bc_coeffs(
#     model_variable: str,
#     model: str,
#     experiment: str,
#     start_year: int,
#     end_year: int,
#     month: int,
#     grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
#     rg_algo: str = "bilinear",
#     periodic: bool = True,
#     grid: dict = udicts.eu_grid,
#     lead_months: int = 17,  # For HadGEM3-GC31-MM all lead months until end of first ONDJFM winter
#     frequency: str = "day",
#     engine: str = "netcdf4",
#     parallel: bool = True,
#     csv_fpath: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
#     obs_fpath: str = "DOWNLOAD_FILE",
#     obs_variable: str = "tas",
# ) -> xr.DataArray:
#     """
#     Calculate the bias correction coefficients for each model, month, lat,
#     and lon for a given model and experiment.
#     Using the bias correction from Luo et al. (2018) (doi:10.3390/w10081046).

#     Parameters
#     ----------

#     model_variable: str
#         The variable to load from the model.

#     model: str
#         The model to load the data from.

#     experiment: str
#         The experiment to load the data from.

#     start_year: int
#         The start year for the bias correction.

#     end_year: int
#         The end year for the bias correction.

#     month: int
#         The month for the bias correction.

#     grid_bounds: list[float]
#         The grid bounds for the global grid.

#     rg_algo: str
#         The regridding algorithm to use.

#     periodic: bool
#         Whether the grid is periodic.

#     grid: dict
#         The dictionary of the grid.

#     lead_months: int
#         The number of lead months to load.

#     frequency: str
#         The frequency of the data.

#     engine: str
#         The engine to use to load the data.

#     parallel: bool
#         Whether to load the data in parallel.

#     csv_fpath: str
#         The file path for the CSV file.

#     obs_fpath: str
#         The file path for the observations.

#     obs_variable: str
#         The variable to load from the observations.

#     Returns
#     -------

#     bc_coeffs: xr.DataArray
#         The bias correction coefficients with shape (lat, lon).

#     """

#     # First load in the model data

#     return None


#  Calculate the heating degree days and cooling degree days
def calc_hdd_cdd(
    df: pd.DataFrame,
    country_name: str = None,
    variable_name: str = None,
    hdd_base: float = 15.5,
    cdd_base: float = 22.0,
    temp_suffix: str = "t2m",
    hdd_suffix: str = "hdd",
    cdd_suffix: str = "cdd",
) -> pd.DataFrame:
    """
    Calculate the heating degree days and cooling degree days.

    Parameters
    ----------

    df: pd.DataFrame
        The CLEARHEADS data.

    country_name: str
        The name of the country.

    variable_name: str
        The name of the variable.

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

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data with the heating degree days and cooling degree days.

    """
    # if lead is not one of the columns
    if "lead" not in df.columns:
        print("lead not in columns, processing observed data")
        # if the data is not already in daily format, resample to daily
        if df.index.freq != "D":
            print("Resampling to daily")

            # Resample the data
            df = df.resample("D").mean()

        # add the temperature suffix to the columns
        df.columns = [f"{col}_{temp_suffix}" for col in df.columns]

        # Loop over the columns
        for col in df.columns:
            # strip t2m from the column name
            col_raw = col.replace(f"_{temp_suffix}", "")

            # set up the column names
            hdd_col = f"{col_raw}_{hdd_suffix}"
            cdd_col = f"{col_raw}_{cdd_suffix}"

            # Calculate the heating degree days
            df[hdd_col] = df[col].apply(lambda x: max(0, hdd_base - x))

            # Calculate the cooling degree days
            df[cdd_col] = df[col].apply(lambda x: max(0, x - cdd_base))
    elif "lead" in df.columns:
        # assert the the len unique leads is greater then 12
        assert len(df["lead"].unique()) > 12, "Lead column not found."

        print("Model data already in daily format.")

        # assert that country name is not none
        assert country_name is not None, "Country name must be provided."

        # assert that variable name is not none
        assert variable_name is not None, "Variable name must be provided."

        # Set up the hdd_col
        hdd_col = f"{country_name}_{variable_name}_{hdd_suffix}"
        cdd_col = f"{country_name}_{variable_name}_{cdd_suffix}"

        # Calculate the heating degree days
        df[hdd_col] = df[f"{country_name}_{variable_name}"].apply(lambda x: max(0, hdd_base - x))

        # Calculate the cooling degree days
        df[cdd_col] = df[f"{country_name}_{variable_name}"].apply(lambda x: max(0, x - cdd_base))
    else:
        raise ValueError("Data not in daily format and lead column not found.")

    return df


# Write a function which calculates the weather dependent demand
# Based on the heating and cooling degree days and the demand coefficients
def calc_national_wd_demand(
    df: pd.DataFrame,
    fpath_reg_coefs: str = "/home/users/benhutch/ERA5_energy_update/ERA5_Regression_coeffs_demand_model.csv",
    demand_year: float = 2017.0,
    country_names: dict = udicts.countries_nuts_id,
    hdd_name: str = "HDD",
    cdd_name: str = "CDD",
) -> pd.DataFrame:
    """
    Calculate the national weather dependent demand.

    Parameters
    ----------

    df: pd.DataFrame
        The CLEARHEADS data.

    fpath_reg_coefs: str
        The file path for the regression coefficients.

    demand_years: float
        The year for the time coefficient.

    country_names: dict
        The dictionary of country names. Matched up the full country names
        with the NUTS IDs.

    hdd_name: str
        The name of the heating degree days.

    cdd_name: str
        The name of the cooling degree days.

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data with the national weather dependent demand.

    """

    # Loop over the columns in the DataFrame
    for col in df.columns:
        # Loop over the country names
        for country_name, country_id in country_names.items():
            # print(f"Calculating demand for {country_name}")
            # print(f"Country ID: {country_id}")
            # if the country id is in the column name
            if country_id in col:
                # Split the column name by _
                col_split = col.split("_")

                # Set up the new column name
                new_col = f"{country_name}_{col_split[1]}"

                # Update the column name
                df = df.rename(columns={col: new_col})

    # Load int the regression coefficients data
    reg_coeffs = pd.read_csv(fpath_reg_coefs)

    # Set the index to the first column
    reg_coeffs.set_index("Unnamed: 0", inplace=True)

    # Loop over the columns in the DataFrame
    for reg_col in reg_coeffs.columns:
        if reg_col != "Unnamed: 0":
            # Split the column name by _regression
            # e.g. Austria
            country = reg_col.split("_regression")[0]

            # if df contains f{country}_hdd and f{country}_cdd
            if f"{country}_hdd" in df.columns and f"{country}_cdd" in df.columns:
                # Extract the time coefficient for col
                time_coeff = reg_coeffs.loc["time", reg_col]

                # Extract the hdd coefficient for col
                hdd_coeff = reg_coeffs.at[hdd_name, reg_col]

                # Extract the cdd coefficient for col
                cdd_coeff = reg_coeffs.at[cdd_name, reg_col]

                # print the coefficients
                # print(f"Time coefficient: {time_coeff}")
                # print(f"HDD coefficient: {hdd_coeff}")
                # print(f"CDD coefficient: {cdd_coeff}")

                # Calculate the demand
                df[f"{country}_demand"] = (
                    (time_coeff * demand_year)
                    + (hdd_coeff * df[f"{country}_hdd"])
                    + (cdd_coeff * df[f"{country}_cdd"])
                )

    return df


# Define a function to save the dataframe
def save_df(
    df: pd.DataFrame,
    fname: str,
    fdir: str = "/gws/nopw/j04/canari/users/benhutch/met_to_energy_dfs",
    ftype: str = "csv",
) -> None:
    """
    Save the DataFrame.

    Parameters
    ----------

    df: pd.DataFrame
        The DataFrame to save.

    fname: str
        The filename to save the DataFrame.

    fdir: str
        The directory to save the DataFrame.

    ftype: str
        The file type to save the DataFrame.

    Returns
    -------

    None

    """

    # If the file type is csv
    if ftype == "csv":
        # Save the DataFrame as a CSV
        df.to_csv(f"{fdir}/{fname}.csv")
    else:
        raise NotImplementedError(f"File type {ftype} not implemented.")

    return None


# define a main function for testing
def main():
    # set up the args
    model_variable = "tas"
    model = "HadGEM3-GC31-MM"
    init_years = np.arange(1960, 1965 + 1)
    experiment = "dcppA-hindcast"
    frequency = "day"

    # load the data
    ds = load_dcpp_data(
        model_variable=model_variable,
        model=model,
        init_years=init_years,
        experiment=experiment,
        frequency=frequency,
    )

    return None


# Run the main function
if __name__ == "__main__":
    main()
