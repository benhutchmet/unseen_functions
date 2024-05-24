# Functions for UNSEEN work

# Local imports
import os
import sys
import glob
import random
import re

# Third party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy import stats, signal
from scipy.stats import genextreme, linregress
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xesmf as xe
import matplotlib.cm as cm
import cftime

# Import types
from typing import Any, Callable, Union, List


# Path to modules
sys.path.append("/home/users/benhutch/unseen_multi_year/")
import dictionaries as dicts


# Function for loading each of the ensemble members for a given model
def load_model_data(
    model_variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    avg_period: int,
    grid: dict,
):
    """
    Function for loading each of the ensemble members for a given model

    Parameters
    ----------

    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation

    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'

    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Set up the directory where the csv files are stored
    csv_dir = "/home/users/benhutch/multi_year_unseen/paths"

    # Assert that the folder exists
    assert os.path.exists(csv_dir), "The csv directory does not exist"

    # Assert that the folder is not empty
    assert os.listdir(csv_dir), "The csv directory is empty"

    # Extract the csv file for the model and experiment
    csv_file = glob.glob(f"{csv_dir}/*.csv")[0]

    # Verify that the csv file exists
    assert csv_file, "The csv file does not exist"

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the model and experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == model_variable),
        "path",
    ].values[0]

    print(model_path)

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Create an empty list of files
    model_file_list = []

    no_members = 0

    # BADC pattern
    # /badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/tas/g?/files/d????????/*.nc"

    # Extract the first part of the model_path
    model_path_root = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root == "gws":
        print("The model path root is gws")
        # List the files in the model path
        model_files = os.listdir(model_path)

        # Loop over the years
        for year in years:
            # Find all of the files for the given year
            year_files = [file for file in model_files if f"s{year}" in file]

            # Find the filenames for the given year
            # After the final '/' in the path
            year_files_split = [file.split("/")[-1] for file in year_files]

            # Split by _ and extract the 4th element
            year_files_split = [file.split("_")[4] for file in year_files_split]

            # Split by - and extract the 1st element
            year_files_split = [file.split("-")[1] for file in year_files_split]

            # Find the unique members
            unique_combinations = np.unique(year_files_split)

            # # Print the year and the number of files
            # print(year, len(year_files))
            if year == years[0]:
                # Set the no members
                no_members = len(year_files)

            # # Print no
            # print("Number of members", no_members)
            # print("Number of unique combinations", len(unique_combinations))
            # print("Unique combinations", unique_combinations)

            # Assert that the len unique combinations is the same as the no members
            assert (
                len(unique_combinations) == no_members
            ), "The number of unique combinations is not the same as the number of members"

            # Assert that the number of files is the same as the number of members
            assert (
                len(year_files) == no_members
            ), "The number of files is not the same as the number of members"

            # Append the year files to the model file list
            model_file_list.append(year_files)
    elif model_path_root == "badc":
        print("The model path root is badc")

        # Loop over the years
        for year in years:
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # Extract the number of members
            # as the number of unique combinations of r*i*p?f?
            # here f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
            # List the directories in model_path
            dirs = os.listdir(model_path)

            # Split these by the delimiter '-'
            dirs_split = [dir.split("-") for dir in dirs]

            # Find the unique combinations of r*i*p?f?
            unique_combinations = np.unique(dirs_split)

            # Set the no members
            no_members = len(unique_combinations)

            # Assert that the number of files is the same as the number of members
            assert (
                len(year_files) == no_members
            ), "The number of files is not the same as the number of members"

            # Append the year files to the model file list
            model_file_list.append(year_files)

    # Flatten the model file list
    model_file_list = [file for sublist in model_file_list for file in sublist]

    # Print the number of files
    print("Number of files:", len(model_file_list))

    # Print
    print(f"opening {model_path}/{model_file_list[0]}")

    # From the first file extract the number of lats and lons
    ds = xr.open_dataset(f"{model_path}/{model_file_list[0]}")

    # Extract the time series for the gridbox
    ds = ds.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(dim=("lat", "lon"))

    # Print the first time of the first file
    print("First time:", ds["time"][0].values)

    # Extract the first year from the first file
    first_year = int(str(ds["time"][0].values)[:4])

    # Print the first year
    print("First year:", first_year)

    # Assert that the first year is the same as the start year
    assert first_year == start_year, "The first year is not the same as the start year"

    # Print the window over which we are slicing the time
    print("Slicing over:", f"{first_year}-12-01", f"{first_year + avg_period}-12-01")

    # Extract the time slice between
    # First december to second march
    ds_slice = ds.sel(
        time=slice(f"{first_year}-12-01", f"{first_year + avg_period}-12-01")
    )

    # Extract the nmonths
    n_months = len(ds_slice["time"])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    model_data = np.zeros([n_years, no_members, n_months])

    # Print the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Loop over the years
    for year in tqdm(years, desc="Processing years"):
        for member in tqdm(
            (unique_combinations),
            desc=f"Processing members for year {year}",
            leave=False,
        ):
            # Find the file for the given year and member
            file = [
                file
                for file in model_file_list
                if f"s{year}" in file and member in file
            ][0]

            # set the member index
            member_index = np.where(unique_combinations == member)[0][0]

            # Load the file
            ds = xr.open_dataset(f"{model_path}/{file}")

            # Extract the time series for the gridbox
            ds = ds.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
                dim=("lat", "lon")
            )

            # Extract the time slice between
            ds_slice = ds.sel(time=slice(f"{year}-12-01", f"{year + avg_period}-12-01"))

            # Extract the data
            model_data[year - start_year, member_index, :] = ds_slice[
                model_variable
            ].values

    # p[rint the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Return the model data
    return model_data


# Define a function for preprocessing the model data
def preprocess(
    ds: xr.Dataset,
):
    """
    Preprocess the model data using xarray

    Parameters

    ds: xr.Dataset
        The dataset to preprocess
    """

    # Return the dataset
    return ds


# Write a new function for loading the model data using xarray
def load_model_data_xarray(
    model_variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    grid: dict,
    first_fcst_year: int,
    last_fcst_year: int,
    months: list,
    frequency: str = "Amon",
    engine: str = "netcdf4",
    parallel: bool = True,
):
    """
    Function for loading each of the ensemble members for a given model using xarray

    Parameters
    ----------

    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation

    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'

    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    first_fcst_year: int
        The first forecast year for taking the time average
        E.g. 1960

    last_fcst_year: int
        The last forecast year for taking the time average
        E.g. 1962

    months: list
        The months to take the time average over
        E.g. [10, 11, 12, 1, 2, 3] for October to March

    frequency: str
        The frequency of the data
        Defaults to 'mon'

    engine: str
        The engine to use for opening the dataset
        Passed to xarray.open_mfdataset
        Defaults to 'netcdf4'

    parallel: bool
        Whether to use parallel processing
        Passed to xarray.open_mfdataset
        Defaults to True

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Extract the lat and lon bounds
    lon1, lon2, lat1, lat2 = grid["lon1"], grid["lon2"], grid["lat1"], grid["lat2"]

    # Set up the path to the csv file
    csv_path = "paths/*.csv"

    # Find the csv file
    csv_file = glob.glob(csv_path)[0]

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the given model and experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == model_variable)
        & (csv_data["frequency"] == frequency),
        "path",
    ].values[0]

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Extract the first part of the model_path
    model_path_root = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root == "gws":
        print("The model path root is gws")

        # List the files in the model path
        model_files = os.listdir(model_path)

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Find all of the files for the given year
            year_files = [file for file in model_files if f"s{year}" in file]

            # Split the year files by '/'
            year_files_split = [file.split("/")[-1] for file in year_files]

            # Split the year files by '_'
            year_files_split = [file.split("_")[4] for file in year_files_split]

            # Split the year files by '-'
            year_files_split = [file.split("-")[1] for file in year_files_split]

            # Find the unique combinations
            unique_combinations = np.unique(year_files_split)

            # Assert that the len unique combinations is the same as the no members
            assert len(unique_combinations) == len(
                year_files
            ), "The number of unique combinations is not the same as the number of members"

    elif model_path_root == "badc":
        print("The model path root is badc")

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/{frequency}/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # Extract the number of members
            # as the number of unique combinations of r*i*p?f?
            # here f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
            # List the directories in model_path
            dirs = os.listdir(model_path)

            # Regular expression pattern for the desired format
            pattern = re.compile(r"s\d{4}-(r\d+i\d+p\d+f\d+)")

            # Extract the 'r*i?p?f?' part from each directory name
            extracted_parts = [
                pattern.match(dir).group(1) for dir in dirs if pattern.match(dir)
            ]

            # PRINT THE EXTRACTED PARTS
            # print("Extracted parts:", extracted_parts)

            # Find the unique combinations of r*i*p?f?
            unique_combinations = np.unique(extracted_parts)

            # FIXME: daily HadGEM will have like 10 members split into diff times
            # # Assert that the number of files is the same as the number of members
            # assert len(year_files) == len(
            #     unique_combinations
            # ), "The number of files is not the same as the number of members"
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Set up unique variant labels
    unique_variant_labels = np.unique(unique_combinations)

    # Print the number of unique variant labels
    print("Number of unique variant labels:", len(unique_variant_labels))
    print("For model:", model)

    # print the first 5 unique variant labels
    print("First 10 unique variant labels:", unique_variant_labels[:10])

    # Create an empty list for forming the list of files for each ensemble member
    member_files = []

    # If the model path root is gws
    if model_path_root == "gws":
        print("Forming the list of files for each ensemble member for gws")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Find the file for the given year and member
                file = [
                    file
                    for file in model_files
                    if f"s{year}" in file and variant_label in file
                ][0]

                # Append the model path to the file
                file = f"{model_path}/{file}"

                # Append the file to the member files
                variant_label_files.append(file)

            # Append the member files to the member files
            member_files.append(variant_label_files)
    elif model_path_root == "badc":
        print("Forming the list of files for each ensemble member for badc")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Form the path to the files for this year
                path = f"{model_path}/s{year}-{variant_label}/{frequency}/{model_variable}/g?/files/d????????/*.nc"

                # Find the files which match the path
                year_files = glob.glob(path)

                # # Assert that the number of files is 1
                # if len(year_files) == 1:
                #     # print that only one file was found
                #     print(f"Only one file found for {year} and {variant_label}")
                # elif len(year_files) > 1:
                #     # print that more than one file was found
                #     print(f"{len(year_files)} found for {year} and {variant_label}")
                # else:
                #     # print that no files were found
                #     # print(f"No files found for {year} and {variant_label}")

                #     # print that we are exiting
                #     sys.exit()

                # Append the files to the variant label files
                variant_label_files.append(year_files)

            # Append the variant label files to the member files
            member_files.append(variant_label_files)
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Assert that member files is a list withiin a list
    assert isinstance(member_files, list), "member_files is not a list"

    # Assert that member files is a list of lists
    assert isinstance(member_files[0], list), "member_files is not a list of lists"

    # Assert that the length of member files is the same as the number of unique variant labels
    assert len(member_files) == len(
        unique_variant_labels
    ), "The length of member_files is not the same as the number of unique variant labels"

    # Initialize the model data
    dss = []

    # Will depend on the model here
    # for s1961 - CanESM5 and IPSL-CM6A-LR both initialized in January 1962
    # So 1962 will be their first year
    if model not in ["CanESM5", "IPSL-CM6A-LR"]:
        # Find the index of the forecast first year
        first_fcst_year_idx = first_fcst_year - start_year
        last_fcst_year_idx = (last_fcst_year - first_fcst_year) + 1
    else:
        # Find the index of the forecast first year
        # First should be index 0 normally
        first_fcst_year_idx = (first_fcst_year - start_year) - 1
        last_fcst_year_idx = last_fcst_year - first_fcst_year

    # print the shape of member files
    print("Shape of member files:", np.shape(member_files))

    # if member_files has shape (x, y)
    if len(np.shape(member_files)) == 2:
        # Flatten the member files list
        member_files = [file for sublist in member_files for file in sublist]
    elif len(np.shape(member_files)) == 3:
        # Flatten the member files list
        member_files = [
            file
            for sublist1 in member_files
            for sublist2 in sublist1
            for file in sublist2
        ]

    # print the shape of flattened member files
    print("Shape of flattened member files:", np.shape(member_files))

    init_year_list = []
    # Loop over init_years
    for init_year in tqdm(
        range(start_year, end_year + 1), desc="Processing init years"
    ):
        print(f"processing init year {init_year}")
        # Set up the member list
        member_list = []
        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Find the matching path for the given year and member
            # e.g file containing f"s{init_year}-{variant_label}
            files = [
                file for file in member_files if f"s{init_year}-{variant_label}" in file
            ]

            # # print how many files were found
            # print(f"Found {len(files)} files for {init_year} and {variant_label}")

            # Open all leads for specified variant label
            # and init_year
            member_ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="time",
                preprocess=lambda ds: preprocess(ds),
                parallel=parallel,
                engine=engine,
                coords="minimal",  # expecting identical coords
                data_vars="minimal",  # expecting identical vars
                compat="override",  # speed up
            ).squeeze()

            # init_year = start_year and variant_label is unique_variant_labels[0]
            if init_year == start_year and variant_label == unique_variant_labels[0]:
                # Set new int time
                member_ds = set_integer_time_axis(
                    xro=member_ds, frequency=frequency, first_month_attr=True
                )
            else:
                # Set new integer time
                member_ds = set_integer_time_axis(member_ds, frequency=frequency)

            # Append the member dataset to the member list
            member_list.append(member_ds)
        # Concatenate the member list along the ensemble_member dimension
        member_ds = xr.concat(member_list, "member")
        # Append the member dataset to the init_year list
        init_year_list.append(member_ds)
    # Concatenate the init_year list along the init dimension
    # and rename as lead time
    ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

    # Set up the members
    ds["member"] = unique_variant_labels
    ds["init"] = np.arange(start_year, end_year + 1)

    # Return ds
    return ds


def set_integer_time_axis(
    xro: Union[xr.DataArray, xr.Dataset],
    frequency: str = "Amon",
    offset: int = 1,
    time_dim: str = "time",
    first_month_attr: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Set time axis to integers starting from `offset`.

    Used in hindcast preprocessing before the concatenation of `intake-esm` happens.

    Inputs:
    xro: xr.DataArray or xr.Dataset
        The input xarray DataArray or Dataset whose time axis is to be modified.

    frequency: str, optional
        The frequency of the data. Default is "Amon".

    offset: int, optional
        The starting point for the new integer time axis. Default is 1.

    time_dim: str, optional
        The name of the time dimension in the input xarray object. Default is "time".

    first_month_attr: bool, optional
        Whether to include the first month as an attribute in the dataset.
        Default is False.

    Returns:
    xr.DataArray or xr.Dataset
        The input xarray object with the time axis set to integers starting from `offset`.
    """

    if first_month_attr:
        # Extract the first forecast year-month pair
        first_month = xro[time_dim].values[0]

        # Add the first month as an attribute to the dataset
        xro.attrs["first_month"] = str(first_month)

        # add an attribute for the type of the time axis
        xro.attrs["time_axis_type"] = type(first_month).__name__

    xro[time_dim] = np.arange(offset, offset + xro[time_dim].size)
    return xro


# Define a function to perform regridding to regular -180 to 180 grid
def regrid_ds(
    ds: xr.Dataset,
    variable: str,
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    rg_algo: str = "bilinear",
    periodic: bool = True,
) -> xr.Dataset:
    """
    Regrid the input dataset to a regular grid with specified bounds.

    Inputs:
    ds: xr.Dataset
        The input xarray Dataset to be regridded.

    variable: str
        The name of the variable to be regridded.

    grid_bounds: list[float], optional
        The bounds of the regular grid to which the input dataset is regridded.
        Default is [-180.0, 180.0, -90.0, 90.0].

    rg_algo: str, optional
        The regridding algorithm to be used. Default is "bilinear".

    periodic: bool, optional
        Whether the input data is on a periodic grid. Default is True.

    Returns:
    xr.Dataset
        The regridded xarray Dataset.
    """

    # Calculate the resolution of the input dataset
    lat_res = (ds["lat"].max() - ds["lat"].min()) / (ds["lat"].count() - 1.0).values
    lon_res = (ds["lon"].max() - ds["lon"].min()) / (ds["lon"].count() - 1.0).values

    # Set up the 2d grid
    ds_out = xe.util.grid_2d(
        grid_bounds[0], grid_bounds[1], lon_res, grid_bounds[2], grid_bounds[3], lat_res
    )

    # Set up the regridder
    regridder = xe.Regridder(
        ds,
        ds_out,
        rg_algo,
        periodic=periodic,
    )

    # Perform the regridding
    da_regridded = regridder(ds[variable])

    # Return the regridded dataset
    return da_regridded


# Define a function to select a gridbox
def select_gridbox(
    ds: xr.Dataset,
    grid: dict[str, float],
    dim: tuple[str, str] = ("y", "x"),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.Dataset:
    """
    Select the gridbox from the input dataset and calculate the mean over it.

    Inputs:
    ds: xr.Dataset
        The input xarray Dataset from which the gridbox is to be selected.

    grid: dict[str, float]
        The dictionary containing the latitudinal and longitudinal bounds of the gridbox.

    dim: tuple[str, str], optional
        The dimensions along which the mean is calculated. Default is ("y", "x").

    lat_name: str, optional
        The name of the latitude coordinate in the input dataset. Default is "lat".

    lon_name: str, optional
        The name of the longitude coordinate in the input dataset. Default is "lon".

    Returns:
    xr.Dataset
        The dataset containing the mean over the selected gridbox.
    """

    # Extract the latitudinal and longitudinal bounds
    lon1, lon2, lat1, lat2 = grid["lon1"], grid["lon2"], grid["lat1"], grid["lat2"]

    # Assert that the latitudinal and longitudinal bounds for ds are -180 to 180
    assert (
        ds[lon_name].min().values >= -180.0 and ds[lon_name].max().values <= 180.0
    ), "The longitudinal bounds for the dataset are not -180 to 180"
    # Assert that the latitudinal bounds for ds are -90 to 90
    assert (
        ds[lat_name].min().values >= -90.0 and ds[lat_name].max().values <= 90.0
    ), "The latitudinal bounds for the dataset are not -90 to 90"

    # Set up the mask
    mask = (
        (ds[lat_name] >= lat1)
        & (ds[lat_name] <= lat2)
        & (ds[lon_name] >= lon1)
        & (ds[lon_name] <= lon2)
    )

    # Mask the dataset
    ds_masked = ds.where(mask)

    # Calculate the mean of the masked dataset
    ds_mean = ds_masked.mean(dim=dim)

    # Return the mean dataset
    return ds_mean


# define a function to load and regrid the observations
def load_regrid_obs(
    model_ds: xr.Dataset,
    obs_variable: str,
    obs_path: str,
    start_year: int,
    end_year: int,
    months: list[int],
    grid: dict[str, float],
    rg_algo: str = "bilinear",
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    periodic: bool = True,
    aggregate_worst_months: bool = False,
) -> xr.Dataset:
    """
    Load and regrid the observations to a regular grid with specified bounds.

    Inputs:

    model_ds: xr.Dataset
        The input xarray Dataset to be regridded.

    obs_variable: str
        The variable to load from the observations.

    obs_path: str
        The path to the observations.

    start_year: int
        The start year for the data.

    end_year: int
        The end year for the data.

    months: list[int]
        The months to take the time average over.
        In format [10, 11, 12, 1, 2, 3] for October to March.

    grid: dict[str, float]
        The dictionary containing the latitudinal and longitudinal bounds of the gridbox.

    rg_algo: str, optional
        The regridding algorithm to be used. Default is "bilinear".

    grid_bounds: list[float], optional
        The bounds of the regular grid to which the input dataset is regridded.

    periodic: bool, optional
        Whether the input data is on a periodic grid. Default is True.

    aggregate_worst_months: bool, optional
        Whether to aggregate the worst months. Default is False.

    Returns:

    obs_data: pd.DataFrame
        The regridded observations in a pandas DataFrame.

    """

    # Extract the latitudinal and longitudinal bounds
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Calculate the resoluion of the input dataset
    lat_res = (model_ds["lat"].max() - model_ds["lat"].min()) / (
        model_ds["lat"].count() - 1.0
    ).values
    lon_res = (model_ds["lon"].max() - model_ds["lon"].min()) / (
        model_ds["lon"].count() - 1.0
    ).values

    # Set up the 2d grid
    ds_out = xe.util.grid_2d(
        grid_bounds[0], grid_bounds[1], lon_res, grid_bounds[2], grid_bounds[3], lat_res
    )

    # Open the observations
    obs = xr.open_mfdataset(
        obs_path,
        combine="by_coords",
        parallel=True,
    )

    # restrict to between the start and end years
    obs = obs.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # If expver is present in the observations
    if "expver" in obs.coords:
        # Combine the first two expver variables
        obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

    # print(obs.coords)
    # print(ds_out.coords)

    # print(obs.longitude.min(), obs.longitude.max())
    # print(obs.latitude.min(), obs.latitude.max())
    # print(ds_out.lon.min(), ds_out.lon.max())
    # print(ds_out.lat.min(), ds_out.lat.max())

    # Convert the lon and lat to 1D
    ds_out["lon"] = ds_out["lon"].mean(dim="y")
    ds_out["lat"] = ds_out["lat"].mean(dim="x")

    # Set up the regriidder
    regridder = xe.Regridder(
        obs,
        ds_out,
        rg_algo,
        periodic=periodic,
    )

    # Perform the regridding
    obs_rg = regridder(obs[obs_variable])

    # print the coordinates
    print("Coordinates of the regridded obs:", obs_rg.coords)

    # Extract the time series for the gridbox
    obs_rg = select_gridbox(
        ds=obs_rg,
        grid=grid,
    )

    # If the type of time is numpy.datetime64
    if isinstance(obs_rg["time"].values[0], np.datetime64):
        # Convert numpy.datetime64 to datetime
        obs_rg["time"] = pd.to_datetime(obs_rg["time"].values)

    # If the months list contains [* 12, 1 *]
    if cross_year(months):
        print("Crosses the year boundary")
        # Select the months
        obs_rg = obs_rg.sel(time=obs_rg["time.month"].isin(months))

        # Shift the time back by months[-1]
        # e.g. if months = [12, 1] then shift back by 1 month
        # and take the annual mean
        obs_rg = obs_rg.shift(time=-months[-1], fill_value=np.nan)

        # Remove the first months[-1] values
        obs_rg = obs_rg.isel(time=slice(months[-1], None))

        # if not aggregate_worst_months:
        if not aggregate_worst_months:
            print("Not aggregating the worst months")
            # Calculate the annual mean
            obs_rg = obs_rg.resample(time="Y").mean()
        else:
            print("Aggregating the lowest wind speed month for each winter")

            # Select the month with the lowest value for each winter
            # group by month and year, then select the month with the lowest value
            obs_rg = (
                obs_rg.groupby("time.year")
                .apply(lambda x: x.where(x == x.min().compute(), drop=True))
                .resample(time="Y")
                .mean()
            )

    else:
        # Select the months
        obs_rg = obs_rg.sel(time=obs_rg["time.month"].isin(months))

        # If not aggregate_worst_months
        if not aggregate_worst_months:
            print("Not aggregating the worst months")
            # Calculate the annual mean
            obs_rg = obs_rg.resample(time="Y").mean()
        else:
            print("Aggregating the lowest wind speed month for each winter")

            # Select the month with the lowest value for each winter
            # group by month and year, then select the month with the lowest value
            obs_rg = (
                obs_rg.groupby("time.year")
                .apply(lambda x: x.where(x == x.min().compute(), drop=True))
                .resample(time="Y")
                .mean()
            )

    # Ensure the DataArray has a name
    obs_rg.name = obs_variable

    # Convert to a pandas dataframe
    # with columns 'year' and 'value'
    obs_df = obs_rg.to_dataframe().reset_index()

    # Extract the years
    years = obs_df["time"].dt.year.values

    # Extract the values
    values = obs_df[obs_variable].values

    # Form the obs dataframes
    obs_data = pd.DataFrame({"year": years, "value": values})

    # Return the obs data
    return obs_data


# Define a function for crossing year
def cross_year(
    list: list[int],
) -> bool:
    """
    Check if the list crosses the year boundary.

    Inputs:
    list: list[int]
        The list of months.

    Returns:
    bool
        Whether the list crosses the year boundary.
    """

    # Loop over the list
    for i in range(len(list) - 1):
        # If the difference between the months is negative
        if list[i] == 12 and list[i + 1] == 1:
            return True
    return False


# Function for loading the observations
def load_obs_data(
    obs_variable: str,
    regrid_obs_path: str,
    start_year: int,
    end_year: int,
    avg_period: int,
    grid: dict,
):
    """
    Function for loading the observations

    Parameters
    ----------

    obs_variable: str
        The variable to load from the model data
        E.g. 'si10' for sfcWind

    regrid_obs_path: str
        The path to the regridded observations

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns

    obs_data: np.array
        The observations
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set up the new years
    new_years = []

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Open the obs
    obs = xr.open_mfdataset(regrid_obs_path, combine="by_coords", parallel=True)[
        obs_variable
    ]

    # Combine the first two expver variables
    obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

    # Extract the time series for the gridbox
    obs = obs.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(dim=("lat", "lon"))

    # Convert numpy.datetime64 to datetime
    final_time = obs["time"][-1].values.astype(str)

    # Extract the year and month
    final_year = int(final_time[:4])
    final_month = int(final_time[5:7])

    # If the final time is not november or december
    if not (final_month == 11 or final_month == 12):
        # Check that the final year - avg_period is not less than the end year
        if (final_year - 1) - avg_period < end_year:
            # Set the end year to the final year - avg_period
            end_year = (final_year - 1) - avg_period
    else:
        print("The final year has november or december")

    # Set the new years
    new_years = np.arange(start_year, end_year + 1)

    # Print the first time of the new years
    print("First time:", new_years[0])
    print("Last time:", new_years[-1])

    # Print the years we are slicing over
    print("Slicing over:", f"{start_year}-12-01", f"{start_year + avg_period}-03-30")

    # Extract the time slice between
    obs_slice = obs.sel(
        time=slice(f"{start_year}-12-01", f"{start_year + avg_period}-11-30")
    )

    # Extract the nmonths
    n_months = len(obs_slice["time"])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    obs_data = np.zeros([len(new_years), n_months])

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Loop over the years
    for year in tqdm(new_years, desc="Processing years"):
        # We only have obs upt to jjuly 2023

        # Extract the time slice between
        obs_slice = obs.sel(time=slice(f"{year}-12-01", f"{year + avg_period}-11-30"))

        # Extract the data
        obs_data[year - start_year, :] = obs_slice.values

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Set up the obs years
    obs_years = np.arange(new_years[0], new_years[-1] + 1)

    # Return the obs data
    return obs_data, obs_years


# Function for calculating the obs_stats
def calculate_obs_stats(
    obs_data: np.ndarray, start_year: int, end_year: int, avg_period: int, grid: dict
):
    """
    Calculate the observations stats

    Parameters
    ----------

        obs_data: np.ndarray
            The observations data
            With shape (nyears, nmonths)

        start_year: int
            The start year for the data
            E.g. 1961

        end_year: int
            The end year for the data
            E.g. 1990

        avg_period: int
            The number of years to average over
            E.g. 1 for 1-year, 5 for 5-year, etc.

        grid: dict
            The grid to load the data over

    Returns
    -------

        obs_stats: dict
            A dictionary containing the obs stats

    """

    # Define the mdi
    mdi = -9999.0

    # Define the obs stats
    obs_stats = {
        "avg_period_mean": [],
        "mean": mdi,
        "sigma": mdi,
        "skew": mdi,
        "kurt": mdi,
        "start_year": mdi,
        "end_year": mdi,
        "avg_period": mdi,
        "grid": mdi,
        "min_20": mdi,
        "max_20": mdi,
        "min_10": mdi,
        "max_10": mdi,
        "min_5": mdi,
        "max_5": mdi,
        "min": mdi,
        "max": mdi,
        "sample_size": mdi,
    }

    # Set the start year
    obs_stats["start_year"] = start_year

    # Set the end year
    obs_stats["end_year"] = end_year

    # Set the avg period
    obs_stats["avg_period"] = avg_period

    # Set the grid
    obs_stats["grid"] = grid

    # Process the obs
    obs_copy = obs_data.copy()

    # Take the mean over the 1th axis (i.e. over the 12 months)
    obs_year = np.mean(obs_copy, axis=1)

    # Set the average period mean
    obs_stats["avg_period_mean"] = obs_year

    # Get the sample size
    obs_stats["sample_size"] = len(obs_year)

    # Take the mean over the 0th axis (i.e. over the years)
    obs_stats["mean"] = np.mean(obs_year)

    # Take the standard deviation over the 0th axis (i.e. over the years)
    obs_stats["sigma"] = np.std(obs_year)

    # Take the skewness over the 0th axis (i.e. over the years)
    obs_stats["skew"] = stats.skew(obs_year)

    # Take the kurtosis over the 0th axis (i.e. over the years)
    obs_stats["kurt"] = stats.kurtosis(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min"] = np.min(obs_year)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max"] = np.max(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_5"] = np.percentile(obs_year, 5)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_5"] = np.percentile(obs_year, 95)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_10"] = np.percentile(obs_year, 10)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_10"] = np.percentile(obs_year, 90)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_20"] = np.percentile(obs_year, 20)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_20"] = np.percentile(obs_year, 80)

    # Return the obs stats
    return obs_stats


# Write a function which does the plotting
def plot_events(
    model_data: np.ndarray,
    obs_data: np.ndarray,
    obs_stats: dict,
    start_year: int,
    end_year: int,
    bias_adjust: bool = True,
    figsize_x: int = 10,
    figsize_y: int = 10,
    do_detrend: bool = False,
):
    """
    Plots the events on the same axis.

    Parameters
    ----------

    model_data: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    obs_data: np.ndarray
        The observations data
        With shape (nyears, nmonths)

    obs_stats: dict
        A dictionary containing the obs stats

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    bias_adjust: bool
        Whether to bias adjust the model data
        Default is True

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    do_detrend: bool
        Whether to detrend the data
        Default is False

    Returns
    -------
    None
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    if len(model_data.shape) == 3:
        # Take the mean over the 2th axis (i.e. over the months)
        # For the model data
        model_year = np.mean(model_data, axis=2)
    else:
        # For the model data
        model_year = model_data

    if len(obs_data.shape) == 2:
        # Take the mean over the 1th axis (i.e. over the members)
        # For the obs data
        obs_year = np.mean(obs_data, axis=1)
    else:
        # For the obs data
        obs_year = obs_data

    # if the bias adjust is True
    if bias_adjust:
        print("Bias adjusting the model data")

        # Flatten the model data
        model_flat = model_year.flatten()

        # Find the difference between the model and obs
        bias = np.mean(model_flat) - np.mean(obs_year)

        # Add the bias to the model data
        model_year = model_year - bias

    # If the detrend is True
    if do_detrend:
        print("Detrending the data")

        # Use the scipy detrend function
        model_year = signal.detrend(model_year, axis=0)

        # Use the scipy detrend function
        obs_year = signal.detrend(obs_year, axis=0)

        # Calculate the new minimum for the obs
        obs_stats["min"] = np.min(obs_year)

        # Calculate the 20th percentile for the obs
        obs_stats["min_20"] = np.percentile(obs_year, 20)

    # Set the figure size
    plt.figure(figsize=(figsize_x, figsize_y))

    # Plot the model data
    for i in range(model_year.shape[1]):

        # Separate data into two groups based on the condition
        below_20th = model_year[:, i] < obs_stats["min_20"]
        above_20th = ~below_20th

        # Plot points below the 20th percentile with a label
        plt.scatter(
            years[below_20th],
            model_year[below_20th, i],
            color="blue",
            alpha=0.8,
            label="model wind drought" if i == 0 else None,
        )

        # Plot points above the 20th percentile without a label
        plt.scatter(
            years[above_20th],
            model_year[above_20th, i],
            color="grey",
            alpha=0.8,
            label="HadGEM3-GC31-MM" if i == 0 else None,
        )

    # Plot the obs
    plt.scatter(years, obs_year, color="k", label="ERA5")

    # Plot the 20th percentile
    plt.axhline(obs_stats["min_20"], color="black", linestyle="-")

    # Plot the min
    plt.axhline(obs_stats["min"], color="black", linestyle="--")

    # Add a legend in the upper left
    plt.legend(loc="upper left")

    # Add the axis labels
    plt.xlabel("Year")

    # Add the axis labels
    plt.ylabel("Average Wind speed (m/s)")

    # Show the plot
    plt.show()


# Write a function which does the bootstrapping to calculate the statistics
def model_stats_bs(model: np.ndarray, nboot: int = 10000) -> dict:
    """
    Repeatedly samples the model data with replacement across its members to
    produce many samples equal in length to the reanalysis time series. This
    gives a single pseudo-time series from which the moments of the distribution
    can be calculated. The process is repeated to give a distribution of the
    moments.

    Parameters
    ----------

    model: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    nboot: int
        The number of bootstrap samples to take
        Default is 10000

    Returns
    -------

    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'
    """

    # Set up the model stats
    model_stats = {"mean": [], "sigma": [], "skew": [], "kurt": []}

    # Set up the number of years
    n_years = model.shape[0]

    # Set up the number of members
    n_members = model.shape[1]

    # TODO: Does autocorrelation need to be accounted for?
    # If so, use a block bootstrap

    # Set up the arrays
    mean_boot = np.zeros(nboot)
    sigma_boot = np.zeros(nboot)

    skew_boot = np.zeros(nboot)
    kurt_boot = np.zeros(nboot)

    # Create the indexes for the ensemble members
    index_ens = range(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(np.arange(nboot)):
        # print(f"Bootstrapping {iboot + 1} of {nboot}")

        # Create the index for time
        ind_time_this = range(0, n_years)

        # Create an empty array to store the data
        model_boot = np.zeros([n_years])

        # Set the year index
        year_index = 0

        # Loop over the years
        for itime in ind_time_this:

            # Select a random ensemble member
            ind_ens_this = random.choices(index_ens)

            # Logging
            # print(f"itime is {itime} of {n_years}")
            # print(f"year_index is {year_index} of {n_years} "
            #       f"iboot is {iboot} of {nboot} "
            #       f"ind_ens_this is {ind_ens_this}")

            # Extract the data
            model_boot[year_index] = model[itime, ind_ens_this]

            # Increment the year index
            year_index += 1

        # Calculate the mean
        mean_boot[iboot] = np.mean(model_boot)

        # Calculate the sigma
        sigma_boot[iboot] = np.std(model_boot)

        # Calculate the skew
        skew_boot[iboot] = stats.skew(model_boot)

        # Calculate the kurtosis
        kurt_boot[iboot] = stats.kurtosis(model_boot)

    # Append the mean to the model stats
    model_stats["mean"] = mean_boot

    # Append the sigma to the model stats
    model_stats["sigma"] = sigma_boot

    # Append the skew to the model stats
    model_stats["skew"] = skew_boot

    # Append the kurt to the model stats
    model_stats["kurt"] = kurt_boot

    # Return the model stats
    return model_stats


# Write a function which plots the four moments
def plot_moments(
    model_stats: dict,
    obs_stats: dict,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the four moments of the distribution of the model data and the
    observations.

    Parameters
    ----------

    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'

    obs_stats: dict
        A dictionary containing the obs stats

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Output
    ------

    None
    """

    # Set up the figure as a 2x2
    fig, axs = plt.subplots(2, 2, figsize=(figsize_x, figsize_y))

    ax1, ax2, ax3, ax4 = axs.ravel()

    # Plot the mean
    ax1.hist(model_stats["mean"], bins=100, density=True, color="red", label="model")

    # Plot the mean of the obs
    ax1.axvline(obs_stats["mean"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs mean in the distribution
    obs_mean_pos = stats.percentileofscore(model_stats["mean"], obs_stats["mean"])

    # Add a title
    ax1.set_title(f"Mean, {obs_mean_pos:.2f}%")

    # Include a textbox in the top right corner
    ax1.text(
        0.95,
        0.95,
        "a)",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the skewness
    ax2.hist(model_stats["skew"], bins=100, density=True, color="red", label="model")

    # Plot the skewness of the obs
    ax2.axvline(obs_stats["skew"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs skewness in the distribution
    obs_skew_pos = stats.percentileofscore(model_stats["skew"], obs_stats["skew"])

    # Add a title
    ax2.set_title(f"Skewness, {obs_skew_pos:.2f}%")

    # Include a textbox in the top right corner
    ax2.text(
        0.95,
        0.95,
        "b)",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the kurtosis
    ax3.hist(model_stats["kurt"], bins=100, density=True, color="red", label="model")

    # Plot the kurtosis of the obs
    ax3.axvline(obs_stats["kurt"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs kurtosis in the distribution
    obs_kurt_pos = stats.percentileofscore(model_stats["kurt"], obs_stats["kurt"])

    # Add a title
    ax3.set_title(f"Kurtosis, {obs_kurt_pos:.2f}%")

    # Include a textbox in the top right corner
    ax3.text(
        0.95,
        0.95,
        "c)",
        transform=ax3.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the sigma
    ax4.hist(model_stats["sigma"], bins=100, density=True, color="red", label="model")

    # Plot the sigma of the obs
    ax4.axvline(obs_stats["sigma"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs sigma in the distribution
    obs_sigma_pos = stats.percentileofscore(model_stats["sigma"], obs_stats["sigma"])

    # Add a title
    ax4.set_title(f"Standard deviation, {obs_sigma_pos:.2f}%")

    # Include a textbox in the top right corner
    ax4.text(
        0.95,
        0.95,
        "d)",
        transform=ax4.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    return


# Write a function to plot the distribution of the model and obs data
def plot_distribution(
    model_data: dict,
    obs_data: dict,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the distribution of the model and obs data

    Parameters
    ----------

    model_data: dict
        A dictionary of the model data

    obs_data: dict
        A dictionary containing the obs data

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Assemble the model data into a continuous time series
    model_data_flat = model_data.flatten()

    # plot the model data
    sns.distplot(model_data_flat, label="model", color="red")

    # Plot the obs data
    sns.distplot(obs_data.mean(axis=1), label="obs", color="black")

    # Include a textbox with the sample size
    plt.text(
        0.05,
        0.90,
        f"model N = {model_data_flat.shape[0]}\n" f"obs N = {obs_data.shape[0]}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Add a legend
    plt.legend()

    # Add a title
    # TODO: hard coded title
    plt.title("Distribution of 10m wind speed")

    return


# Define a function to take the average over leadtimes
def lead_time_avg(
    ds: xr.Dataset,
    months: List[int],
    first_month: str,
    lead_time_dim: str = "lead",
    lead_time_units: str = "months",
) -> xr.Dataset:
    """
    Take the average over the lead times for the specified months.

    Parameters
    ----------

    ds: xr.Dataset
        The input xarray Dataset to be averaged over the lead times.

    months: list[int]
        The months to average over.

    first_month: str
        The first month of the lead time in format "YYYY-MM-DD".

    lead_time_dim: str
        The name of the lead time dimension in the input xarray object.
        Default is "lead".

    Returns
    -------

    ds_avg: xr.Dataset
        The dataset containing the average over the specified months.
    """

    # Ensure that the first month is formatted as a datetime object
    first_month = pd.to_datetime(first_month)

    # Print the first month
    print(f"The first month is {first_month}")

    # Create a list to store the lead times in months
    lead_times = []

    # Loop over the lead times
    for i in range(len(ds[lead_time_dim])):
        # Calculate the lead time in months
        lead_time = first_month + pd.DateOffset(months=i)

        # Append the lead time to the list
        lead_times.append(lead_time)

    # Print the last month
    print(f"The last month is {lead_times[-1]}")

    # Create a mask for the specified months
    mask = [lead_time.month in months for lead_time in lead_times]

    # Count the number of sequences of True which have
    # length equal to the number of months in the mask list
    sequences = get_sequences(mask, len(months))

    # Print the number of sequences
    print(f"The number of sequences is {len(sequences)}")

    # Create a list of ints to store the indices of the sequences
    new_lead = np.arange(1, len(sequences) + 1)

    # Create a list to store the average over the lead times
    avg_seq = []

    # Loop over the sequences
    for i, sequence in enumerate(sequences):
        # Print the length of the sequence
        print(f"The length of sequence {i} is {len(sequence)}")

        # Extract the indices of the sequence
        idx = [i for _, i in sequence]

        # print the indices
        print(f"The indices of sequence {i} are {idx}")

        # Take the mean over the sequence of lead
        seq_mean = ds.isel(lead=idx).mean(dim=lead_time_dim)

        # Append the mean to avg_seq
        avg_seq.append(seq_mean)

    # Convert avg_seq to a DataArray
    avg_seq_da = xr.concat(avg_seq, dim=lead_time_dim)

    # Set up the values of the new lead dimension
    avg_seq_da["lead"] = new_lead

    # Return the dataset
    return avg_seq_da

# define a function to select the months and extract the days
def select_months(
        ds: xr.Dataset,
        months: List[int],
        first_day: str,
        first_time: str,
        frequency: str,
        lead_time_dim: str = "lead",
        time_axis_type: str = "Datetime360Day",
) -> xr.Dataset:
    """
    Selects the months of interests and extracts the days of the month as a lead time variable.
    
    Parameters
    ----------
    
    ds: xr.Dataset
        The input xarray Dataset to be averaged over the lead times.
        
    months: list[int]
        The months to average over.
        
    first_day: str
        The first month of the lead time in format "YYYY-MM-DD".
        
    first_time: str
        The first time of the dataset in format "HH:MM:SS".

    frequency: str
        The frequency of the time axis.
        E.g. "day" or "Amon".

    lead_time_dim: str
        The name of the lead time dimension in the input xarray object.
        Default is "lead".

    time_axis_type: str
        The type of time axis.
        Default is "Datetime360Day".

    Returns
    -------

    ds_avg: xr.Dataset
        The dataset for the specified months with lead as the days of the month.
    """

    # Ensure that the first month is formatted as a datetime object
    first_day = pd.to_datetime(first_day)

    # print the first day
    print(f"The first day is {first_day}")

    # Create a list to store the lead times in days
    dates = []

    # Loop over the lead times
    for i in range(len(ds[lead_time_dim])):
        if time_axis_type == "Datetime360Day":
            # Calculate the number of years, months, and days to add
            years_to_add = i // 360
            months_to_add = (i % 360) // 30
            days_to_add = (i % 360) % 30

            # Calculate the new date
            new_year = first_day.year + years_to_add
            new_month = first_day.month + months_to_add
            new_day = first_day.day + days_to_add

            # Handle overflow of days
            if new_day > 30:
                new_day -= 30
                new_month += 1

            # Handle overflow of months
            if new_month > 12:
                new_month -= 12
                new_year += 1

            new_date = cftime.Datetime360Day(new_year, new_month, new_day)

            # Add the new date to the list
            dates.append(new_date)
        else:
            raise NotImplementedError("Only Datetime360Day is supported")

    # Print the first date
    print(f"The first date is {dates[0]}")
    print(f"The last date is {dates[-1]}")

    # Add the dates to a dataframe
    dates_df = pd.DataFrame(dates, columns=["time"])

    # return the dates
    return dates_df


def get_sequences(mask, len_months):
    """
    Get the sequences of True values in the mask which have length equal to
    the number of months in the mask list.
    
    Parameters
    ----------
    
    mask: list[bool]
        The mask of True and False values.
        
    len_months: int
        The number of months in the mask list.

    Returns
    -------

    sequences: list[list[tuple]]
        A list of sequences of True values in the mask which have length equal to

    """
    sequences = []
    sequence = []
    for i, value in enumerate(mask):
        if value:
            sequence.append((value, i))
        else:
            if sequence and len(sequence) == len_months:
                sequences.append(sequence)
            sequence = []
    if sequence and len(sequence) == len_months:
        sequences.append(sequence)
    return sequences


# Translate the indepence testing function from R to Python
# Calculates the correlation between each unique pair of ensemble members
# for each lead time
# Based on Kelder UNSEEN functions:
# https://github.com/timokelder/UNSEEN/blob/master/R/independence_testing.R
def independence_test(
    ensemble: pd.DataFrame,
    members: List[str],
    n_leads: int,
    var_name: str,
    member_name: str = "member",
    lead_name: str = "lead",
    detrend: bool = False,
) -> np.ndarray:
    """
    Calculates the correlation between each unique pair of ensemble members
    using the Spearman correlation to quantify the dependence between the
    ensemble members.

    Parameters

    ensemble: pd.DataFrame
        The DataFrame containing the ensemble data. With columns for the
        ensemble member, lead time, and the variable of interest.

    members: list[str]
        The list of ensemble members.

    n_leads: int
        The number of lead times.

    var_name: str
        The name of the variable of interest.

    member_name: str
        The name of the ensemble member column. Default is "member".

    lead_name: str
        The name of the lead time column. Default is "lead".

    detrend: bool
        Whether to detrend the data before calculating the correlation.
        Default is False.
        If True, data are detrended by differencing.

    Returns

    corr_matrix: np.ndarray
        The correlation matrix between each unique pair of ensemble members.
    """

    # Set up the correlation matrix
    corr_matrix = np.zeros((n_leads, len(members), len(members)))

    # Loop over the lead times
    for lead in tqdm(range(1, n_leads + 1), desc="Calculating correlations"):

        # print the lead index
        print(f"Calculating correlations for lead {lead}")

        # Loop over the ensemble members
        for i, m1 in enumerate(members):
            for j, m2 in enumerate(members):
                # Only caluclate for the top half of the correlation matrix
                if i > j:
                    # Extract the data for the two ensemble members
                    m1_data = ensemble[
                        (ensemble[member_name] == m1) & (ensemble[lead_name] == lead)
                    ][var_name].values
                    m2_data = ensemble[
                        (ensemble[member_name] == m2) & (ensemble[lead_name] == lead)
                    ][var_name].values

                    # Detrend the data if specified
                    if detrend:
                        # Calculate the difference between the data
                        m1_data = np.diff(m1_data)
                        m2_data = np.diff(m2_data)

                    # Calculate the Spearman correlation
                    corr = stats.spearmanr(m1_data, m2_data)[0]

                    # Store the correlation in the matrix
                    corr_matrix[lead - 1, i, j] = corr

    return corr_matrix


# Write a function for plotting independence
def plot_independence(
    corr_matrix: np.ndarray,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the correlation matrix between each unique pair of ensemble members.

    Parameters
    ----------

    corr_matrix: np.ndarray
        The correlation matrix between each unique pair of ensemble members with shape (n_leads, len(members), len(members)).

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Set up the figure as a single plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

    # Dashed line at y=0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.8)

    # Loop over the lead times
    for lead in tqdm(range(corr_matrix.shape[0]), desc="Plotting correlations"):
        # Flatten the correlation matrix
        corr_flat = corr_matrix[lead, :, :].flatten()

        # Plot the correlation matrix as a boxplot
        # using matplotlib
        ax.boxplot(corr_flat, positions=[lead + 1], widths=0.8, whis=[5, 95])

    # Set the x-axis label
    ax.set_xlabel("Lead time")

    # Set the y-axis label
    ax.set_ylabel("Spearman correlation")

    # Set the x-ticks
    ax.set_xticks(range(1, corr_matrix.shape[0] + 1))
    # Set the x-tick labels
    ax.set_xticklabels(range(1, corr_matrix.shape[0] + 1))

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"corr_ensemble_members_{date}_{time}.pdf"))

    return


def plot_independence_sb(
    corr_matrix: np.ndarray,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the correlation matrix between each unique pair of ensemble members.

    Parameters
    ----------

    corr_matrix: np.ndarray
        The correlation matrix between each unique pair of ensemble members with shape (n_leads, len(members), len(members)).

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Set up the figure as a single plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

    # Create a DataFrame to hold the lead times and correlation values
    data = []

    # Loop over the lead times
    for lead in range(corr_matrix.shape[0]):
        # Flatten the correlation matrix
        corr_flat = corr_matrix[lead, :, :].flatten()

        # Add the lead times and correlation values to the DataFrame
        for value in corr_flat:
            data.append({"lead": lead, "correlation": value})

    df = pd.DataFrame(data)

    # clean the data by removing the NaN values
    df = df.dropna()

    # Plot the correlation matrix as a boxplot using seaborn
    sns.boxplot(x="lead", y="correlation", data=df, ax=ax)

    # Set the x-axis label
    ax.set_xlabel("Lead time")

    # Set the y-axis label
    ax.set_ylabel("Spearman correlation")

    # Set the x-ticks
    ax.set_xticks(range(corr_matrix.shape[0]))

    # Set the x-tick labels
    ax.set_xticklabels(range(1, corr_matrix.shape[0] + 1))

    # Constrain the y-axis to between -1 and 1
    ax.set_ylim(-0.3, 0.3)

    # # Set the current time
    # now = datetime.now()

    # # Set the current date
    # date = now.strftime("%Y-%m-%d")

    # # Set the current time
    # time = now.strftime("%H:%M:%S")

    # # Save the plot
    # plt.savefig(os.path.join(save_dir, f"corr_ensemble_members_{date}_{time}.pdf"))

    return df


def plot_independence_violin(
    corr_matrix: np.ndarray,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the correlation matrix between each unique pair of ensemble members.

    Parameters
    ----------

    corr_matrix: np.ndarray
        The correlation matrix between each unique pair of ensemble members with shape (n_leads, len(members), len(members)).

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Set up the figure as a single plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

    # Create a DataFrame to hold the lead times and correlation values
    data = []

    # Loop over the lead times
    for lead in range(corr_matrix.shape[0]):
        # Flatten the correlation matrix
        corr_flat = corr_matrix[lead, :, :].flatten()

        # Add the lead times and correlation values to the DataFrame
        for value in corr_flat:
            data.append({"lead": lead, "correlation": value})

    df = pd.DataFrame(data)

    # clean the data by removing the NaN values
    df = df.dropna()

    # Plot the correlation matrix as a violin plot using seaborn
    sns.violinplot(x="lead", y="correlation", data=df, ax=ax)

    # Set the x-axis label
    ax.set_xlabel("Lead time")

    # Set the y-axis label
    ax.set_ylabel("Spearman correlation")

    # Set the x-ticks
    ax.set_xticks(range(corr_matrix.shape[0]))

    # Set the x-tick labels
    ax.set_xticklabels(range(1, corr_matrix.shape[0] + 1))

    # Constrain the y-axis to between -1 and 1
    ax.set_ylim(-0.3, 0.3)

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"corr_ensemble_members_{date}_{time}.pdf"))

    return df


def plot_independence_pd(
    corr_matrix: np.ndarray,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the correlation matrix between each unique pair of ensemble members.

    Parameters
    ----------

    corr_matrix: np.ndarray
        The correlation matrix between each unique pair of ensemble members with shape (n_leads, len(members), len(members)).

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Set up the figure as a single plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

    # Create a DataFrame to hold the lead times and correlation values
    data = []

    # Loop over the lead times
    for lead in range(corr_matrix.shape[0]):
        # Flatten the correlation matrix
        corr_flat = corr_matrix[lead, :, :].flatten()

        # Add the lead times and correlation values to the DataFrame
        for value in corr_flat:
            data.append({"lead": lead, "correlation": value})

    df = pd.DataFrame(data)

    # Plot the correlation matrix as a boxplot using pandas
    df.boxplot(column="correlation", by="lead", ax=ax)

    # Set the x-axis label
    ax.set_xlabel("Lead time")

    # Set the y-axis label
    ax.set_ylabel("Spearman correlation")

    # Constrain the y-axis to between -1 and 1
    ax.set_ylim(-0.3, 0.3)

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"corr_ensemble_members_{date}_{time}.pdf"))

    return df


# Define a function to plot the model stability in terms of density
# After Timo Kelders functions
# https://github.com/timokelder/UNSEEN/blob/master/R/stability_test.R
def stability_density(
    ensemble: pd.DataFrame,
    var_name: str,
    label: str,
    cmap: str = "Blues",
    lead_name: str = "lead",
    fontsize: int = 12,
    fig_size: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
):
    """
    Function which plots the density distribution of different lead times.

    Parameters

    ensemble: pd.DataFrame
        The DataFrame containing the ensemble data. With columns for the
        ensemble member, lead time, and the variable of interest.

    var_name: str
        The name of the variable of interest.

    label: str
        The label for the variable of interest.

    cmap: str
        The colormap to use. Default is "Blues".

    lead_name: str
        The name of the lead time column. Default is "lead".

    fontsize: int
        The fontsize for the labels. Default is 12.

    fig_size: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns

    None

    """

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Get the unique lead times
    leads = sorted(ensemble[lead_name].unique())

    # Create a colormap
    colormap = cm.get_cmap(cmap, len(leads))

    # Loop over the lead times
    for i, lead in tqdm(enumerate(leads)):
        # Extract the data for the lead time
        data = ensemble[ensemble[lead_name] == lead][var_name]

        # Plot the density distribution with the color from the colormap
        sns.kdeplot(data, label=f"{lead}", color=colormap(i))

    # Set the x-axis label
    ax.set_xlabel(label, fontsize=fontsize)

    # Set the y-axis label
    ax.set_ylabel("Density", fontsize=fontsize)

    # Add a legend
    ax.legend(title="Lead time")

    # # Set the current time
    # now = datetime.now()

    # # Set the current date
    # date = now.strftime("%Y-%m-%d")

    # # Set the current time
    # time = now.strftime("%H:%M:%S")

    # # Save the plot
    # plt.savefig(os.path.join(save_dir, f"density_{date}_{time}.pdf"))

    return


# Define a function to plot the model stability in terms of density
# After Timo Kelders functions
# Showing the confidence interval of the distribution of all lead times pooled
# together
# Test whether the individual lead time falls within these confidence intervals
# In this case we bootstrap the pooled lead times into series with an equal
# length to the individual lead times (54 init years * 10 members = 540)
# with nboot = 10,000
def model_stability_boot(
    ensemble: pd.DataFrame,
    var_name: str,
    label: str,
    nboot: int = 10000,
    cmap: str = "Blues",
    lead_name: str = "lead",
    fontsize: int = 12,
    fig_size: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
):
    """
    Function which plots the density distribution of different lead times.

    Parameters

    ensemble: pd.DataFrame
        The DataFrame containing the ensemble data. With columns for the
        ensemble member, lead time, and the variable of interest.

    var_name: str
        The name of the variable of interest.

    label: str
        The label for the variable of interest.

    nboot: int
        The number of bootstrap samples to take. Default is 10000.

    cmap: str
        The colormap to use. Default is "Blues".

    lead_name: str
        The name of the lead time column. Default is "lead".

    fontsize: int
        The fontsize for the labels. Default is 12.

    fig_size: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns

    None

    """

    # Set up the length of the pooled ensemble
    pooled_length = ensemble.shape[0]  # i.e. all lead times pooled together

    # Print the pooled length
    print(f"The pooled length is {pooled_length}")

    # Set up the length of the individual lead time
    lead_length = sum(ensemble[lead_name] == 2)

    # Print the lead length
    print(f"The lead length is {lead_length}")

    # Initialise the bootstrap array
    # e.g. with shape (10000, 540) for 10,000 bootstraps
    # and 540 pooled ensemble members
    boot = np.zeros([nboot, lead_length])

    # Loop over the number of bootstraps
    for i in tqdm(range(nboot), desc="Performing bootstrapping"):
        # Sample the pooled ensemble
        boot[i, :] = np.random.choice(ensemble[var_name], lead_length)

    # Calculate the return periods
    pooled_rps = pooled_length / np.arange(1, pooled_length + 1)
    ld_rps = lead_length / np.arange(1, lead_length + 1)

    # Calculate the quantiles for each bootstrap sample
    return_vs = np.quantile(boot, q=1 - 1 / pooled_rps, axis=0)

    # Calculate the 2.5% and 97.5% quantiles for each return period
    ci_return_vs = np.quantile(return_vs, q=[0.025, 0.975], axis=1)

    # Create a DataFrame including the return periods, empirical values and confidence intervals
    df_quantiles = ensemble.copy()
    df_quantiles["rps_all"] = pooled_rps

    # Calculate the quantiles for the full ensemble
    df_quantiles["quantiles_all"] = df_quantiles.apply(
        lambda row: df_quantiles[var_name].quantile(1 - 1 / row["rps_all"]), axis=1
    )

    # Identify the unique lead times
    leads = ensemble[lead_name].unique()

    # Loop over the lead times
    for lead in leads:
        # Extract the data for the lead time
        data = ensemble[ensemble[lead_name] == lead]

        # Group the DataFrame by ld_name
        grouped = data.groupby(lead_name)

        # Where df_quantiles[df_quantiles[lead_name] == lead], we want to add the return periods
        # and the quantiles to the DataFrame
        # add the return periods to these specific rows of the dataframe
        ensemble.loc[ensemble[lead_name] == lead, "rps_ld"] = ld_rps

        # Calculate the quantiles for the lead time
        ensemble.loc[ensemble[lead_name] == lead, "quantiles_ld"] = ensemble[
            ensemble[lead_name] == lead
        ].apply(
            lambda row: ensemble[ensemble[lead_name] == lead][var_name].quantile(
                1 - 1 / row["rps_ld"]
            ),
            axis=1,
        )

    # add the new columns in ensemble to the DataFrame
    df_quantiles["rps_ld"] = ensemble["rps_ld"]

    # add the new columns in ensemble to the DataFrame
    df_quantiles["quantiles_ld"] = ensemble["quantiles_ld"]

    # Add the confidence intervals to the DataFrame
    df_quantiles["ci_2.5"] = ci_return_vs[0, :]
    df_quantiles["ci_97.5"] = ci_return_vs[1, :]

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Set ensemble to the DataFrame
    ensemble = df_quantiles

    # Create a colour map
    colormap = cm.get_cmap(cmap, len(ensemble[lead_name].unique()))

    # Add the lines to the plot
    for i, lead in tqdm(enumerate(ensemble[lead_name].unique())):
        # Extract a subset of the dataframe for the specific lead time
        data = ensemble[ensemble[lead_name] == lead]

        # Plot the line
        sns.lineplot(
            x="rps_ld",
            y="quantiles_ld",
            data=data,
            color=colormap(i),
            ax=ax,
            label=f"{lead}",
        )

    # plot the quantiles for the full ensemble
    sns.lineplot(
        x="rps_all",
        y="quantiles_all",
        data=df_quantiles,
        color="black",
        ax=ax,
        label="Full ensemble",
        linestyle="--",
    )

    # Add the shaded area to the plot
    plt.fill_between(
        df_quantiles["rps_all"],
        df_quantiles["ci_2.5"],
        df_quantiles["ci_97.5"],
        color="black",
        alpha=0.1,
    )

    # Set the x-axis to a logarithmic scale
    plt.xscale("log")

    # Set the xlabels as 10, 100, 1000, 10000
    plt.xticks([1, 10, 100, 1000])

    # Set the xlim
    plt.xlim([1, 1000])

    # Set the labels of the x-axis and y-axis
    plt.xlabel("Return period (years)")
    plt.ylabel(label)

    # Add a legend
    plt.legend()

    # Set up the time now
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"stability_EV_{today}_{time}.pdf"))

    # Show the plot
    plt.show()

    return


# Define a function to plot the model fidelity
# After my own functions
def plot_fidelity(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str = "year",
    model_time_name: str = "init",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
    nboot: int = 10000,
    figsize: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Calculates the bootstrap statistics for the model fidelity for the hindcast
    and the statistics for the obs datasets before plotting these.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the
        observation value and the observation time.

    model_df: pd.DataFrame
        The DataFrame containing the model data with columns for the
        model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    obs_time_name: str
        The name of the observation time column. Default is "year".

    model_time_name: str
        The name of the model time column. Default is "init".

    model_member_name: str
        The name of the model member column. Default is "member".

    model_lead_name: str
        The name of the model lead time column. Default is "lead".

    nboot: int
        The number of bootstrap samples to take. Default is 10000.

    figsize: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns

    None

    """

    # Set up the model stats dict
    model_stats = {
        "mean": [],
        "sigma": [],
        "skew": [],
        "kurt": [],
    }

    # Assert that the len of unique init in model_df
    # is equal to the len of unique year in obs_df
    assert len(model_df[model_time_name].unique()) == len(
        obs_df[obs_time_name].unique()
    ), "The number of unique initialisation dates in the model data must be equal to the number of unique years in the observations."

    # Set up the number of unique initialisation dates
    n_years = len(model_df[model_time_name].unique())

    # Set up the number of unique ensemble members
    n_members = len(model_df[model_member_name].unique())

    n_leads = len(model_df[model_lead_name].unique())

    # Set up zeros for the bootstrapped values
    boot_mean = np.zeros(nboot)
    boot_sigma = np.zeros(nboot)
    boot_skew = np.zeros(nboot)
    boot_kurt = np.zeros(nboot)

    # Extract the unique model times
    model_times = model_df[model_time_name].unique()

    # Extract the unique model members
    model_members = model_df[model_member_name].unique()

    # Extract the unique model leads
    model_leads = model_df[model_lead_name].unique()

    # Create the indexes for the ensemble members
    member_idx = np.arange(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(range(nboot), desc="Calculating bootstrap statistics"):
        # Create the time index
        idx_time_this = range(0, n_years)

        # Create an empty array to store the bootstrapped values
        model_boot = np.zeros([n_years])

        # Set the year index to 0
        idx_year = 0

        # Loop over the number of years
        for itime in idx_time_this:
            # Set up random indices for the ensemble members
            idx_ens_this = random.choices(member_idx)

            # Set up a random choice for the lead time
            idx_lead_this = random.choices(range(n_leads))

            # Find the time at the itime index
            model_time_this = model_times[itime]

            # Find the name for the member at this index
            model_member_this = model_members[idx_ens_this]

            # Find the name for the lead at this index
            model_lead_this = model_leads[idx_lead_this]

            # Extract the model data for the year and ensemble members
            model_data = model_df[
                (model_df[model_time_name] == model_time_this)
                & (model_df[model_member_name] == model_member_this[0])
                & (model_df[model_lead_name] == model_lead_this[0])
            ][model_val_name].values

            # Append the model data to the bootstrapped array
            model_boot[idx_year] = model_data

            # Increment the year index
            idx_year += 1

        # Calculate the statistics for the bootstrapped array
        boot_mean[iboot] = np.mean(model_boot)

        boot_sigma[iboot] = np.std(model_boot)

        boot_skew[iboot] = stats.skew(model_boot)

        boot_kurt[iboot] = stats.kurtosis(model_boot)

    # Append the bootstrapped statistics to the model_stats dict
    model_stats["mean"] = boot_mean
    model_stats["sigma"] = boot_sigma
    model_stats["skew"] = boot_skew
    model_stats["kurt"] = boot_kurt

    # Calculate the obs stats
    # Define the mdi
    # mdi = -9999.0

    # Set up the dictionary to store the obs stats
    obs_stats = {
        "mean": obs_df[obs_val_name].mean(),
        "sigma": obs_df[obs_val_name].std(),
        "skew": stats.skew(obs_df[obs_val_name]),
        "kurt": stats.kurtosis(obs_df[obs_val_name]),
    }

    # Set up the figure as 2x2
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Form a list of the axes
    axes = axes.flatten()

    # Form a list of the model stats
    model_stats_list = [
        model_stats["mean"],
        model_stats["sigma"],
        model_stats["skew"],
        model_stats["kurt"],
    ]

    # For the same list of the obs stats
    obs_stats_list = [
        obs_stats["mean"],
        obs_stats["sigma"],
        obs_stats["skew"],
        obs_stats["kurt"],
    ]

    # For the same list of the stat names
    stat_names = ["mean", "sigma", "skew", "kurt"]

    # Form the list of the axes labels
    axes_labels = ["a", "b", "c", "d"]

    # Loop over the axes
    for i, ax in enumerate(axes):
        # Plot the histogram of the model stats
        ax.hist(model_stats_list[i], bins=100, density=True, color="red", label="model")

        # Plot the obs stats
        ax.axvline(obs_stats_list[i], color="black", linestyle="-", label="ERA5")

        # Calculate the position of the obs stat in the distribution
        obs_pos = stats.percentileofscore(model_stats_list[i], obs_stats_list[i])

        # Plot vertical black dashed lines for the 2.5% and 97.5% quantiles of the model stats
        ax.axvline(
            np.quantile(model_stats_list[i], 0.025), color="black", linestyle="--"
        )

        ax.axvline(
            np.quantile(model_stats_list[i], 0.975), color="black", linestyle="--"
        )

        # Add a title in bold
        ax.set_title(f"{stat_names[i]}, {obs_pos:.2f}%", fontweight="bold")

        # Add the axes label
        # in the top left
        ax.text(
            0.05,
            0.95,
            axes_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5),
            zorder=100,
        )

    # # Set up teh current time in d m y h m s
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # # Save the plot
    plt.savefig(os.path.join(save_dir, f"fidelity_{date}_{time}.pdf"))

    # show the plot
    plt.show()

    return


# Define a function for plotting the events
# using the dataframes
def plot_events_ts(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    ylabel: str,
    model_name: str = "HadGEM3-GC31-MM",
    obs_time_name: str = "year",
    model_time_name: str = "init",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
    delta_shift_bias: bool = False,
    do_detrend: bool = False,
    figsize: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plots the hindcast events on the same axis as the observed events.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the
        observation value and the observation time.

    model_df: pd.DataFrame
        The DataFrame containing the model data with columns for the
        model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    ylabel: str
        The y-axis label.

    model_name: str
        The name of the model. Default is "HadGEM3-GC31-MM".

    obs_time_name: str
        The name of the observation time column. Default is "year".

    model_time_name: str
        The name of the model time column. Default is "init".

    model_member_name: str
        The name of the model member column. Default is "member".

    model_lead_name: str
        The name of the model lead time column. Default is "lead".

    delta_shift_bias: bool
        Whether to shift the model data by the bias. Default is False.

    do_detrend: bool
        Whether to detrend the data. Default is False.

    figsize: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns
    -------

    None

    """

    # Set up the years
    years = obs_df[obs_time_name].unique()

    # If bias shift is True
    if delta_shift_bias:
        print("Shifting the model data by the bias")
        # Calculate the bias
        bias = model_df[model_val_name].mean() - obs_df[obs_val_name].mean()

        # Shift the model data by the bias
        model_df[model_val_name] = model_df[model_val_name] - bias

    # if do detrend is true
    if do_detrend:
        print("Detrending the data")
        # Detrend the data
        obs_df[obs_val_name] = signal.detrend(obs_df[obs_val_name])
        model_df[model_val_name] = signal.detrend(model_df[model_val_name])

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Loop over the ensemble members
    for i, member in enumerate(model_df[model_member_name].unique()):
        # Seperate the data based on the condition
        model_data = model_df[model_df[model_member_name] == member]

        # Seperate data into below and above the threshold
        # model_data_below20 = model_data[model_data[model_val_name] < np.quantile(obs_df[obs_val_name], 0.2)]

        model_data_below20 = (
            obs_df[obs_val_name].min() < model_data[model_val_name]
        ) & (model_data[model_val_name] < np.quantile(obs_df[obs_val_name], 0.2))

        # Above the threshold
        model_data_above20 = (
            model_data[model_val_name] >= obs_df[obs_val_name].min()
        ) & ~model_data_below20

        # below the minimum of the obs
        model_data_below_obs_min_bool = (
            model_data[model_val_name] < obs_df[obs_val_name].min()
        )

        # Plot the points below the 20th percentile
        ax.scatter(
            model_data[model_data_below20][model_time_name],
            model_data[model_data_below20][model_val_name],
            color="blue",
            alpha=0.8,
            label="model wind drought" if i == 0 else None,
        )

        # Plot the points above the 20th percentile
        ax.scatter(
            model_data[model_data_above20][model_time_name],
            model_data[model_data_above20][model_val_name],
            color="grey",
            alpha=0.8,
            label=model_name if i == 0 else None,
        )

        # Plot the points below the minimum of the obs
        ax.scatter(
            model_data[model_data_below_obs_min_bool][model_time_name],
            model_data[model_data_below_obs_min_bool][model_val_name],
            color="red",
            alpha=0.8,
            marker="x",
            label="model wind drought" if i == 0 else None,
        )

    # Plot the observed data
    ax.scatter(
        obs_df[obs_time_name],
        obs_df[obs_val_name],
        color="black",
        label="ERA5",
    )

    # Plot the 20th percentile of the obs as a horizontal line
    ax.axhline(np.quantile(obs_df[obs_val_name], 0.2), color="blue", linestyle="--")

    # Plot the minimum of the obs as a horizontal line
    ax.axhline(obs_df[obs_val_name].min(), color="red", linestyle="--")

    # Legend in the upper left
    ax.legend(loc="upper left")

    # Set the x-axis label
    ax.set_xlabel("Year")

    # Set the y-axis label
    ax.set_ylabel(ylabel)

    # Show the plot
    plt.show()

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    # plt.savefig(os.path.join(save_dir, f"events_{date}_{time}.pdf"))

    return


# Define a function to plot the events time series using boxplots
# for the hindcast data
def plot_events_ts_bp(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    ylabel: str,
    model_name: str = "HadGEM3-GC31-MM",
    obs_time_name: str = "year",
    model_time_name: str = "init",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
    delta_shift_bias: bool = False,
    do_detrend: bool = False,
    figsize: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plots the hindcast events on the same axis as the observed events using boxplots.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the
        observation value and the observation time.

    model_df: pd.DataFrame
        The DataFrame containing the model data with columns for the
        model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    ylabel: str
        The y-axis label.

    model_name: str
        The name of the model. Default is "HadGEM3-GC31-MM".

    obs_time_name: str
        The name of the observation time column. Default is "year".

    model_time_name: str
        The name of the model time column. Default is "init".

    model_member_name: str
        The name of the model member column. Default is "member".

    model_lead_name: str
        The name of the model lead time column. Default is "lead".

    delta_shift_bias: bool
        Whether to shift the model data by the bias. Default is False.

    do_detrend: bool
        Whether to detrend the data. Default is False.

    figsize: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns
    -------

    None

    """

    # Set up the years
    years = obs_df[obs_time_name].unique()

    # If bias shift is True
    if delta_shift_bias:
        print("Shifting the model data by the bias")
        # Calculate the bias
        bias = model_df[model_val_name].mean() - obs_df[obs_val_name].mean()

        # Shift the model data by the bias
        model_df[model_val_name] = model_df[model_val_name] - bias

    # if do detrend is true
    if do_detrend:
        print("Detrending the data")
        # Detrend the data
        obs_df[obs_val_name] = signal.detrend(obs_df[obs_val_name])
        model_df[model_val_name] = signal.detrend(model_df[model_val_name])

    # Set up the figure with two subplots
    # Set up the figure with two subplots with different widths
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [8, 1]},
    )

    # plot a horizontal line for the 20th percentil of the obs
    axs[0].axhline(np.quantile(obs_df[obs_val_name], 0.2), color="blue", linestyle="--")

    # plot a horizontal line for the minimum of the obs
    axs[0].axhline(obs_df[obs_val_name].min(), color="blue", linestyle="-.")

    # Loop over the years
    for i, year in enumerate(years):
        # Extract the model data for the year
        model_data = model_df[model_df[model_time_name] == year]

        # Plot the boxplot for the model data on the first subplot
        axs[0].boxplot(
            model_data[model_val_name],
            positions=[year],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="red", color="red"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            zorder=1,
            flierprops=dict(
                marker="o",
                markerfacecolor="k",
                markersize=5,
                linestyle="none",
                markeredgecolor="k",
                alpha=0.5,
            ),  # Set flier properties
        )

    # Plot the observed data as blue crosses on the first subplot
    axs[0].scatter(
        years,
        obs_df[obs_val_name],
        color="blue",
        marker="x",
        label="ERA5",
        zorder=2,
    )

    # Plot the boxplot for the observed data on the second subplot
    axs[1].boxplot(
        obs_df[obs_val_name],
        positions=[1],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="blue", color="blue"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        zorder=1,
        flierprops=dict(
            marker="o",
            markerfacecolor="k",
            markersize=5,
            linestyle="none",
            markeredgecolor="k",
        ),  # Set flier properties
    )

    # also include a red boxplot for the model data
    axs[1].boxplot(
        model_df[model_val_name],
        positions=[2],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="red", color="red"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        zorder=1,
        flierprops=dict(
            marker="x",
            markerfacecolor="k",
            markersize=5,
            linestyle="none",
            markeredgecolor="k",
            alpha=0.5,
        ),  # Set flier properties
    )

    # Set the x-axis label
    axs[0].set_xlabel("Year")

    # Set the y-axis label
    axs[0].set_ylabel(ylabel)

    # print years min and max
    print(f"The years min is {years.min()} and the years max is {years.max()}")

    # # Format the x-ticks for ticks every 10 years
    # ax.set_xticks(np.arange(years.min(), years.max() + 1, 10))

    # shift years back by 1
    years = years - 1

    axs[0].set_xticks(range(years[0], years[-1] + 1, 10))
    axs[0].set_xticklabels(range(years[0], years[-1] + 1, 10))
    # Set the legend
    axs[0].legend()

    # specify a tight layout
    plt.tight_layout()

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"events_{date}_{time}.pdf"))

    # Show the plot
    plt.show()

    return


# Write a function to create a GEV distribution which
# is linearly related to the time period of the hindcast data
# Then plot the return value plots for two periods specified
# e.g. 1961-1981 and 1994-2014
def plot_gev_return_values(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    ylabel: str,
    model_name: str = "HadGEM3-GC31-MM",
    obs_time_name: str = "year",
    model_time_name: str = "init",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
    delta_shift_bias: bool = False,
    do_detrend: bool = False,
    figsize: tuple = (10, 10),
    rperiods: list = [
        2,
        5,
        10,
        20,
        50,
        80,
        100,
        120,
        200,
        250,
        300,
        500,
        800,
        2000,
        5000,
    ],
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plots the return values for the GEV distribution for two time periods.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the
        observation value and the observation time.

    model_df: pd.DataFrame
        The DataFrame containing the model data with columns for the
        model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    ylabel: str
        The y-axis label.

    model_name: str
        The name of the model. Default is "HadGEM3-GC31-MM".

    obs_time_name: str
        The name of the observation time column. Default is "year".

    model_time_name: str
        The name of the model time column. Default is "init".

    model_member_name: str
        The name of the model member column. Default is "member".

    model_lead_name: str
        The name of the model lead time column. Default is "lead".

    delta_shift_bias: bool
        Whether to shift the model data by the bias. Default is False.

    do_detrend: bool
        Whether to detrend the data. Default is False.

    figsize: tuple
        The figure size. Default is (10, 10).

    rperiods: list
        The return periods to plot. Default is [2, 5, 10, 20, 50, 80, 100, 120, 200, 250, 300, 500, 800, 2000, 5000].

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns
    -------

    None

    """

    # Set up the years
    years = obs_df[obs_time_name].unique()

    # Print the min and max years
    print(f"The minimum year is {years.min()} and the maximum year is {years.max()}")

    # Call the RV_ci function for the first time period
    rvs1 = RV_ci(
        extremes=model_df[model_val_name],
        covariate=model_df[model_time_name],
        return_period=rperiods,
        covariate_values=years,
    )

    return rvs1


# Define a function to fit a GEV distribution to provided data
# and then calculate the return values and confidence intervals for
# a specified year/time period
def RV_ci(
    extremes: np.ndarray,
    covariate: np.ndarray,
    return_period: int,
    covariate_values: np.ndarray,
) -> np.ndarray:
    """
    Fits a GEV distribution to the provided data and calculates the
    return values and confidence intervals for a specified year/time period.

    Parameters
    ----------

    extremes: np.ndarray
        The array of extreme values.
        E.g. ONDJFM average 10m wind speeds

    covariate: np.ndarray
        The array of covariate values.

    return_period: int
        The return period.

    covariate_values: np.ndarray
        The array of covariate values.

    Returns
    -------

    rvs: np.ndarray
        The array of return values.

    """

    # Fit a GEV distribution to the extremes data
    c, loc, scale = genextreme.fit(extremes)

    # print these
    print(f"The shape parameter is {c}")
    print(f"The location parameter is {loc}")
    print(f"The scale parameter is {scale}")

    # print the covariate values
    print(f"The covariate values are {covariate_values}")

    # print the covariate shape
    print(f"The covariate shape is {covariate.shape}")

    # Fit a linear regression model to the location and scale parameters
    loc_slope, loc_intercept, _, _, _ = linregress(covariate, loc)
    scale_slope, scale_intercept, _, _, _ = linregress(covariate, scale)

    # Plot a scatter between the covariate and the location parameter
    plt.scatter(covariate, loc)

    # Plot the linear regression line
    plt.plot(
        covariate_values,
        loc_slope * covariate_values + loc_intercept,
        color="red",
        label="Location",
    )

    # Predict the location and scale parameters for the covariate values
    loc_values = loc_slope * covariate_values + loc_intercept
    scale_values = scale_slope * covariate_values + scale_intercept

    # Calculate the return values for the predicted location and scale parameters
    rvs = genextreme.ppf(
        (1.0 - 1.0 / return_period), c, loc=loc_values, scale=scale_values
    )

    return rvs
