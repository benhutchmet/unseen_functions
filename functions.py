# Functions for UNSEEN work

# Local imports
import os
import sys
import glob
import random
import re
import calendar
import time

# Third party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy import stats, signal
from scipy.stats import genextreme as gev
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.gridspec as gridspec

# from matplotlib.gridspec import GridSpecFromSubplotSpec
import iris
import shapely.geometry
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from datetime import datetime, timedelta

# import xesmf as xe
import matplotlib.cm as cm
import cftime

# Import types
from typing import Any, Callable, Union, List
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, norm, ks_2samp
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from typing import Any, List, Tuple
from datetime import datetime
from iris.util import equalise_attributes

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
    model_path_root_psl = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root_psl == "gws":
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
    elif model_path_root_psl == "badc":
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


# # Define a function for preprocessing the model data
# def preprocess(
#     ds: xr.Dataset,
# ):
#     """
#     Preprocess the model data using xarray

#     Parameters

#     ds: xr.Dataset
#         The dataset to preprocess
#     """

#     # Return the dataset
#     return ds


# Write a new function for loading the model data using xarray
def load_model_data_xarray(
    model_variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    first_fcst_year: int,
    last_fcst_year: int,
    months: list,
    member: str = "None",
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

    first_fcst_year: int
        The first forecast year for taking the time average
        E.g. 1960

    last_fcst_year: int
        The last forecast year for taking the time average
        E.g. 1962

    months: list
        The months to take the time average over
        E.g. [10, 11, 12, 1, 2, 3] for October to March

    member: str
        The ensemble member to load the data for. Default is 'None'

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

    # print the model path
    print("Model path:", model_path)

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Extract the first part of the model_path
    model_path_root_psl = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root_psl == "gws":
        print("The model path root is gws")

        # List the files in the model path
        model_files = [file for file in os.listdir(model_path) if file.endswith(".nc")]

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Find all of the files for the given year that are .nc files
            year_files = [
                file
                for file in model_files
                if f"s{year}" in file and file.endswith(".nc")
            ]

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

    elif model_path_root_psl == "badc":
        print("The model path root is badc")

        # Set up the list
        model_files = []

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/{frequency}/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # append the year files to the model files
            model_files.extend(year_files)

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

    # assert that model files has one dimension
    assert len(np.shape(model_files)) == 1, "model_files is not one dimensional"

    # Print the number of unique variant labels
    print("Number of unique variant labels:", len(unique_variant_labels))
    print("For model:", model)

    # print the first 5 unique variant labels
    print("First 10 unique variant labels:", unique_variant_labels[:10])

    # if the model is CanESM5
    # limit the variant labels to r1 - r20
    if model == "CanESM5":
        # Extract the variant labels
        unique_variant_labels = [
            label
            for label in unique_variant_labels
            if "r" in label and int(label[1:-6]) <= 20
        ]

    # Print the number of unique variant labels
    print("Number of unique variant labels:", len(unique_variant_labels))

    # print the unique variant lbles
    print("Unique variant labels:", unique_variant_labels)

    # Create an empty list for forming the list of files for each ensemble member
    member_files = []

    # print the first 10 model files
    print("First 10 model files:", model_files[:10])

    # If the model path root is gws
    if model_path_root_psl == "gws":
        print("Forming the list of files for each ensemble member for gws")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # # print the year and variant label
                # print(year, variant_label)

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
    elif model_path_root_psl == "badc":
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

                # Debug print to check the number of files found
                print(
                    f"Year: {year}, Variant: {variant_label}, Files found: {len(year_files)}"
                )

                # Handle cases where no files are found
                if not year_files:
                    print(
                        f"No files found for year {year} and variant label {variant_label}"
                    )
                    # Optionally, you can add a placeholder or skip this year
                    # year_files = ["placeholder.nc"]  # Example placeholder
                    # sys.exit()  # Uncomment to exit if no files are found

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

    # Print the member files list
    print("Member files list:", member_files)

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

    # # print the shape of member files
    # print("Shape of member files:", np.shape(member_files))

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

    # print the unqiue variant labels
    print("Unique variant labels:", unique_variant_labels)

    # if member is not in "none", "None"
    if member not in ["none", "None"]:
        print(f"Extracting specific member: {member}")

        member_files_subset = []
        unique_variant_labels_subset = [member]

        # loop over the member files
        for member_file in member_files:
            if member in member_file:
                member_files_subset.append(member_file)

        # print the len of member files subset
        print("Length of member files subset:", len(member_files_subset))
        print(
            "Length of unique variant labels subset:", len(unique_variant_labels_subset)
        )

        # set member files as the subset
        member_files = member_files_subset
        unique_variant_labels = unique_variant_labels_subset

    init_year_list = []
    # Loop over init_years
    for init_year in tqdm(
        range(start_year, end_year + 1), desc="Processing init years"
    ):
        # print(f"processing init year {init_year}")
        # Set up the member list
        member_list = []
        # Loop over the unique variant labels
        for variant_label in tqdm(unique_variant_labels):
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
                preprocess=lambda ds: preprocess_boilerplate(ds),
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

    # print the lat and lon res
    print("Lat res:", lat_res)
    print("Lon res:", lon_res)

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
    calc_mean: bool = True,
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

    calc_mean: bool, optional
        Whether to calculate the mean over the selected gridbox. Default is True.

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

    if not calc_mean:
        # Return the masked dataset
        return ds_masked

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
    parallel: bool = False,
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

    parallel: bool, optional
        Whether to use parallel processing. Default is False.

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
        parallel=parallel,
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
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    xlabel: str,
    figsize=(6, 6),
    nbins: int = 100,
    title: str = "Distribution of 10m wind speed",
    obs_val_name: str = "obs",
    model_val_name: str = "data",
    fname_prefix: str = "distribution",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the distribution of the model and obs data

    Parameters
    ----------

    obs_df: pd.DataFrame
        The observations dataframe

    model_df: pd.DataFrame
        The model dataframe

    nbins: int
        The number of bins for the histogram

    obs_val_name: str
        The name of the observations value
        Default is "obs"

    model_val_name: str
        The name of the model value
        Default is "data"

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Plot the distributions of the data as histograms
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the model data on the first y-axis
    plt.hist(
        model_df[model_val_name], color="red", label="model", alpha=0.5, density=True
    )

    # Plot the obs data on the second y-axis
    plt.hist(obs_df[obs_val_name], color="black", label="obs", alpha=0.5, density=True)

    # # Include a textbox with the sample size
    # ax.text(
    #     0.95,
    #     0.90,
    #     f"model N = {len(model_df)}\nobs N = {len(obs_df)}",
    #     transform=ax.transAxes,
    #     bbox=dict(facecolor="white", alpha=0.5),
    #     horizontalalignment='right'
    # )

    # plt.show()

    # include a dashed vertical line for the mean
    # plt.axvline(
    #     model_df[model_val_name].mean(),
    #     color="red",
    #     linestyle="--",
    #     label="model mean",
    # )

    # # include a dashed vertical line for the mean
    # plt.axvline(
    #     obs_df[obs_val_name].mean(), color="black", linestyle="--", label="obs mean"
    # )

    # Include a textbox with the sample size
    plt.text(
        0.95,
        0.90,
        f"model N = {len(model_df)}\nobs N = {len(obs_df)}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
        horizontalalignment="right",
    )
    # # Add a legend
    # plt.legend()

    # remove the ticks for the y axis
    plt.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    # remove the numbers from the y axis
    plt.yticks([])

    # Add a title
    # TODO: hard coded title
    plt.title(title)

    # set the x-axis label
    plt.xlabel(xlabel)

    # Set the current time
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    # tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{fname_prefix}_{current_datetime}.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    # Show the plot
    plt.show()

    return


# Define a function to plot the distributions and also fidelity test
def plot_distributions_fidelity(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    model_time_name: str,
    model_member_name: str,
    model_lead_name: str,
    title: str,
    nboot: int = 1000,
    figsize=(10, 6),
    nbins: int = 100,
    fname_prefix: str = "distribution_fidelity",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plots the distribution and calculates the model fidelity
    for the hindcast using the bootstrap statistics.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The observations dataframe
    model_df: pd.DataFrame
        The model dataframe
    obs_val_name: str
        The name of the observations value
    model_val_name: str
        The name of the model value
    obs_time_name: str
        The name of the observations time
    model_time_name: str
        The name of the model time
    model_member_name: str
        The name of the model member
    model_lead_name: str
        The name of the model lead
    title: str
        The title of the plot
    nboot: int
        The number of bootstrap samples to take
        Default is 1000
    figsize: tuple
        The figure size
        Default is (10, 6)
    nbins: int
        The number of bins for the histogram
        Default is 100
    fname_prefix: str
        The prefix for the filename
        Default is "distribution_fidelity"
    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None

    """

    # Set up the figure
    # two columns, one row
    # fig, axs = plt.subplots(
    #     ncols=3, nrows=2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1, 1]}
    # )

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[2, 1, 1])

    ax_main = ax_main = fig.add_subplot(gs[:, 0])  # Span all rows in the first column

    # Plot the distributions on the first axis
    ax_main.hist(
        model_df[model_val_name], color="red", label="model", alpha=0.5, density=True
    )

    # Obs data
    ax_main.hist(
        obs_df[obs_val_name], color="black", label="obs", alpha=0.5, density=True
    )

    # Include a textbox with the sample size
    ax_main.text(
        0.95,
        0.90,
        f"model N = {len(model_df)}\nobs N = {len(obs_df)}",
        transform=ax_main.transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
        horizontalalignment="right",
    )

    # Add a legend
    ax_main.legend()

    # Remove the ticks for the y axis
    ax_main.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )

    # Remove the numbers from the y axis
    ax_main.set_yticks([])

    # Add a title
    ax_main.set_title(title)

    # Set up the model stats dict
    model_stats = {
        "mean": [],
        "sigma": [],
        "skew": [],
        "kurt": [],
    }

    # Set up the number of unique initialisation dates
    n_years = len(model_df[model_time_name].unique())

    # Set up the number of unique ensemble members
    n_members = len(model_df[model_member_name].unique())

    # if the model_lead_name is not None
    if model_lead_name is not None:
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

    if model_lead_name is not None:
        # Extract the unique model leads
        model_leads = model_df[model_lead_name].unique()

    # Create the indexes for the ensemble members
    member_idx = np.arange(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(range(nboot), desc="Calculating bootstrap statistics"):
        # Set up random indices for the ensemble members
        idx_time_this = range(0, n_years)

        # Create an empty array to store the bootstrapped values
        model_boot = np.zeros([n_years])

        # Set the year index to 0
        idx_year = 0

        # Loop over the number of years
        # Randomly select an ensemble member and lead time for
        # each year
        # But year range stays constant
        for itime in idx_time_this:
            # Set up random indices for the ensemble members
            idx_ens_this = random.choices(member_idx)

            # Find the time at the itime index
            model_time_this = model_times[itime]

            # Find the name for the member at this index
            model_member_this = model_members[idx_ens_this]

            # if model_lead_name is not None
            if model_lead_name is not None:
                # Set up a random choice for the lead time
                idx_lead_this = random.choices(range(n_leads))

                # Find the name for the lead at this index
                model_lead_this = model_leads[idx_lead_this]

                # Extract the model data for the year and ensemble members
                model_data = model_df[
                    (model_df[model_time_name] == model_time_this)
                    & (model_df[model_member_name] == model_member_this[0])
                    & (model_df[model_lead_name] == model_lead_this[0])
                ][model_val_name].values
            else:
                # Extract the model data for the year and ensemble members
                model_data = model_df[
                    (model_df[model_time_name] == model_time_this)
                    & (model_df[model_member_name] == model_member_this[0])
                ][model_val_name].values

            # Check if model_data is empty
            if model_data.size == 0:
                # print(f"No data available for time {model_time_this}, member {model_member_this}, lead {model_lead_this if model_lead_name else 'N/A'}")
                continue  # Skip this iteration if no data is available

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

    # Additional subplots for metrics
    ax_mean = fig.add_subplot(gs[0, 1])
    ax_skew = fig.add_subplot(gs[0, 2])
    ax_stddev = fig.add_subplot(gs[1, 1])
    ax_kurtosis = fig.add_subplot(gs[1, 2])

    axes = [ax_mean, ax_skew, ax_stddev, ax_kurtosis]

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

        # rmeove the yticks
        ax.set_yticks([])

        # Add a title in bold with obs_pos rounded to the closest integer
        ax.set_title(f"{stat_names[i]}, {round(obs_pos)}%", fontweight="bold")

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

    # specify a tight layout
    fig.tight_layout()

    # # Set up teh current time in d m y h m s
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")

    # # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{fname_prefix}_{date}_{time}.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    # show the plot
    plt.show()

    return


# write a function to plot the distributions for the indiviudal months
def plot_distribution_months(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    xlabel: str,
    months: list,
    figsize=(10, 8),
    n_fcst_years: int = 10,
    nbins: int = 100,
    title: str = "Distribution of 10m wind speed",
    obs_val_name: str = "obs",
    model_val_name: str = "data",
    fname_prefix: str = "distribution_months",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plots the direct comparison of the model and obs data for the individual months.

    E.g. O, N, D, J, F, M all on the same plot.

    Parameters
    ----------

    obs_df: pd.DataFrame
        The observations dataframe

    model_df: pd.DataFrame
        The model dataframe

    xlabel: str
        The x-axis label

    months: list
        The months to plot

    figsize: tuple
        The figure size
        Default is (6, 6)

    nbins: int
        The number of bins for the histogram
        Default is 100

    title: str
        The title of the plot
        Default is "Distribution of 10m wind speed"

    obs_val_name: str
        The name of the observations value
        Default is "obs"

    model_val_name: str
        The name of the model value
        Default is "data"

    fname_prefix: str
        The prefix for the filename
        Default is "distribution_months"

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Set up the figure
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=figsize)

    if months != [10, 11, 12, 1, 2, 3]:
        raise ValueError("The months must be [10, 11, 12, 1, 2, 3]")

    # Loop over the months
    for i, month in enumerate(months):
        # set up the axes
        ax = axes.ravel()[i]

        # if the month has two digits
        if month >= 10:
            # Set up the leads to sel
            leads = [12 * i + (month - 10) for i in range(1, n_fcst_years + 1)]
        elif month < 10:
            # Set up the leads to sel
            leads = [12 * i + (month - 10 + 12) for i in range(1, n_fcst_years + 1)]

        # plot all of the model data
        ax.hist(
            model_df[model_val_name],
            color="red",
            label="model",
            alpha=0.0,
            density=True,
        )

        # Plot the obs data on the second y-axis
        ax.hist(
            obs_df[obs_val_name], color="black", label="obs", alpha=0.0, density=True
        )

        # subset the model_df for the leads
        model_df_sub = model_df[model_df["lead"].isin(leads)]

        # subset the obs_df for the leads
        obs_df_sub = obs_df[obs_df["time"].dt.month.isin([month])]

        # # print obs df sub
        # print(obs_df_sub.head())

        # plot the model data
        ax.hist(
            model_df_sub[model_val_name],
            color="red",
            label="model",
            alpha=0.5,
            density=True,
        )

        # plot the obs data
        ax.hist(
            obs_df_sub[obs_val_name],
            color="black",
            label="obs",
            alpha=0.5,
            density=True,
        )

        # include a subplot title for the month
        ax.set_title(f"{calendar.month_abbr[month]}")

        # Include a vertical dahsed red line for the model mean
        ax.axvline(
            model_df_sub[model_val_name].mean(),
            color="red",
            linestyle="--",
            label="model mean",
        )

        # Include a vertical dashed black line for the obs mean
        ax.axvline(
            obs_df_sub[obs_val_name].mean(),
            color="black",
            linestyle="--",
            label="obs mean",
        )

        # # print the month
        # print(f"Month: {month}")

        # # print the leads
        # print(f"Leads: {leads}")

        # Set the x-axis label
        ax.set_xlabel(xlabel)

        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax.set_yticks([])

    # Remove the ticks for the y axis
    plt.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    # remove the numbers from the y axis
    plt.yticks([])

    # Set the current time
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    # tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{fname_prefix}_{current_datetime}.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    # Show the plot
    plt.show()

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
    fname_root: str = "density",
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

    fname_root: str
        The root name for the file. Default is "density".

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

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{fname_root}_{date}_{time}.pdf"))

    # show the plot
    plt.show()

    return


# Define a function to plot the stability as boxplots
def plot_stability_boxplots(
    ensemble: pd.DataFrame,
    var_name: str,
    label: str,
    lead_name: str = "lead",
    fontsize: int = 12,
    fig_size: tuple = (10, 10),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
    fname_root: str = "density_boxplot",
):
    """
    Function which plots boxplots of the distribution for the different lead
    times.

    Inputs:
    -------

    ensemble: pd.DataFrame
        The DataFrame containing the ensemble data. With columns for the
        ensemble member, lead time, and the variable of interest.

    var_name: str
        The name of the variable of interest.

    label: str
        The label for the variable of interest.

    lead_name: str
        The name of the lead time column. Default is "lead".

    fontsize: int
        The fontsize for the labels. Default is 12.

    fig_size: tuple
        The figure size. Default is (10, 10).

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    fname_root: str
        The root name for the file. Default is "density_boxplot".

    Returns:
    -------

    None

    """

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Get the unique lead times
    leads = sorted(ensemble[lead_name].unique())

    # Loop over the lead times
    for lead in tqdm(leads, desc="Plotting boxplots"):
        # Extract the data for the lead time
        data = ensemble[ensemble[lead_name] == lead][var_name]

        # Plot the boxplot
        ax.boxplot(data, positions=[lead], widths=0.8)

        # if the lead is the first one
        if lead == leads[0]:
            # plot a horixzontal line of the mean
            ax.axhline(
                data.mean(), color="black", linestyle="--", label="First lead mean"
            )

    # Set the x-axis label
    ax.set_xlabel("Lead time", fontsize=fontsize)

    # Set the y-axis label
    ax.set_ylabel(label, fontsize=fontsize)

    # Set the x-ticks
    ax.set_xticks(leads)

    # Set the x-tick labels
    ax.set_xticklabels(leads)

    # Set the current time
    date_time_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{fname_root}_{date_time_now}.pdf"), dpi=600)

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
    fname_root: str = "fidelity",
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

    fname_root: str
        The root name for the saved files. Default is "fidelity".

    Returns

    None

    """

    # print the mean bias
    print(
        f"The mean bias is {np.mean(model_df[model_val_name] - np.mean(obs_df[obs_val_name]))}"
    )

    # print the spread bias
    print(
        f"The spread bias is {np.std(model_df[model_val_name]) - np.std(obs_df[obs_val_name])}"
    )

    # Set up the model stats dict
    model_stats = {
        "mean": [],
        "sigma": [],
        "skew": [],
        "kurt": [],
    }

    # print the len model_time_name unique
    print(
        f"The number of unique model times is {len(model_df[model_time_name].unique())}"
    )
    print(f"The number of unique obs times is {len(obs_df[obs_time_name].unique())}")

    # Assert that the len of unique init in model_df
    # is equal to the len of unique year in obs_df
    # assert len(model_df[model_time_name].unique()) == len(
    #     obs_df[obs_time_name].unique()
    # ), "The number of unique initialisation dates in the model data must be equal to the number of unique years in the observations."

    # Set up the number of unique initialisation dates
    n_years = len(model_df[model_time_name].unique())

    # Set up the number of unique ensemble members
    n_members = len(model_df[model_member_name].unique())

    # if the model_lead_name is not None
    if model_lead_name is not None:
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

    if model_lead_name is not None:
        # Extract the unique model leads
        model_leads = model_df[model_lead_name].unique()

    # Create the indexes for the ensemble members
    member_idx = np.arange(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(range(nboot), desc="Calculating bootstrap statistics"):
        # Set up random indices for the ensemble members
        idx_time_this = range(0, n_years)

        # Create an empty array to store the bootstrapped values
        model_boot = np.zeros([n_years])

        # Set the year index to 0
        idx_year = 0

        # Loop over the number of years
        # Randomly select an ensemble member and lead time for
        # each year
        # But year range stays constant
        for itime in idx_time_this:
            # Set up random indices for the ensemble members
            idx_ens_this = random.choices(member_idx)

            # Find the time at the itime index
            model_time_this = model_times[itime]

            # Find the name for the member at this index
            model_member_this = model_members[idx_ens_this]

            # if model_lead_name is not None
            if model_lead_name is not None:
                # Set up a random choice for the lead time
                idx_lead_this = random.choices(range(n_leads))

                # Find the name for the lead at this index
                model_lead_this = model_leads[idx_lead_this]

                # Extract the model data for the year and ensemble members
                model_data = model_df[
                    (model_df[model_time_name] == model_time_this)
                    & (model_df[model_member_name] == model_member_this[0])
                    & (model_df[model_lead_name] == model_lead_this[0])
                ][model_val_name].values
            else:
                # Extract the model data for the year and ensemble members
                model_data = model_df[
                    (model_df[model_time_name] == model_time_this)
                    & (model_df[model_member_name] == model_member_this[0])
                ][model_val_name].values

            # Check if model_data is empty
            if model_data.size == 0:
                # print(f"No data available for time {model_time_this}, member {model_member_this}, lead {model_lead_this if model_lead_name else 'N/A'}")
                continue  # Skip this iteration if no data is available

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
    plt.savefig(os.path.join(save_dir, f"{fname_root}_{date}_{time}.pdf"))

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
    fname_prefix: str = "events",
    ind_months_flag: bool = False,
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

    fname_prefix: str
        The prefix for the saved files. Default is "events".

    ind_months_flag: bool
        Whether to plot the individual months. Default is False.

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

    # # create an empty array to store the values of slope and intercept
    # slopes = np.zeros(len(model_df[model_member_name].unique()))
    # intercepts = np.zeros(len(model_df[model_member_name].unique()))

    # # loop over the unique members
    # for i, member in enumerate(model_df[model_member_name].unique()):
    #     # Extract the data for the member and the first lead = 1
    #     model_data_this = model_df[
    #         (model_df[model_member_name] == member) & (model_df[model_lead_name] == 1)
    #     ]

    #     # Fit a linear trend to the model data
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(
    #         model_data_this[model_time_name], model_data_this[model_val_name]
    #     )

    #     # Store the slope and intercept
    #     slopes[i] = slope
    #     intercepts[i] = intercept

    #     # Plot the model trend as a dashed line
    #     ax.plot(
    #         model_data_this[model_time_name],
    #         intercept + slope * model_data_this[model_time_name],
    #         color="grey",
    #         linestyle="--",
    #         label="model trend" if i == 0 else None,
    #     )

    # # print the intercepts
    # print(intercepts.flatten().mean())

    # # print the slopes
    # print(slopes.flatten().mean())

    # # print the slopes 5%tile
    # print(f"2.5%tile slope: {np.quantile(slopes, 0.025)}")
    # print(f"97.5%tile slope: {np.quantile(slopes, 0.975)}")

    # # print the slopes 95%tile
    # print(np.quantile(slopes, 0.95))

    # # plot the mean trend as a red dashed line
    # ax.plot(
    #     model_data_this[model_time_name],
    #     intercepts.flatten().mean()
    #     + slopes.flatten().mean() * model_data_this[model_time_name],
    #     color="red",
    #     linestyle="--",
    #     label="model mean trend",
    # )

    # # Quantify this trend line
    # trend_line = (
    #     intercepts.flatten().mean()
    #     + slopes.flatten().mean() * model_data_this[model_time_name]
    # )

    # # subtract this trend line from the model data
    # model_data_this[f"{model_val_name}_detrended"] = (
    #     model_data_this[model_val_name] - trend_line
    # )

    # # subtract this trend line from the obs data
    # obs_df[f"{obs_val_name}_detrended"] = obs_df[obs_val_name] - trend_line

    # # modify the obs_val_name
    # obs_val_name = f"{obs_val_name}_detrended"
    # model_val_name = f"{model_val_name}_detrended"

    # if the model time name column is not formatted as datetime
    if not isinstance(model_df[model_time_name].values[0], np.datetime64):
        # convert the column to datetime (years) from ints for the year
        model_df[model_time_name] = pd.to_datetime(
            model_df[model_time_name], format="%Y"
        )

    # if the units are kelvin (we can tell by the values)
    # convert both model and obs to celsius
    if model_df[model_val_name].max() > 100:
        model_df[model_val_name] = model_df[model_val_name] - 273.15
        obs_df[obs_val_name] = obs_df[obs_val_name] - 273.15

    # Set up the counts
    n_extreme = 0
    n_unseen = 0

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

        # Plot the points above the 20th percentile
        ax.scatter(
            model_data[model_data_above20][model_time_name],
            model_data[model_data_above20][model_val_name],
            color="grey",
            alpha=0.2,
            label=model_name if i == 0 else None,
        )

        # Plot the points below the 20th percentile
        ax.scatter(
            model_data[model_data_below20][model_time_name],
            model_data[model_data_below20][model_val_name],
            color="blue",
            alpha=0.2,
            label="Extreme events" if i == 0 else None,
        )

        # Plot the points below the minimum of the obs
        ax.scatter(
            model_data[model_data_below_obs_min_bool][model_time_name],
            model_data[model_data_below_obs_min_bool][model_val_name],
            color="red",
            alpha=0.8,
            marker="x",
            label="Unseen events" if i == 0 else None,
        )

        # add to the counts
        n_extreme += model_data_below20.sum()
        n_unseen += model_data_below_obs_min_bool.sum()

    # # plot the trend line for the 5%tile slope
    # ax.plot(
    #     model_data_this[model_time_name],
    #     intercepts.flatten().mean() + np.quantile(slopes, 0.05) * model_data_this[model_time_name],
    #     color="red",
    #     linestyle="--",
    #     label="model 5% quantile",
    # )

    # # plot the trend line for the 95%tile slope
    # ax.plot(
    #     model_data_this[model_time_name],
    #     intercepts.flatten().mean() + np.quantile(slopes, 0.95) * model_data_this[model_time_name],
    #     color="red",
    #     linestyle="--",
    #     label="model 95% quantile",
    # )

    # # Calculate the 5-95% range for the trendlines
    # x_values = np.arange(model_df[model_time_name].min(), model_df[model_time_name].max())
    # lower_bound = np.quantile(slopes, 0.05) * x_values + np.quantile(intercepts, 0.05)
    # upper_bound = np.quantile(slopes, 0.95) * x_values + np.quantile(intercepts, 0.95)

    # # Plot the 5-95% range
    # ax.fill_between(x_values, lower_bound, upper_bound, color='grey', alpha=0.5)

    # # Quantify the 5% and 95% quantiles of the model data
    # slopes_05 = np.quantile(slopes.flatten(), 0.05)
    # slopes_95 = np.quantile(slopes.flatten(), 0.95)

    # intercepts_05 = np.quantile(intercepts.flatten(), 0.05)
    # intercepts_95 = np.quantile(intercepts.flatten(), 0.95)

    # # Plot the 5% and 95% quantiles of the model data
    # ax.plot(
    #     model_data_this[model_time_name],
    #     intercepts_05 + slopes_05 * model_data_this[model_time_name],
    #     color="grey",
    #     linestyle="--",
    #     label="model 5% quantile",
    # )

    # ax.plot(
    #     model_data_this[model_time_name],
    #     intercepts_95 + slopes_95 * model_data_this[model_time_name],
    #     color="grey",
    #     linestyle="--",
    #     label="model 95% quantile",
    # )

    # if the individual months flag is true
    if ind_months_flag:
        print("Plotting the individual months")
        # assert that the time axis of the obs is datetime
        assert isinstance(obs_df[obs_time_name].values[0], np.datetime64)

        # shift each row value back by 3 months
        obs_df[obs_time_name] = obs_df[obs_time_name] - pd.DateOffset(months=3)

        # Assuming obs_df is your DataFrame and obs_time_name is the column name
        obs_df[obs_time_name] = pd.to_datetime(obs_df[obs_time_name]).dt.year

        # Convert the year back to a datetime object with only the year component
        obs_df[obs_time_name] = pd.to_datetime(obs_df[obs_time_name], format="%Y")

        # Print the head of the dataframe
        print(obs_df.head())

        # print the head of the dataframe
        print(obs_df.head())

    # Plot the observed data
    ax.scatter(
        obs_df[obs_time_name],
        obs_df[obs_val_name],
        color="black",
        label="ERA5",
    )

    # # fit a linear trend to the observations
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     obs_df[obs_time_name], obs_df[obs_val_name]
    # )

    # # plot the obs trend as a dashed line
    # ax.plot(
    #     obs_df[obs_time_name],
    #     intercept + slope * obs_df[obs_time_name],
    #     color="black",
    #     linestyle="--",
    #     label="ERA5 trend",
    # )

    # include a textbox in the bottom right hand corner
    # with N_extreme and N_unseen
    ax.text(
        0.95,
        0.05,
        f"N extreme: {n_extreme}\nN unseen: {n_unseen}",
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5),
        zorder=100,
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

    # tight layout
    plt.tight_layout()

    # # Show the plot
    # plt.show()

    # Set the current time
    now = datetime.now()

    # Set the current date
    date = now.strftime("%Y-%m-%d")

    # Set the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{fname_prefix}_{date}_{time}.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

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
    low_bad: bool = True,
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

    low_bad: bool
        Whether the lower values are bad. Default is True.

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns
    -------

    None

    """

    # print the model mean
    print(f"The model mean is {model_df[model_val_name].mean()}")
    # print the obs mean
    print(f"The obs mean is {obs_df[obs_val_name].mean()}")

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

    if low_bad:
        # plot a horizontal line for the 20th percentil of the obs
        axs[0].axhline(
            np.quantile(obs_df[obs_val_name], 0.2), color="blue", linestyle="--"
        )

        # plot a horizontal line for the minimum of the obs
        axs[0].axhline(obs_df[obs_val_name].min(), color="blue", linestyle="-.")
    else:
        # plot a horizontal line for the 80th percentil of the obs
        axs[0].axhline(
            np.quantile(obs_df[obs_val_name], 0.8), color="blue", linestyle="--"
        )

        # plot a horizontal line for the maximum of the obs
        axs[0].axhline(obs_df[obs_val_name].max(), color="blue", linestyle="-.")

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


def create_masked_matrix(
    country,
    cube,
) -> np.ndarray:
    """
    Create a masked matrix for the specified country.

    Parameters
    ----------

    country: str
        The name of the country to create the mask for.

    cube: iris.cube.Cube
        The cube to create the mask for.

    Returns
    -------

    MASK_MATRIX_RESHAPE: np.ndarray
        The masked matrix of 1s and 0s for the specified country.

    """
    # apply the mask
    LONS, LATS = iris.analysis.cartography.get_xy_grids(cube[0])
    x, y = LONS.flatten(), LATS.flatten()

    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    # Shapely treats lons between -180 and 180
    # so we need to convert the lons to this
    # if there are no lons below 0 then we can just convert the lons
    if np.min(x) >= 0:
        x = x - 180
        LONS = LONS - 180

    MASK_MATRIX_TMP = np.zeros([len(x), 1])
    country_shapely = []
    for country_record in shpreader.Reader(countries_shp).records():
        if country_record.attributes["NAME"][0:14] == country:
            print("Found Country " + country)
            country_shapely.append(country_record.geometry)

    # Create a mask for the country
    for i in range(0, len(x)):
        point = shapely.geometry.Point(x[i], y[i])
        if country_shapely[0].contains(point) == True:
            MASK_MATRIX_TMP[i, 0] = 1.0

    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX_TMP, (np.shape(LONS)))

    # if the country is the UK then we want to mask out NI
    # to constrain to GB
    if country == "United Kingdom":
        lats, lons = np.unique(LATS), np.unique(LONS)
        for i in range(len(lats)):
            for j in range(len(lons)):
                if (lats[i] < 55.3) and (lats[i] > 54.0):
                    if (lons[j]) < -5.0:  # convert back to -180 to 180 scale
                        # find the corresponding indices in the LATS and LONS arrays
                        indices = np.argwhere((LATS == lats[i]) & (LONS == lons[j]))
                        # set the mask value to 0 at these indices
                        for idx in indices:
                            MASK_MATRIX_RESHAPE[tuple(idx)] = 0.0

    return MASK_MATRIX_RESHAPE


# Define a function to plot the events time series using boxplots
# for the hindcast data
def plot_events_ts_errorbars(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    X1_col_model: str,
    X2_col_model: str,
    Y_col: str,
    ylabel: str,
    X1_col_obs: str = "sfcWind_mon",
    X2_col_obs: str = "tas_mon",
    X1_col_obs_dt: str = "sfcWind_mon_dt",
    X2_col_obs_dt: str = "tas_mon_dt",
    num_trials: int = 1000,
    block_length: int = 10,
    nboot: int = 1000,
    model_name: str = "HadGEM3-GC31-MM",
    obs_time_name: str = "year",
    model_time_name: str = "init",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
    figsize: tuple = (10, 10),
    low_bad: bool = True,
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

    figsize: tuple
        The figure size. Default is (10, 10).

    low_bad: bool
        Whether the lower values are bad. Default is True.

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots/".

    Returns
    -------

    None

    """

    # Sample the uncertainty in the linear fit
    # and due to the limited sample size of the obs
    ntimes = len(obs_df[obs_time_name].unique())

    # Get the number of blocks
    nblocks = int(ntimes / block_length)

    # if the nblocks * block is less than the ntimes
    if (nblocks * block_length) < ntimes:
        # add one to the nblocks
        nblocks = nblocks + 1

    # set up the index for time
    index_time = range(ntimes - block_length + 1)

    # set up the empty array for the bootstrapped data
    X1_boot_full = np.zeros((nboot, ntimes))
    X2_boot_full = np.zeros((nboot, ntimes))
    Y_boot_full = np.zeros((nboot, ntimes))

    # Set up an empty array for the residuals
    residuals_boot = np.zeros((nboot, ntimes))

    # Set up an empty array for the spread
    res_spread_boot = np.zeros(nboot)

    # set up an empty array for the r2 and rmse values
    r2_boot = np.zeros(nboot)
    rmse_boot = np.zeros(nboot)

    # loop over the nboot
    for iboot in tqdm(np.arange(nboot)):
        # Select starting indices for the blocks
        if iboot == 0:
            ind_time_this = range(0, ntimes, block_length)
        else:  # random samples
            ind_time_this = np.array(
                [random.choice(index_time) for _ in range(nblocks)]
            )

        # Set up the shape of the bootstrapped data
        X1_boot = np.zeros(ntimes)
        X2_boot = np.zeros(ntimes)

        # Same for the predictand
        Y_boot = np.zeros(ntimes)

        # reset time index
        itime = 0

        # loop over the indices
        for ithis in ind_time_this:
            # Set up the block index
            ind_block = np.arange(ithis, ithis + block_length)

            # if the block index is greater than the number of times
            # then subtract the number of times from the block index
            ind_block[(ind_block > ntimes - 1)] = (
                ind_block[(ind_block > ntimes - 1)] - ntimes
            )

            # Restrict the block index to the minimum of the block length
            ind_block = ind_block[: min(block_length, ntimes - itime)]

            # loop over the blocks
            for iblock in ind_block:
                # Set up the bootstrapped data
                X1_boot[itime] = obs_df[X1_col_obs].values[iblock]
                X2_boot[itime] = obs_df[X2_col_obs].values[iblock]
                Y_boot[itime] = obs_df[Y_col].values[iblock]

                # increment the time index
                itime += 1

        # Append the data
        X1_boot_full[iboot, :] = X1_boot
        X2_boot_full[iboot, :] = X2_boot
        Y_boot_full[iboot, :] = Y_boot

        if iboot == 0:
            ind_time_this = range(0, ntimes, block_length)

            X_boot_first = np.column_stack((X1_boot, X2_boot))

            # Fit the model
            model_first = LinearRegression().fit(X_boot_first, Y_boot)

            # predict the values of Y
            Y_pred_first = model_first.predict(X_boot_first)

            # calculate and append the r2 and rmse values
            r2_boot_first = model_first.score(X_boot_first, Y_boot)

            # Calculate the residuals
            # the difference between the actual and predicted values
            residuals_boot_first = Y_pred_first - Y_boot

            # Calculate the spread of the residuals
            res_spread_boot_first = np.std(residuals_boot_first)
        else:  # random samples
            # print the shape of the bootstrapped data
            # # print(X_boot_full.shape)
            # # print(y_boot_full.shape)
            # print(np.shape(X_boot))
            # print(np.shape(y_boot))

            # # # print the shape of X1_boot and X2_boot
            # print(np.shape(X1_boot))
            # print(np.shape(X2_boot))

            # Set up the predictors
            X_boot = np.column_stack((X1_boot, X2_boot))

            # Fit the model
            model = LinearRegression().fit(X_boot, Y_boot)

            # predict the values of Y
            Y_pred = model.predict(X_boot)

            # calculate and append the r2 and rmse values
            r2_boot[iboot] = model.score(X_boot, Y_boot)
            rmse_boot[iboot] = np.sqrt(mean_squared_error(Y_boot, Y_pred))

            # Calculate the residuals
            # the difference between the actual and predicted values
            residuals_boot[iboot, :] = Y_pred - Y_boot

            # Calculate the spread of the residuals
            res_spread_boot[iboot] = np.std(residuals_boot[iboot, :])

    # Quantify the 95th percentile value of the residual spread
    # ie. the upper bounds of the residuals spread
    res_spread_95 = np.percentile(res_spread_boot, 95)

    # create a stochastic fit
    # with the upper bounds of the residuals spread
    stoch_95 = np.random.normal(0, res_spread_95, size=(len(model_df), num_trials))

    # create stoch_95 for the obs
    stoch_95_obs = np.random.normal(0, res_spread_95, size=(len(obs_df), num_trials))

    # set up the MLR model
    # which we'll apply to the detrended obs data
    # if the columns exist
    if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
        print("Detrended columns exist")

        # set up the predictors
        X_obs = obs_df[[X1_col_obs_dt, X2_col_obs_dt]]

        # Predict the values of Y
        obs_df[f"{Y_col}_pred"] = model_first.predict(X_obs)
    else:
        print("Detrended columns do not exist")

    # Now set up the MLR model
    # which predicts CLEARHEADS DnW
    # given model data
    X_model = model_df[[X1_col_model, X2_col_model]]

    # print the shape of X_model
    print(f"The shape of X_model is {X_model.shape}")

    # predict the values of Y
    model_df[f"{Y_col}_pred"] = model_first.predict(X_model)

    # # print the model mean
    # print(f"The model mean is {model_df[model_val_name].mean()}")
    # # print the obs mean
    # print(f"The obs mean is {obs_df[obs_val_name].mean()}")

    # Set up the years
    years = obs_df[obs_time_name].unique()

    # Set up the figure with two subplots
    # Set up the figure with two subplots with different widths
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [8, 1]},
    )

    if low_bad:
        if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
            axs[0].axhline(
                np.quantile(obs_df[f"{Y_col}_pred"], 0.2), color="blue", linestyle="--"
            )

            axs[0].axhline(obs_df[f"{Y_col}_pred"].min(), color="blue", linestyle="-.")
        else:
            # plot a horizontal line for the 20th percentil of the obs
            axs[0].axhline(
                np.quantile(obs_df[obs_val_name], 0.2), color="blue", linestyle="--"
            )

            # plot a horizontal line for the minimum of the obs
            axs[0].axhline(obs_df[obs_val_name].min(), color="blue", linestyle="-.")

        # Do the same for the model data
        # plot a horizontal line for the 20th percentil of the model data
        axs[0].axhline(
            np.quantile(model_df[f"{Y_col}_pred"], 0.2), color="red", linestyle="--"
        )

        # plot a horizontal line for the minimum of the model data
        axs[0].axhline(model_df[f"{Y_col}_pred"].min(), color="red", linestyle="-.")

    else:
        if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
            axs[0].axhline(
                np.quantile(obs_df[f"{Y_col}_pred"], 0.8), color="blue", linestyle="--"
            )

            axs[0].axhline(obs_df[f"{Y_col}_pred"].max(), color="blue", linestyle="-.")
        else:
            # plot a horizontal line for the 80th percentil of the obs
            axs[0].axhline(
                np.quantile(obs_df[obs_val_name], 0.8), color="blue", linestyle="--"
            )

            # plot a horizontal line for the maximum of the obs
            axs[0].axhline(obs_df[obs_val_name].max(), color="blue", linestyle="-.")

        # Do the same for the model data
        # plot a horizontal line for the 80th percentil of the model data
        axs[0].axhline(
            np.quantile(model_df[f"{Y_col}_pred"], 0.8), color="red", linestyle="--"
        )

        # plot a horizontal line for the maximum of the model data
        axs[0].axhline(model_df[f"{Y_col}_pred"].max(), color="red", linestyle="-.")

    # # print the shape of stochastic 95
    # print(f"The shape of the stochastic 95 is {stoch_95.shape}")

    # # print the shape of model_df[f"{Y_col}_pred"].values
    # print(
    #     f"The shape of the model_df[f'{Y_col}_pred'].values is {model_df[f'{Y_col}_pred'].values[:, np.newaxis].shape}"
    # )

    # # print the head of the model_df
    # print(f"The head of the model_df is {model_df.head()}")

    # Set the index of the model df to be
    # init year, member, and lead
    model_df.set_index(
        [model_time_name, model_member_name, model_lead_name], inplace=True
    )

    # Add the random trials to the deterministic model time series
    trials_95 = pd.DataFrame(
        model_df[f"{Y_col}_pred"].values[:, np.newaxis] + stoch_95,
        index=model_df.index,
        columns=range(num_trials),
    )

    # if the dt columsn exist
    if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
        # add the randome trials to the deterministic obs time series
        trials_95_obs = pd.DataFrame(
            obs_df[f"{Y_col}_pred"].values[:, np.newaxis] + stoch_95_obs,
            index=obs_df.index,
            columns=range(num_trials),
        )

        # group the trials by the year
        trials_95_obs_grouped = trials_95_obs.groupby(obs_df.index).mean()

        # Find the 5th and 95th percentiles of the trials
        p05_95_obs, p95_95_obs = [
            trials_95_obs_grouped.T.quantile(q) for q in [0.05, 0.95]
        ]

        # convert to a dataframe
        p05_95_obs = p05_95_obs.to_frame()
        p95_95_obs = p95_95_obs.to_frame()

        # reset the index of p05_95
        p05_95_obs.reset_index(inplace=True)
        p95_95_obs.reset_index(inplace=True)

        # rename the columns
        p05_95_obs.columns = [model_time_name, f"{Y_col}_pred"]
        p95_95_obs.columns = [model_time_name, f"{Y_col}_pred"]

        # turn them into series
        p05_95_obs[[obs_time_name]] = p05_95_obs[model_time_name].apply(
            lambda x: pd.Series(x)
        )
        p95_95_obs[[obs_time_name]] = p95_95_obs[model_time_name].apply(
            lambda x: pd.Series(x)
        )

    # # print the head of trials_95
    # print(f"The head of trials_95 is {trials_95.head()}")

    # # print the shape of trials_95
    # print(f"The shape of trials_95 is {trials_95.shape}")

    # group the trials by the year
    trials_95_grouped = trials_95.groupby(model_df.index).mean()

    # print(f"The shape of the grouped trials_95 is {trials_95_grouped.shape}")

    # Find the 5th and 95th percentiles of the trials
    p05_95, p95_95 = [trials_95_grouped.T.quantile(q) for q in [0.05, 0.95]]

    # # # print the shape of the 5th and 95th percentiles
    # print(f"The shape of the 5th percentile is {p05_95.shape}")
    # # print(f"The shape of the 95th percentile is {p95_95.shape}")

    # # # print the values
    # print(f"The 5th percentile values are {p05_95}")
    # # print(f"The 95th percentile values are {p95_95}")

    # # print the type of the 5th percentile
    # print(f"The type of the 5th percentile is {type(p05_95)}")

    # convert to a dataframe
    p05_95 = p05_95.to_frame()
    p95_95 = p95_95.to_frame()

    # reset the index of p05_95
    p05_95.reset_index(inplace=True)
    p95_95.reset_index(inplace=True)

    # rename the columns
    p05_95.columns = [model_time_name, f"{Y_col}_pred"]
    p95_95.columns = [model_time_name, f"{Y_col}_pred"]

    # # print the head of p05_95
    # print(f"The head of p05_95 is {p05_95.head()}")

    p05_95[[model_time_name, model_member_name, model_lead_name]] = p05_95[
        model_time_name
    ].apply(lambda x: pd.Series(x))
    p95_95[[model_time_name, model_member_name, model_lead_name]] = p95_95[
        model_time_name
    ].apply(lambda x: pd.Series(x))

    # # drop the model time name
    # p05_95.drop(model_time_name, axis=1, inplace=True)

    # # print the head of p05_95
    # print(f"The head of p05_95 is {p05_95.head()}")

    # # print the head of the model df
    # print(f"The head of the model df is {model_df.head()}")

    # sys.exit()

    # # reset the index of p05_95 and p95_95
    # p05_95.reset_index(inplace=True)
    # p95_95.reset_index(inplace=True)

    # # # print the head of p05_95
    # print(f"The head of p05_95 is {p05_95.head()}")

    # sys.exit()

    # reset the index of model df
    model_df.reset_index(inplace=True)

    # if the detrended columns exist
    if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
        axs[0].scatter(
            obs_df[obs_time_name],
            obs_df[f"{Y_col}_pred"],
            color="blue",
            marker="x",
            label="ERA5",
            zorder=2,
        )

        # quantify the error bars
        yerr = abs(p95_95_obs - p05_95_obs)

        # # print the values of yerr
        # print(f"The values of yerr are {yerr}")

        # assign the error to the obs_df
        # obs_df[f"{Y_col}_pred_err"] = (
        #     obs_df[f"{Y_col}_pred"].values - yerr[f"{Y_col}_pred"].values / 2
        # )

        # # print the values of pred err
        # print(f"The values of pred err are {obs_df[f'{Y_col}_pred_err']}")

        # plot the error bars for the obs data
        axs[0].errorbar(
            obs_df[obs_time_name],
            obs_df[f"{Y_col}_pred"],
            yerr=yerr[f"{Y_col}_pred"] / 2,
            fmt="o",
            ecolor="blue",
            color="blue",
            alpha=0.5,
            capsize=2,
        )

    else:
        # Plot the observed data as blue crosses on the first subplot
        axs[0].scatter(
            years,
            obs_df[obs_val_name],
            color="blue",
            marker="x",
            label="ERA5",
            zorder=2,
        )

    # Loop over the unique members
    # and plot the scatter points with error bars
    for i, member in enumerate(model_df[model_member_name].unique()):
        # print("iteration", i)
        # Seperate the data based on the condition
        model_data = model_df[model_df[model_member_name] == member]

        # Subset the p05_95 and p95_95 data
        # by member
        p05_95_member = p05_95[p05_95[model_member_name] == member]
        p95_95_member = p95_95[p95_95[model_member_name] == member]

        # quantify the error bars
        yerr = abs(p95_95_member - p05_95_member)

        # # print the shape of yerr
        # print(yerr.shape)

        # # print the shape of model_data y_col_pred
        # print(model_data[f"{Y_col}_pred"].shape)

        if low_bad:
            # Seperate data by threshold
            model_data_below20 = (
                obs_df[obs_val_name].min() < model_data[f"{Y_col}_pred"]
            ) & (model_df[f"{Y_col}_pred"] < np.quantile(obs_df[obs_val_name], 0.2))

            # Above the threshold
            model_data_above20 = (
                model_data[f"{Y_col}_pred"] >= obs_df[obs_val_name].min()
            ) & ~model_data_below20

            # below the minimum of the obs
            model_data_below_obs_min_bool = (
                model_data[f"{Y_col}_pred"] < obs_df[obs_val_name].min()
            )

            # Clear up the names
            very_bad_events = model_data_below_obs_min_bool
            bad_events = model_data_below20
            events = model_data_above20
        else:

            # # if i = 0
            # if i == 0:
            #     # print the values of the model data
            #     print(f"The values of the model data are {model_data[f'{Y_col}_pred']}")
            #     # print the values of the yerr
            #     print(f"The values of the yerr are {yerr[f'{Y_col}_pred'] / 2}")

            #     # print the shape of the model data
            #     print(f"The shape of the model data is {model_data[Y_col+'_pred'].shape}")
            #     print(f"The shape of the yerr is {yerr[f'{Y_col}_pred'].shape}")

            # Get the data - error
            model_data[f"{Y_col}_pred_err"] = (
                model_data[f"{Y_col}_pred"].values - yerr[f"{Y_col}_pred"].values / 2
            )

            # # print the values of model data pred err
            # print(f"The values of model data pred err are {model_data[f'{Y_col}_pred_err']}")

            # # if i is 0
            # # print the values
            # if i == 0:
            #     print(f"values of ycol pred are {model_data[f'{Y_col}_pred']}")
            #     print(f"values of ycol pred err are {model_data[f'{Y_col}_pred_err']}")

            # Seperate data by threshold
            model_data_above_obs_max = (
                model_data[f"{Y_col}_pred_err"] >= obs_df[obs_val_name].max()
            )

            # Model data above the 80th percentile but below the maximum of the obs
            model_data_above80 = (
                np.quantile(obs_df[obs_val_name], 0.8)
                <= model_data[f"{Y_col}_pred_err"]
            ) & (model_data[f"{Y_col}_pred"] < obs_df[obs_val_name].max())

            # Model data below the 80th percentile
            model_data_below80 = model_data[f"{Y_col}_pred"] < np.quantile(
                obs_df[obs_val_name], 0.8
            )

            # Clear up the names
            very_bad_events = model_data_above_obs_max
            bad_events = model_data_above80
            events = model_data_below80

        # Plot the points below the minimum of the obs
        axs[0].scatter(
            model_data.loc[very_bad_events, model_time_name],
            model_data.loc[very_bad_events, f"{Y_col}_pred"],
            color="red",
            alpha=0.8,
            label="UNSEEN Events" if i == 0 else None,
        )

        # Plot the points below the 20th percentile
        axs[0].scatter(
            model_data.loc[bad_events, model_time_name],
            model_data.loc[bad_events, f"{Y_col}_pred"],
            color="orange",
            alpha=0.8,
            label="Extreme Events" if i == 0 else None,
        )

        # Plot the points above the 20th percentile
        axs[0].scatter(
            model_data.loc[events, model_time_name],
            model_data.loc[events, f"{Y_col}_pred"],
            color="grey",
            alpha=0.8,
            label="Events" if i == 0 else None,
        )

        # # Plot the error bars
        # axs[0].errorbar(
        #     model_data[model_time_name],
        #     model_data[f"{Y_col}_pred"],
        #     yerr=yerr[f"{Y_col}_pred"]/2, # divide by 2 to get the error bars
        #     fmt="o",
        #     color="black",
        #     alpha=0.5,
        # )

        # if the first value of the index of yerr is not 0
        # if yerr.index[0] != 0:
        #     # reset the index of yerr
        #     yerr = yerr.reset_index(drop=True)

        # if not all indexes of model_data_above_obs_max are in yerr's index
        if not model_data_above_obs_max.index.isin(yerr.index).all():
            # reset the index of yerr
            yerr_reset = yerr.reset_index(drop=True)

            # reset the index of the model data above obs max
            model_data_above_obs_max_reset = model_data_above_obs_max.reset_index(
                drop=True
            )

            # subset yerr to only the values above the obs max
            yerr_subset = yerr_reset.loc[model_data_above_obs_max_reset]
        else:
            # subset yerr to only the values above the obs max
            yerr_subset = yerr.loc[model_data_above_obs_max]

        # # print the values of the yerr subset model pred
        # print(f"The values of the yerr subset model pred are {yerr_subset[f'{Y_col}_pred']}")

        # # print the values of yerr_subset
        # print(f"The values of yerr_subset are {yerr_subset}")

        # Plot the error bars for the model data above obs max
        axs[0].errorbar(
            model_data.loc[model_data_above_obs_max, model_time_name],
            model_data.loc[model_data_above_obs_max, f"{Y_col}_pred"],
            yerr=yerr_subset[f"{Y_col}_pred"] / 2,  # divide by 2 to get the error bars
            fmt="o",
            ecolor="red",  # make the error bars red
            color="red",  # make the points red
            alpha=0.5,
            capsize=2,  # set the capsize to 5
        )

    # if the columns are not empty
    if X1_col_obs_dt in obs_df.columns and X2_col_obs_dt in obs_df.columns:
        # plot the boxplot for the observed data
        axs[1].boxplot(
            obs_df[f"{Y_col}_pred"],
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

        # print the worst obs year (i.e. highest demand net wind)
        print(
            f"The worst observed year is {obs_df[obs_time_name].values[np.argmax(obs_df[f'{Y_col}_pred'])]}"
        )

    else:
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

        # print the worst observed year
        print(
            f"The worst observed year is {obs_df[obs_time_name].values[np.argmax(obs_df[obs_val_name])]}"
        )

    # also include a red boxplot for the model data
    axs[1].boxplot(
        model_df[f"{Y_col}_pred"],
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

    # print the worst model init year, member, and lead
    print(
        f"The worst model init year is {model_df[model_time_name].values[np.argmax(model_df[f'{Y_col}_pred'])]}"
    )
    print(
        f"The worst model member is {model_df[model_member_name].values[np.argmax(model_df[f'{Y_col}_pred'])]}"
    )
    print(
        f"The worst model lead is {model_df[model_lead_name].values[np.argmax(model_df[f'{Y_col}_pred'])]}"
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


# Define a function to apply the detrending
def apply_detrend(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str = "year",
    model_time_name: str = "init_year",
    model_member_name: str = "member",
    model_lead_name: str = "lead",
) -> pd.DataFrame:
    """
    Applies a detrend using Gillian's pivot method. Trend is quantified as
    the ensemble mean trend from the model data. This same trend is then
    removed from both the model and the observations data.

    Parameters
    ==========

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
        The name of the model time column. Default is "init_year".

    model_member_name: str
        The name of the model member column. Default is "member".

    model_lead_name: str
        The name of the model lead time column. Default is "lead".

    Returns
    =======

    pd.DataFrame
        A DataFrame containing the detrended observations and model data.

    """

    # Create empty arrays to store the slopes for different member lead combinations
    slopes = np.zeros(
        [
            len(model_df[model_member_name].unique()),
            len(model_df[model_lead_name].unique()),
        ]
    )
    intercepts = np.zeros(
        [
            len(model_df[model_member_name].unique()),
            len(model_df[model_lead_name].unique()),
        ]
    )

    # Loop over the unique members
    for m, member in enumerate(model_df[model_member_name].unique()):
        for l, lead in enumerate(model_df[model_lead_name].unique()):
            # Select the data for this member and lead
            data = model_df[
                (model_df[model_member_name] == member)
                & (model_df[model_lead_name] == lead)
            ]

            # Fit a linear trend to the model data
            slope, intercept, _, _, _ = linregress(
                data[model_time_name], data[model_val_name]
            )

            # Store the slope and intercept
            slopes[m, l] = slope
            intercepts[m, l] = intercept

    # Print the mean slope
    print(f"The mean slope is {np.mean(slopes.flatten())}")

    # print the 2.5th and 97.5th percentiles of the slopes
    print(
        f"The 2.5th percentile of the slopes is {np.percentile(slopes.flatten(), 2.5)}"
    )
    print(
        f"The 97.5th percentile of the slopes is {np.percentile(slopes.flatten(), 97.5)}"
    )

    # quantify the slope of the observations
    slope_obs, intercept_obs, _, _, _ = linregress(
        obs_df[obs_time_name].dt.year.astype(int).values, obs_df[obs_val_name]
    )

    # print the slope of the observations
    print(f"The slope of the observations is {slope_obs}")

    # Set up the trend line as the mean of slopes flat and intercepts flat
    trend_line = np.mean(slopes.flatten()) * model_df[model_time_name].values + np.mean(
        intercepts.flatten()
    )

    # Calculate the value of the trend line at the final point
    trend_final = np.mean(slopes.flatten()) * model_df[model_time_name].values[
        -1
    ] + np.mean(intercepts.flatten())

    # Detrend the data by subtracting the trend line and adding the final value
    model_df[model_val_name + "_dt"] = (
        model_df[model_val_name] - trend_line + trend_final
    )

    # interpolate the trend line for the observations
    trend_line_obs = np.interp(
        obs_df[obs_time_name], model_df[model_time_name], trend_line
    )

    # # print obs_df[obs_time_name]
    # # print model_df[model_time_name]
    # print(f"The type of obs_df[obs_time_name] is {type(obs_df[obs_time_name].values[0])}")

    # # print the values
    # print(f"The values of obs_df[obs_time_name] are {obs_df[obs_time_name].values}")

    # # set up the trend final for the obs
    trend_final_obs = np.mean(slopes.flatten()) * obs_df[obs_time_name].dt.year.astype(
        int
    ).iloc[-1] + np.mean(intercepts.flatten())

    # Detrend the observations
    obs_df[obs_val_name + "_dt"] = obs_df[obs_val_name] - trend_line_obs + trend_final

    # print the trend line obs
    print(f"The trend line obs is {trend_line_obs}")

    # print the trend line model
    print(f"The trend line model is {trend_line}")

    # print the trend final and trend final obs
    print(f"The trend final is {trend_final}")
    print(f"The trend final obs is {trend_final_obs}")

    # print the model_val_name
    print(f"The model_val_name is {model_val_name}")

    # print the obs_val_name
    print(f"The obs_val_name is {obs_val_name}")

    # priny the mean of the model and obs data vlaues
    # print the mean of the model and obs data values
    print(f"The mean of the model data is {model_df[f'{model_val_name}_dt'].mean()}")

    print(f"The mean of the obs data is {obs_df[f'{obs_val_name}_dt'].mean()}")

    return obs_df, model_df


# write a function to peform the linear scaling bias correction
# Linear scaling in: https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance---how-to-bias-correct.pdf
def bc_linear_scaling(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
) -> pd.DataFrame:
    """
    Applies a linear scaling bias correction to the model data.

    X(t) = Ohat_base - Xhat_base + X_fut(t)

    Parameters
    ==========

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

    Returns
    =======

    pd.DataFrame
        A DataFrame containing the bias corrected model data.

    """

    # Quantify the bias
    bias = model_df[model_val_name].mean() - obs_df[obs_val_name].mean()

    # Apply the linear scaling bias correction
    model_df[model_val_name + "_bc"] = model_df[model_val_name] - bias

    return model_df


# write a function to perform the variance scaling bias correction
# Variance scaling in: https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance---how-to-bias-correct.pdf
def bc_variance_scaling(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
) -> pd.DataFrame:
    """
    Applies a mean-variance bias correction to the model data.

    X(t) = σX_fut/σX_base * (O_base(t) - X_base) + X_fut

    Parameters
    ==========

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

    Returns
    =======

    pd.DataFrame
        A DataFrame containing the bias corrected model data.

    """

    # Apply the mean-variance bias correction
    # Calculate the means
    obs_mean = np.mean(obs_df[obs_val_name])
    model_mean = np.mean(model_df[model_val_name])

    # Calculate the standard deviations
    obs_std = np.std(obs_df[obs_val_name])
    model_std = np.std(model_df[model_val_name])

    # print the obersed val name
    print(f"The observed value name is {obs_val_name}")
    # print the observed spread
    print(f"The observed spread is {obs_std}")

    # quantiy the observed spread
    print(f"The observed spread is {np.std(obs_df[obs_val_name])}")

    # print the model spread
    print(f"The model spread is {model_std}")

    # Adjust the mean of the model data
    model_df[model_val_name + "_mean_bc"] = model_df[model_val_name] + (
        obs_mean - model_mean
    )

    # Normalise the mean corrected model data to a zero mean
    model_df[model_val_name + "_mean_bc_norm"] = model_df[
        model_val_name + "_mean_bc"
    ] - np.mean(model_df[model_val_name + "_mean_bc"])

    # Scale the variance
    model_df[model_val_name + "_mean_var_bc"] = (
        np.std(obs_df[obs_val_name])
        / np.std(model_df[model_val_name + "_mean_bc_norm"])
    ) * model_df[model_val_name + "_mean_bc_norm"]

    # add the mean back to the variance scaled model data
    model_df[model_val_name + "_bc"] = model_df[
        model_val_name + "_mean_var_bc"
    ] + np.mean(model_df[model_val_name + "_mean_bc"])

    # remove the columns that are not needed
    model_df.drop(
        [
            model_val_name + "_mean_bc",
            model_val_name + "_mean_bc_norm",
            model_val_name + "_mean_var_bc",
        ],
        axis=1,
        inplace=True,
    )

    # print the observed spread
    print(
        f"The observed spread before leaving function is {np.std(obs_df[obs_val_name])}"
    )

    # pfing thd mocdl spread
    print(
        f"The model spread before leaving function is {np.std(model_df[model_val_name + '_bc'])}"
    )

    return model_df


# Define a function to perform bias correction using quantile mapping
def bc_quantile_mapping(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    save_prefix: str = "qmap",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> pd.DataFrame:
    """
    Applies a quantile mapping bias correction to the model data.

    X(t) = F_O^-1(F_X(X(t)))

    Parameters
    ==========

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

    save_prefix: str
        The prefix to use when saving the plots. Default is "qmap".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    pd.DataFrame
        A DataFrame containing the bias corrected model data.

    """

    # Set up a square figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # fit a normal distribution to the obs
    obs_mu, obs_std = norm.fit(obs_df[obs_val_name])

    # fit a normal distribution to the model
    model_mu, model_std = norm.fit(model_df[model_val_name])

    # generate evenly spaced values over the range of the data for plotting
    x = np.linspace(
        min(min(obs_df[obs_val_name]), min(model_df[model_val_name])),
        max(max(obs_df[obs_val_name]), max(model_df[model_val_name])),
        100,
    )

    # calculate the CDFs of the fitted normal distributions
    obs_cdf = norm.cdf(x, obs_mu, obs_std)
    model_cdf = norm.cdf(x, model_mu, model_std)

    # plot the model data
    plt.plot(x, model_cdf, label="HadGEM3-GC31-MM", color="red")

    # plot the obs data
    plt.plot(x, obs_cdf, label="ERA5", color="black")

    plt.legend()

    # set up the current time
    now = datetime.now()

    # set up the current date
    date = now.strftime("%Y-%m-%d")

    # set up the current time
    time = now.strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_cdf_{date}_{time}.pdf"))

    # Get an array of cdf values
    # based on the model data and the normal distribution params
    cdf_vals = norm.cdf(model_df[model_val_name], loc=model_mu, scale=model_std)

    # set up the cdf hreshold as default
    cdf_threshold = 1e-10

    # threshold the array of cdf-values
    # away from 0 and 1
    threshold_cdf_vals = np.maximum(
        np.minimum(cdf_vals, 1 - cdf_threshold), cdf_threshold
    )

    # compute the fitted quantiles using the thresholded cdf values
    # and the fitted obs normal distribution
    fitted_quantiles = norm.ppf(threshold_cdf_vals, loc=obs_mu, scale=obs_std)

    # print the values of fitted quantiles and the type
    print(f"The values of the fitted quantiles are {fitted_quantiles}")
    print(f"The type of the fitted quantiles is {type(fitted_quantiles)}")

    # print the values of the model data and the type
    print(f"The values of the model data are {model_df[model_val_name]}")

    # print the type of the model data
    print(f"The type of the model data is {type(model_df[model_val_name])}")

    # # quantify the differences between the fitted quantiles and the model data
    # diff = fitted_quantiles - model_df[model_val_name]

    # # print the cumulative sum of the dif
    # print(f"The cumulative sum of the diff is {np.cumsum(diff)}")

    # plot the fitted quantiles against the model data
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the model data against the fitted quantiles
    ax.scatter(model_df[model_val_name], fitted_quantiles, color="red")

    # plot the 1:1 line
    ax.plot(
        [
            min(min(model_df[model_val_name]), min(fitted_quantiles)),
            max(max(model_df[model_val_name]), max(fitted_quantiles)),
        ],
        [
            min(min(model_df[model_val_name]), min(fitted_quantiles)),
            max(max(model_df[model_val_name]), max(fitted_quantiles)),
        ],
        color="black",
        linestyle="--",
    )

    # set the x-axis label
    ax.set_xlabel("HadGEM3-GC31-MM")

    # set the y-axis label
    ax.set_ylabel("ERA5")

    # set the title
    ax.set_title("Quantile Mapping")

    # save the figure
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_qmap_{date}_{time}.pdf"))

    # fit a normal distribution to the bias corrected data
    bc_mu, bc_std = norm.fit(fitted_quantiles)

    # print the values of the fitted quantiles
    print(f"The values of the fitted quantiles are {fitted_quantiles}")

    # print the bc_mu and bc_std
    print(f"The bc_mu is {bc_mu}")
    print(f"The bc_std is {bc_std}")

    # set up a new x for the cdf
    x = np.linspace(
        min(min(fitted_quantiles), min(model_df[model_val_name])),
        max(max(fitted_quantiles), max(model_df[model_val_name])),
        100,
    )

    # calculate the cdf of the fitted normal distribution
    bc_cdf = norm.cdf(x, bc_mu, bc_std)

    # Set up another figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the cdf of the obs data as a solid black line
    plt.plot(x, obs_cdf, color="black", label="ERA5")

    # plot the cdf of the bias corrected data as a red dashed line
    plt.plot(x, bc_cdf, color="red", linestyle="--", label="HadGEM quantile mapping")

    # plot the original model data as a solid red line
    plt.plot(x, model_cdf, color="red", label="HadGEM3-GC31-MM")

    # set the x-axis label
    plt.xlabel("Value")

    # set the y-axis label
    plt.ylabel("Cumulative Probability")

    # set the title
    plt.title("CDFs of the model and bias corrected data")

    # set the legend
    plt.legend()

    # save the figure
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_cdf_bc_{date}_{time}.pdf"))

    # Apply the quantile mapping bias correction
    model_df[model_val_name + "_bc"] = fitted_quantiles

    return model_df


# Define a function to plot the chance of an event
# being worse (i.e. lower values)
# than a specific year (in this case 2010)
def plot_chance_of_event(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    variable: str,
    num_samples: int = 1000,
    obs_year: int = 2010,
    save_prefix: str = "chance_of_event",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the chance of an event being worse than a specific year.

    Parameters
    ==========

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

    obs_year: int
        The year to compare against. Default is 2010.

    save_prefix: str
        The prefix to use when saving the plots. Default is "chance_of_event".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    None

    """

    # Set up the params for the obs and the model
    params_obs = []
    params_model_obs_len = []
    params_model = []

    # find the years of the lowest value event in the obs data
    obs_time_event = obs_df[obs_df[obs_val_name] == obs_df[obs_val_name].min()][
        "time"
    ].values[0]

    # print the year and the value
    print(f"The time with the lowest value in the obs data is {obs_time_event}")

    # print the value at this time
    print(f"The value at this time is {obs_df[obs_val_name].min()}")

    # Convert numpy.datetime64 to datetime
    obs_time_event = obs_time_event.astype(datetime)

    # Convert timestamp to datetime
    obs_time_event = pd.to_datetime(obs_time_event, unit="ns")

    # print obs time lowesty
    print(f"The obs time lowest is {obs_time_event}")

    # Format datetime to "YYYY-MM"
    obs_time_event = obs_time_event.strftime("%Y-%m")

    print(f"The obs time lowest is {obs_time_event}")

    # apply thresholds to the data
    # calculte the 17.5%tile of the observations
    obs_175_threshold = np.percentile(obs_df[obs_val_name], 17.5)

    # print the 17.5% threshold
    print(f"The 17.5% threshold is {obs_175_threshold}")

    # Apply this threshold to the observations
    obs_175 = obs_df[obs_df[obs_val_name] < obs_175_threshold]

    # print the length of the obs_175
    print(f"The length of the obs_175 is {len(obs_175)}")

    # calculate the 82.5%tile of the observations
    model_175 = model_df[model_df[model_val_name] < obs_175_threshold]

    # Set up the years
    years = np.arange(1.1, 1000, 0.1)

    # Generate 1000 values by resamlping data with replacement
    for i in tqdm(range(num_samples)):
        params_obs.append(
            gev.fit(
                np.random.choice(
                    obs_175[obs_val_name], size=len(obs_175[obs_val_name]), replace=True
                )
            )
        )
        params_model_obs_len.append(
            gev.fit(
                np.random.choice(
                    model_175[model_val_name],
                    size=len(obs_175[obs_val_name]),  # resample to obs length
                    replace=True,
                )
            )
        )

        params_model.append(
            gev.fit(
                np.random.choice(
                    model_175[model_val_name],
                    size=len(model_175[model_val_name]),
                    replace=True,
                )
            )
        )

    # initialize the list for return levels
    levels_obs = []
    levels_model_obs_len = []
    levels_model = []

    # # print params obs
    # print(f"The params obs are {params_obs}")

    # # print params model
    # print(f"The params model are {params_model}")

    # negate each of the params individually
    # Negate each element in each tuple
    # params_obs = [tuple(-x for x in param) for param in params_obs]
    # params_model = [tuple(-x for x in param) for param in params_model]

    # print(f"The params obs are {params_obs}")

    # print(f"The params model are {params_model}")

    # Calculate the return levels for each of the 1000 samples
    for i in range(num_samples):
        levels_obs.append(gev.ppf(1 / years, *params_obs[i]))
        levels_model_obs_len.append(gev.ppf(1 / years, *params_model_obs_len[i]))
        levels_model.append(gev.ppf(1 / years, *params_model[i]))

    # turn this into arrays
    levels_obs = np.array(levels_obs)
    levels_model_obs_len = np.array(levels_model_obs_len)
    levels_model = np.array(levels_model)

    # set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # # Find empirical return levels
    # _ = empirical_return_level(obs_df[obs_val_name].values).plot(
    #     ax=ax, color="black", linestyle="None", marker="."
    # )
    # _ = empirical_return_level(model_df[model_val_name].values).plot(
    #     ax=ax, color="red", linestyle="None", marker="."
    # )

    # set up the probablities
    probs = 1 / years * 100

    # print the years
    print(f"The years are {years}")

    # print the probabilities
    print(f"The probabilities are {probs}")

    # worst obs event
    obs_worst_event = obs_df[obs_val_name].min()

    # Calculate the empirical return levels
    obs_df_rl = empirical_return_level(obs_df[obs_val_name].values)
    model_df_rl = empirical_return_level(model_df[model_val_name].values)

    # add a new column for the anomalies from the worst observed event
    obs_df_rl["anomalies"] = obs_df_rl["sorted"] - obs_worst_event
    model_df_rl["anomalies"] = model_df_rl["sorted"] - obs_worst_event

    # Plot the anomalies and the probabilities
    _ = plt.plot(
        obs_df_rl["anomalies"],
        obs_df_rl["probability"],
        color="black",
        linestyle="None",
        marker=".",
    )
    _ = plt.plot(
        model_df_rl["anomalies"],
        model_df_rl["probability"],
        color="red",
        linestyle="None",
        marker=".",
    )

    # _ = plt.plot(probs, obs_anomalies, color="black", linestyle="None", marker=".")
    # _ = plt.plot(probs, model_anomalies, color="red", linestyle="None", marker=".")

    # subtract the worst event from the obs data
    levels_obs_anomaly = levels_obs - obs_worst_event
    levels_model_obs_len_anomaly = levels_model_obs_len - obs_worst_event
    levels_model_anomaly = levels_model - obs_worst_event

    # Plot the return mean levels
    # _ = plt.plot(np.mean(levels_obs_anomaly, axis=0), probs, "k-", label="ERA5")
    # _ = plt.plot(
    #     np.mean(levels_model_anomaly, axis=0), probs, "r-", label="HadGEM3-GC31-MM"
    # )

    # Plot the confidence intervals
    _ = ax.plot(np.quantile(levels_obs_anomaly, [0.025, 0.975], axis=0).T, probs, "k--")
    _ = ax.plot(
        np.quantile(levels_model_obs_len_anomaly, [0.025, 0.975], axis=0).T,
        probs,
        "r--",
    )
    _ = ax.plot(np.quantile(levels_model_anomaly, [0.025, 0.975], axis=0).T, probs, "r")

    _ = ax.fill_betweenx(
        probs,
        np.quantile(levels_model_anomaly, 0.025, axis=0),
        np.quantile(levels_model_anomaly, 0.975, axis=0),
        color="red",
        alpha=0.2,
    )

    # # aesthetics
    ax.set_ylim(0.1, 20)  # Adjust as needed
    ax.set_xlim(2, -1)  # Adjust as needed

    # set the xpoints
    x_points = np.array([30, 15, 10, 5, 2, 1, 0.5, 0.2, 0.1])

    # y_labels = [f"{x:.1f}%" for x in x_points]

    # depending on the variable set up the unnits
    if variable == "sfcWind":
        # set the units
        units = "m/s"
    elif variable == "tas":
        # set the units
        units = "°C"
    else:
        print(f"variable {variable} not recognized")

    # inclduie a dotted line for the 1% threshold (i.e. 1 in 100 year event)
    ax.axhline(1, color="black", linestyle="dotted")  # 1 in 100 year event

    # # plot a vertical line at 0
    # ax.axvline(0, color="black", linestyle="dotted")

    # create a list of levels to quantify the values for
    levels = [5, 10, 20, 100, 1000]

    # loop over the levels
    for level in levels:
        return_level_obs = estimate_return_level_period(
            period=level,
            loc=np.mean(params_obs, axis=0)[1],
            scale=np.mean(params_obs, axis=0)[2],
            shape=np.mean(params_obs, axis=0)[0],
        )

        return_level_model = estimate_return_level_period(
            period=level,
            loc=np.mean(params_model, axis=0)[1],
            scale=np.mean(params_model, axis=0)[2],
            shape=np.mean(params_model, axis=0)[0],
        )

        # print the level and the return level
        print(
            f"The level is {level} and the return value for the obs fit is {return_level_obs} m/s"
        )

        # print the level and the return level
        print(
            f"The level is {level} and the return value for the model fit is {return_level_model} m/s"
        )

    # create a lits of values to estimate for
    values = [4.5, 4.0, 3.75, 3.5, 3.25]

    # loop over the values
    for value in values:
        period_obs = estimate_period(
            return_level=value,
            loc=np.mean(params_obs, axis=0)[1],
            scale=np.mean(params_obs, axis=0)[2],
            shape=np.mean(params_obs, axis=0)[0],
        )

        period_model = estimate_period(
            return_level=value,
            loc=np.mean(params_model, axis=0)[1],
            scale=np.mean(params_model, axis=0)[2],
            shape=np.mean(params_model, axis=0)[0],
        )

        # print the values for the obs
        print(
            f"The value is {value} and the period for the obs fit is {period_obs} years"
        )

        # print the values for the model
        print(
            f"The value is {value} and the period for the model fit is {period_model} years"
        )

    # assign a variable to the obs lowest value
    lowest_obs = obs_df[obs_val_name].min()

    # Estimate the period for the obs mean data
    period_obs_mean = estimate_period(
        return_level=lowest_obs,
        loc=np.mean(params_obs, axis=0)[1],
        scale=np.mean(params_obs, axis=0)[2],
        shape=np.mean(params_obs, axis=0)[0],
    )

    # period model mean
    period_model_mean = estimate_period(
        return_level=lowest_obs,
        loc=np.mean(params_model, axis=0)[1],
        scale=np.mean(params_model, axis=0)[2],
        shape=np.mean(params_model, axis=0)[0],
    )

    # obs 05 percentile
    period_obs_05 = estimate_period(
        return_level=lowest_obs,
        loc=np.percentile(params_obs, 5, axis=0)[1],
        scale=np.percentile(params_obs, 5, axis=0)[2],
        shape=np.percentile(params_obs, 5, axis=0)[0],
    )

    # obs 95 percentile
    period_obs_95 = estimate_period(
        return_level=lowest_obs,
        loc=np.percentile(params_obs, 95, axis=0)[1],
        scale=np.percentile(params_obs, 95, axis=0)[2],
        shape=np.percentile(params_obs, 95, axis=0)[0],
    )

    # model 05 percentile
    period_model_05 = estimate_period(
        return_level=lowest_obs,
        loc=np.percentile(params_model, 5, axis=0)[1],
        scale=np.percentile(params_model, 5, axis=0)[2],
        shape=np.percentile(params_model, 5, axis=0)[0],
    )

    # model 95 percentile
    period_model_95 = estimate_period(
        return_level=lowest_obs,
        loc=np.percentile(params_model, 95, axis=0)[1],
        scale=np.percentile(params_model, 95, axis=0)[2],
        shape=np.percentile(params_model, 95, axis=0)[0],
    )

    # quantify the obs 90th percentile
    period_obs_90 = abs(period_obs_95 - period_obs_05)
    period_model_90 = abs(period_model_95 - period_model_05)

    # include a textbox with this information in the top right
    ax.text(
        0.95,
        0.95,
        f"ERA5: {period_obs_mean:.2f}% (+- {period_obs_90:.2f}%), 1:{round(1/period_obs_mean * 100)}-yr  \nHadGEM3-GC31-MM: {period_model_mean:.2f}% (+- {period_model_90:.2f}%), 1:{round(1/period_model_mean * 100)}-yr",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    ax.set_yscale("log")
    # set the yticks
    ax.set_yticks(x_points)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Chance of event (%)")
    ax.set_xlabel(f"Lower than {obs_time_event} ({units})")

    # Set the title
    ax.set_title(f"Chance of {variable} event being worse than {obs_time_event}")

    # show the legend
    ax.legend(loc="lower left")

    # set the title
    time_now = datetime.now()

    # set the date
    date = time_now.strftime("%Y-%m-%d-%H-%M-%S")

    # save the figure
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
        bbox_inches="tight",
        dpi=600,
    )

    return


# Define a function to plot the chance of an event
# being worse (i.e. lower values)
# than a specific year (in this case 2010)
def plot_chance_of_event_rank(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    variable: str,
    num_samples: int = 1000,
    obs_year: int = 2010,
    save_prefix: str = "chance_of_event_rank",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the chance of an event being worse than a specific year. Using rank estimation.

    Parameters
    ==========

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

    obs_year: int
        The year to compare against. Default is 2010.

    save_prefix: str
        The prefix to use when saving the plots. Default is "chance_of_event".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    None

    """

    # Set up the params for the obs and the model
    params_obs = []
    params_model_obs_len = []
    params_model = []

    # find the years of the lowest value event in the obs data
    obs_time_event = obs_df[obs_df[obs_val_name] == obs_df[obs_val_name].min()][
        "time"
    ].values[0]

    # print the year and the value
    print(f"The time with the lowest value in the obs data is {obs_time_event}")

    # print the value at this time
    print(f"The value at this time is {obs_df[obs_val_name].min()}")

    # Convert numpy.datetime64 to datetime
    obs_time_event = obs_time_event.astype(datetime)

    # Convert timestamp to datetime
    obs_time_event = pd.to_datetime(obs_time_event, unit="ns")

    # print obs time lowesty
    print(f"The obs time lowest is {obs_time_event}")

    # Format datetime to "YYYY-MM"
    obs_time_event = obs_time_event.strftime("%Y-%m")

    print(f"The obs time lowest is {obs_time_event}")

    # Set up the years
    years = np.arange(1.1, 1000, 0.1)

    # # Generate 1000 values by resamlping data with replacement
    # for i in tqdm(range(num_samples)):
    #     params_obs.append(
    #         gev.fit(
    #             np.random.choice(
    #                 obs_df[obs_val_name], size=len(obs_df[obs_val_name]), replace=True
    #             )
    #         )
    #     )
    #     params_model_obs_len.append(
    #         gev.fit(
    #             np.random.choice(
    #                 model_df[model_val_name],
    #                 size=len(obs_df[obs_val_name]),  # resample to obs length
    #                 replace=True,
    #             )
    #         )
    #     )

    #     params_model.append(
    #         gev.fit(
    #             np.random.choice(
    #                 model_df[model_val_name], size=len(model_df[model_val_name]), replace=True
    #             )
    #         )
    #     )

    # initialize the list for return levels
    levels_obs = []
    levels_model_obs_len = []
    levels_model = []

    # # print params obs
    # print(f"The params obs are {params_obs}")

    # # print params model
    # print(f"The params model are {params_model}")

    # negate each of the params individually
    # Negate each element in each tuple
    # params_obs = [tuple(-x for x in param) for param in params_obs]
    # params_model = [tuple(-x for x in param) for param in params_model]

    # print(f"The params obs are {params_obs}")

    # print(f"The params model are {params_model}")

    # # Calculate the return levels for each of the 1000 samples
    # for i in range(num_samples):
    #     levels_obs.append(gev.ppf(1 / years, *params_obs[i]))
    #     levels_model_obs_len.append(gev.ppf(1 / years, *params_model_obs_len[i]))
    #     levels_model.append(gev.ppf(1 / years, *params_model[i]))

    # # turn this into arrays
    # levels_obs = np.array(levels_obs)
    # levels_model_obs_len = np.array(levels_model_obs_len)
    # levels_model = np.array(levels_model)

    # set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # # Find empirical return levels
    # _ = empirical_return_level(obs_df[obs_val_name].values).plot(
    #     ax=ax, color="black", linestyle="None", marker="."
    # )
    # _ = empirical_return_level(model_df[model_val_name].values).plot(
    #     ax=ax, color="red", linestyle="None", marker="."
    # )

    # set up the probablities
    probs = 1 / years * 100

    # print the years
    print(f"The years are {years}")

    # print the probabilities
    print(f"The probabilities are {probs}")

    # worst obs event
    obs_worst_event = obs_df[obs_val_name].min()

    # Calculate the empirical return levels
    obs_df_rl = empirical_return_level(obs_df[obs_val_name].values)
    model_df_rl = empirical_return_level(model_df[model_val_name].values)

    # add a new column for the anomalies from the worst observed event
    # obs_df_rl["anomalies"] = obs_df_rl["sorted"] - obs_worst_event
    # model_df_rl["anomalies"] = model_df_rl["sorted"] - obs_worst_event

    obs_anoms = obs_df_rl["sorted"] - obs_worst_event
    model_anoms = model_df_rl["sorted"] - obs_worst_event

    # # print the sanomalies
    # print(f"The anomalies are {obs_df_rl['anomalies']}")

    # # print the shape of the model danaomlies
    # print(f"The shape of the model anomalies is {model_df_rl['anomalies'].shape}")

    # sys.exit()

    # # # Plot the anomalies and the probabilities
    # _ = plt.plot(
    #     obs_df_rl["anomalies"],
    #     obs_df_rl["probability"],
    #     color="black",
    # )
    _ = plt.plot(
        model_anoms,
        model_df_rl["probability"],
        color="red",
    )

    # create an empty array
    model_anomalies_full = np.zeros([num_samples, len(model_anoms)])
    model_anomalies_len_obs = np.zeros([num_samples, len(obs_anoms)])

    # Create empty arrays for the obs data
    obs_anomalies = np.zeros([num_samples, len(obs_anoms)])
    obs_probs = np.zeros([num_samples, len(obs_df_rl["probability"])])

    model_probs_full = np.zeros([num_samples, len(model_df_rl["probability"])])
    model_probs_len_obs = np.zeros([num_samples, len(obs_df_rl["probability"])])

    # loop over the num samples
    for i in tqdm(range(num_samples)):
        # randomly select the model data
        model_data_this = np.random.choice(
            model_df[model_val_name], size=len(model_df[model_val_name]), replace=True
        )

        # same for the obs len
        # model_data_this_obs_len = np.random.choice(
        #     model_df[model_val_name], size=len(obs_df[obs_val_name]), replace=True
        # )

        # same for the obs data
        obs_data_this = np.random.choice(
            obs_df[obs_val_name], size=len(obs_df[obs_val_name]), replace=True
        )

        # calculate the empirical return levels
        model_df_rl_this = empirical_return_level(model_data_this)
        # model_df_rl_this_obs_len = empirical_return_level(model_data_this_obs_len)
        obs_df_rl_this = empirical_return_level(obs_data_this)

        # add the anomalies to the array
        model_anomalies_full[i, :] = model_df_rl_this["sorted"] - obs_worst_event
        # model_anomalies_len_obs[i, :] = model_df_rl_this_obs_len["sorted"] - obs_worst_event

        # add the probabilities to the array
        model_probs_full[i, :] = model_df_rl_this["probability"]
        # model_probs_len_obs[i, :] = model_df_rl_this_obs_len["probability"]

        # add the obs anomalies to the array
        obs_anomalies[i, :] = obs_df_rl_this["sorted"] - obs_worst_event

    # calculate the 2.5 and 97.5 percentiles
    model_anomalies_full_025 = np.percentile(model_anomalies_full, 2.5, axis=0)
    model_anomalies_full_975 = np.percentile(model_anomalies_full, 97.5, axis=0)

    # # same but for those obs
    obs_anoms_025 = np.percentile(obs_anomalies, 2.5, axis=0)
    obs_anoms_975 = np.percentile(obs_anomalies, 97.5, axis=0)

    # # same but for those at obs len
    # model_anomalies_len_obs_025 = np.percentile(model_anomalies_len_obs, 2.5, axis=0)
    # model_anomalies_len_obs_975 = np.percentile(model_anomalies_len_obs, 97.5, axis=0)

    # print the shape of the model anomalies
    print(f"The shape of the model anomalies is {model_anomalies_full_025.shape}")
    print(f"The shape of the model anomalies is {model_anomalies_full_975.shape}")

    # print the shape of the obs anomalies
    print(f"The shape of the obs anomalies is {obs_anoms_025.shape}")
    print(f"The shape of the obs anomalies is {obs_anoms_975.shape}")

    # plot the model data
    plt.fill_betweenx(
        model_probs_full[0, :],
        model_anomalies_full_025,
        model_anomalies_full_975,
        color="red",
        alpha=0.2,
    )

    # # plot the obs data
    # plt.fill_betweenx(
    #     obs_df_rl["probability"],
    #     obs_anoms_025,
    #     obs_anoms_975,
    #     color="black",
    #     alpha=0.2,
    # )

    # # same for the subsamples
    # plt.fill_betweenx(
    #     model_probs_len_obs[0, :],
    #     model_anomalies_len_obs_025,
    #     model_anomalies_len_obs_975,
    #     color="blue",
    #     alpha=0.2,
    # )

    # _ = plt.plot(probs, obs_anomalies, color="black", linestyle="None", marker=".")
    # _ = plt.plot(probs, model_anomalies, color="red", linestyle="None", marker=".")

    # # subtract the worst event from the obs data
    # levels_obs_anomaly = levels_obs - obs_worst_event
    # levels_model_obs_len_anomaly = levels_model_obs_len - obs_worst_event
    # levels_model_anomaly = levels_model - obs_worst_event

    # # Plot the return mean levels
    # # _ = plt.plot(np.mean(levels_obs_anomaly, axis=0), probs, "k-", label="ERA5")
    # # _ = plt.plot(
    # #     np.mean(levels_model_anomaly, axis=0), probs, "r-", label="HadGEM3-GC31-MM"
    # # )

    # # Plot the confidence intervals
    # _ = ax.plot(np.quantile(levels_obs_anomaly, [0.025, 0.975], axis=0).T, probs, "k--")
    # _ = ax.plot(np.quantile(levels_model_obs_len_anomaly, [0.025, 0.975], axis=0).T, probs, "r--")
    # _ = ax.plot(
    #     np.quantile(levels_model_anomaly, [0.025, 0.975], axis=0).T, probs, "r"
    # )

    # _ = ax.fill_betweenx(
    #     probs,
    #     np.quantile(levels_model_anomaly, 0.025, axis=0),
    #     np.quantile(levels_model_anomaly, 0.975, axis=0),
    #     color="red",
    #     alpha=0.2,
    # )

    # # aesthetics
    ax.set_ylim(0.1, 20)  # Adjust as needed

    # set the xpoints
    x_points = np.array([20, 10, 5, 2, 1, 0.5, 0.2, 0.1])

    # y_labels = [f"{x:.1f}%" for x in x_points]

    # depending on the variable set up the unnits
    if variable == "sfcWind":
        # set the units
        units = "m/s"

        # Set the xlim
        ax.set_xlim(1, -1)  # Adjust as needed
    elif variable == "tas":
        # set the units
        units = "°C"
        ax.set_xlim(2, -1)  # Adjust as needed
    else:
        print(f"variable {variable} not recognized")

    # inclduie a dotted line for the 1% threshold (i.e. 1 in 100 year event)
    ax.axhline(1, color="black", linestyle="dotted")  # 1 in 100 year event

    # # plot a vertical line at 0
    # ax.axvline(0, color="black", linestyle="dotted")

    # create a list of levels to quantify the values for
    levels = [5, 10, 20, 100, 1000]

    # # loop over the levels
    # for level in levels:
    #     return_level_obs = estimate_return_level_period(
    #         period=level,
    #         loc=np.mean(params_obs, axis=0)[1],
    #         scale=np.mean(params_obs, axis=0)[2],
    #         shape=np.mean(params_obs, axis=0)[0],
    #     )

    #     return_level_model = estimate_return_level_period(
    #         period=level,
    #         loc=np.mean(params_model, axis=0)[1],
    #         scale=np.mean(params_model, axis=0)[2],
    #         shape=np.mean(params_model, axis=0)[0],
    #     )

    #     # print the level and the return level
    #     print(
    #         f"The level is {level} and the return value for the obs fit is {return_level_obs} m/s"
    #     )

    #     # print the level and the return level
    #     print(
    #         f"The level is {level} and the return value for the model fit is {return_level_model} m/s"
    #     )

    # # create a lits of values to estimate for
    # values = [4.5, 4.0, 3.75, 3.5, 3.25]

    # # loop over the values
    # for value in values:
    #     period_obs = estimate_period(
    #         return_level=value,
    #         loc=np.mean(params_obs, axis=0)[1],
    #         scale=np.mean(params_obs, axis=0)[2],
    #         shape=np.mean(params_obs, axis=0)[0],
    #     )

    #     period_model = estimate_period(
    #         return_level=value,
    #         loc=np.mean(params_model, axis=0)[1],
    #         scale=np.mean(params_model, axis=0)[2],
    #         shape=np.mean(params_model, axis=0)[0],
    #     )

    #     # print the values for the obs
    #     print(
    #         f"The value is {value} and the period for the obs fit is {period_obs} years"
    #     )

    #     # print the values for the model
    #     print(
    #         f"The value is {value} and the period for the model fit is {period_model} years"
    #     )

    # # assign a variable to the obs lowest value
    # lowest_obs = obs_df[obs_val_name].min()

    # # Estimate the period for the obs mean data
    # period_obs_mean = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.mean(params_obs, axis=0)[1],
    #     scale=np.mean(params_obs, axis=0)[2],
    #     shape=np.mean(params_obs, axis=0)[0],
    # )

    # # period model mean
    # period_model_mean = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.mean(params_model, axis=0)[1],
    #     scale=np.mean(params_model, axis=0)[2],
    #     shape=np.mean(params_model, axis=0)[0],
    # )

    # # obs 05 percentile
    # period_obs_05 = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.percentile(params_obs, 5, axis=0)[1],
    #     scale=np.percentile(params_obs, 5, axis=0)[2],
    #     shape=np.percentile(params_obs, 5, axis=0)[0],
    # )

    # # obs 95 percentile
    # period_obs_95 = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.percentile(params_obs, 95, axis=0)[1],
    #     scale=np.percentile(params_obs, 95, axis=0)[2],
    #     shape=np.percentile(params_obs, 95, axis=0)[0],
    # )

    # # model 05 percentile
    # period_model_05 = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.percentile(params_model, 5, axis=0)[1],
    #     scale=np.percentile(params_model, 5, axis=0)[2],
    #     shape=np.percentile(params_model, 5, axis=0)[0],
    # )

    # # model 95 percentile
    # period_model_95 = estimate_period(
    #     return_level=lowest_obs,
    #     loc=np.percentile(params_model, 95, axis=0)[1],
    #     scale=np.percentile(params_model, 95, axis=0)[2],
    #     shape=np.percentile(params_model, 95, axis=0)[0],
    # )

    # # quantify the obs 90th percentile
    # period_obs_90 = abs(period_obs_95 - period_obs_05)
    # period_model_90 = abs(period_model_95 - period_model_05)

    # print the model anoms
    print(f"The model anomalies are {model_anoms}")

    # print the min value of model anoms
    print(f"The min value of model anoms is {np.min(model_anoms)}")

    # Convert model_anoms to a NumPy array
    model_anoms_array = np.array(model_anoms)

    # same for the 025
    model_anoms_025_array = np.array(model_anomalies_full_025)
    model_anoms_975_array = np.array(model_anomalies_full_975)

    # Find the index of the value closest to zero
    model_anoms_zero_idx = np.argmin(np.abs(model_anoms_array))

    # same with the other ones
    model_anoms_025_zero_idx = np.argmin(np.abs(model_anoms_025_array))
    model_anoms_975_zero_idx = np.argmin(np.abs(model_anoms_975_array))

    # Apply this index to the model probabilities
    model_probs_zero = model_df_rl["probability"].iloc[model_anoms_zero_idx]

    # Print the model anomalies value at the zero index
    model_probs_025_zero = model_df_rl["probability"].iloc[model_anoms_025_zero_idx]
    model_probs_975_zero = model_df_rl["probability"].iloc[model_anoms_975_zero_idx]

    # Print the model anomalies value at the zero index
    print(
        f"The model anomalies value at the zero index is {model_anoms_array[model_anoms_zero_idx]}"
    )

    # the model anomalies value at the 025 index
    print(
        f"The model anomalies value at the 025 index is {model_anoms_025_array[model_anoms_025_zero_idx]}"
    )
    print(
        f"The model anomalies value at the 975 index is {model_anoms_975_array[model_anoms_975_zero_idx]}"
    )

    # Print the model probabilities at the zero index
    print(f"The model probabilities at the zero index is {model_probs_zero}")

    # Print the model probabilities at the zero index
    print(f"The model probabilities at the 025 index is {model_probs_025_zero}")
    print(f"The model probabilities at the 975 index is {model_probs_975_zero}")

    # calculate the period model 95
    period_model_95 = abs(model_probs_025_zero - model_probs_975_zero) / 2

    # print the period model 95
    print(f"The period model 95 is {period_model_95}")
    print(f"The model_probs_zero is {model_probs_zero}")

    # Calculate the return period for the 025 and 975 percentiles
    return_period_975 = 1 / model_probs_025_zero * 100
    return_period_025 = 1 / model_probs_975_zero * 100

    # print the return period
    print(f"The return period for the 975 percentile is {return_period_975}")
    print(f"The return period for the 025 percentile is {return_period_025}")

    # calculate the return period range
    return_period_range = abs(return_period_025 - return_period_975) / 2

    # include a textbox with this information in the top right
    ax.text(
        0.95,
        0.95,
        f"HadGEM3-GC31-MM: 1:{round(1/model_probs_zero * 100)}-yr (+- {round(return_period_range)}-yr)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    ax.set_yscale("log")
    # set the yticks
    ax.set_yticks(x_points)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Chance of event (%)")
    ax.set_xlabel(f"Lower than {obs_time_event} ({units})")

    # Set the title
    ax.set_title(f"Chance of {variable} event being worse than {obs_time_event}")

    # show the legend
    ax.legend(loc="lower left")

    # set the title
    time_now = datetime.now()

    # set the date
    date = time_now.strftime("%Y-%m-%d-%H-%M-%S")

    # save the figure
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
        bbox_inches="tight",
        dpi=600,
    )

    return


# Define a function to plot the chance of an event
# with time
def plot_chance_of_event_with_time(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    variable: str,
    lowest: bool = True,
    num_samples: int = 1000,
    fname_prefix: str = "chance_of_event_with_time",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """

    Plots the chance of an event being worse than a specific year with time.

    Parameters
    ==========

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
        The name of the observation time column.

    variable: str
        The name of the variable being analysed.

    num_samples: int
        The number of samples to use when estimating the return levels. Default is 1000.

    fname_prefix: str
        The prefix to use when saving the plots. Default is "chance_of_event_with_time".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    None
    """

    # Set up the params for the obs and the model
    params_obs = []
    params_model = []

    # extrcat the unique years from th3e model df
    unique_years = np.unique(model_df["init_year"])

    if lowest:
        # Find the time with the lowest value in the obs data
        obs_time_event = obs_df[obs_df[obs_val_name] == obs_df[obs_val_name].min()][
            obs_time_name
        ].values[0]
    else:
        # Find the time with the highest value in the obs data
        obs_time_event = obs_df[obs_df[obs_val_name] == obs_df[obs_val_name].max()][
            obs_time_name
        ].values[0]

    # Print the time and the value
    print(f"The time with the lowest value in the obs data is {obs_time_event}")
    print(f"The value at this time is {obs_df[obs_val_name].min()}")

    # Convert numpy.datetime64 to datetime
    obs_time_event = obs_time_event.astype(datetime)

    # convert back to datetime
    obs_time_event = pd.to_datetime(obs_time_event, unit="ns")

    # Format datetime to "YYYY-MM"
    obs_time_event = obs_time_event.strftime("%Y-%m")

    # print the obs time lowest
    print(f"The obs time lowest is {obs_time_event}")

    # Set up the years
    years = np.arange(1.1, 1000, 0.1)

    # set up the probabilities
    probs = 1 / years * 100

    # Initialize the lists for th return levels
    level_model_year = {}

    # set up params year
    params_model_year = {}

    # Loop over unique years to get the model params
    for year in tqdm(unique_years, desc="Fitting GEV"):
        # Subset the model data to this year
        model_year = model_df[model_df["init_year"] == year]

        # initialize the list for the model data
        params_model = []

        # Generate 1000 values by resampling data with replacement
        for i in range(num_samples):
            params_model.append(
                gev.fit(
                    np.random.choice(
                        model_year[model_val_name],
                        size=len(model_year[model_val_name]),
                        replace=True,
                    )
                )
            )

        # append the params to the model year
        params_model_year[year] = params_model

    if lowest:
        # Set up the worst obs event
        obs_worst_event = obs_df[obs_val_name].min()
    else:
        # Set up the worst obs event
        obs_worst_event = obs_df[obs_val_name].max()

    period_model_year = {}

    # set up empty arrays for the return levels
    period_model_mean = np.zeros([len(unique_years)])
    period_model_025 = np.zeros([len(unique_years)])
    period_model_975 = np.zeros([len(unique_years)])

    # loop over the unique years to fit the ppfs
    for i, year in tqdm(enumerate(unique_years), desc="estimating return periods"):
        # select the year params
        params_model = params_model_year[year]

        # Estimate the period for the model mean data
        period_model_mean[i] = estimate_period(
            return_level=obs_worst_event,
            loc=np.mean(params_model, axis=0)[1],
            scale=np.mean(params_model, axis=0)[2],
            shape=np.mean(params_model, axis=0)[0],
        )

        # model 025 percentile
        period_model_025[i] = estimate_period(
            return_level=obs_worst_event,
            loc=np.percentile(params_model, 2.5, axis=0)[1],
            scale=np.percentile(params_model, 2.5, axis=0)[2],
            shape=np.percentile(params_model, 2.5, axis=0)[0],
        )

        # model 95 percentile
        period_model_975[i] = estimate_period(
            return_level=obs_worst_event,
            loc=np.percentile(params_model, 97.5, axis=0)[1],
            scale=np.percentile(params_model, 97.5, axis=0)[2],
            shape=np.percentile(params_model, 97.5, axis=0)[0],
        )

    # mean_return_level = np.zeros([len(unique_years)])
    # return_level_025 = np.zeros([len(unique_years)])
    # return_level_975 = np.zeros([len(unique_years)])

    # # loop over the unique years
    # for i, year in enumerate(unique_years):
    #     # Select the levels model year
    #     levels_model = level_model_year[year]

    #     # add the mean return level
    #     mean_return_level[i] = np.mean(levels_model, axis=0)[0]

    #     # add the 025 return level
    #     return_level_025[i] = np.quantile(levels_model, 0.025, axis=0).T[0]

    #     # add the 975 return level
    #     return_level_975[i] = np.quantile(levels_model, 0.975, axis=0).T[0]

    # put these into a dataframe with the years
    model_df_rl = pd.DataFrame(
        {
            "year": unique_years,
            "mean": period_model_mean,
            "025": period_model_025,
            "975": period_model_975,
        }
    )

    # for the mean, 025, and 975 columns
    # turn into year

    # print the head of the model df rl
    print(model_df_rl.head())

    # take a centred 8-year running mean for mean 025 and 975
    model_df_rl["mean_8yr"] = model_df_rl["mean"].rolling(window=8, center=True).mean()
    model_df_rl["025_8yr"] = model_df_rl["025"].rolling(window=8, center=True).mean()
    model_df_rl["975_8yr"] = model_df_rl["975"].rolling(window=8, center=True).mean()

    # remove Nans from the dataframe
    model_df_rl = model_df_rl.dropna()

    # set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the mean return level
    ax.plot(
        model_df_rl["year"],
        model_df_rl["mean_8yr"],
        color="red",
        label="HadGEM3-GC31-MM",
    )

    # plot the 025 return level
    ax.plot(
        model_df_rl["year"],
        model_df_rl["025_8yr"],
        color="red",
        linestyle="--",
    )
    ax.plot(
        model_df_rl["year"],
        model_df_rl["975_8yr"],
        color="red",
        linestyle="--",
    )

    # shadde between the 025 and 975 return levels
    ax.fill_between(
        model_df_rl["year"],
        model_df_rl["025_8yr"],
        model_df_rl["975_8yr"],
        color="red",
        alpha=0.2,
    )

    # # set the x-axis label
    # ax.set_xlabel("Year")

    # set the y-axis label
    ax.set_ylabel(f"Chance of event (%)")

    if lowest:
        # Set the title
        ax.set_title(f"Chance of {variable} event <{obs_time_event}")
    else:
        # Set the title
        ax.set_title(f"Chance of {variable} event >{obs_time_event}")

    # set the current date_time
    time_now = datetime.now()
    date = time_now.strftime("%Y-%m-%d-%H-%M-%S")

    # save the figure
    plt.savefig(
        os.path.join(save_dir, f"{fname_prefix}_{date}.pdf"),
        bbox_inches="tight",
        dpi=600,
    )

    return


# define a function for calculating the empirical return period
def empirical_return_level(
    data: np.ndarray,
    high_values_rare: bool = False,
):
    """
    Function to calculate the empirical return level for a given dataset.

    Args:
        data (np.ndarray): Array containing the data of interest.

    Returns:
        np.ndarray: Array containing the empirical return levels.

    """

    # assert that the data is a numpy array
    assert isinstance(data, np.ndarray), "Data must be a numpy array"

    # create a dataframe from the data
    df = pd.DataFrame(index=np.arange(data.size))

    if not high_values_rare:
        # Sort the data
        df["sorted"] = np.sort(data)
    else:
        # Sort the data in descending order
        df["sorted"] = np.sort(data)[::-1]

    # rank via scipy
    df["rank_sp"] = np.sort(stats.rankdata(data))

    # find the exceedance probability
    n = data.size
    df["exceedance"] = df["rank_sp"] / (n + 1)

    # find the return period
    df["period"] = 1 / df["exceedance"]

    # calculate the probability in %
    df["probability"] = 1 / df["period"] * 100

    # reverse the order of rows
    df = df[::-1]

    # # transform into xarray dataarray
    # out = xr.DataArray(
    #     dims=["probability"],
    #     coords={"probability": df["probability"]},
    #     data=df["sorted"],
    #     name="level",
    # )

    return df


def estimate_return_level_period(period, loc, scale, shape):
    """
    Compute GEV-based return level for a given return period, and GEV parameters
    """
    return gev.ppf(1 / period, shape, loc=loc, scale=scale)


def estimate_period(return_level, loc, scale, shape):
    # Use the cumulative distribution function (CDF) of the GEV distribution
    # to estimate the cumulative probability for a given return level
    prob = gev.cdf(return_level, c=shape, loc=loc, scale=scale)

    period = 1 / prob

    probs = 1 / period * 100

    return probs


# plot the cdfs for the two datasets
def plot_cdfs(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    save_prefix: str = "cdfs",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the CDFs of the observations and model data.

    Parameters
    ==========

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

    save_prefix: str
        The prefix to use when saving the plots. Default is "cdfs".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    None

    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # sort the obs data
    obs_x = np.sort(obs_df[obs_val_name])
    obs_y = np.arange(1, len(obs_x) + 1) / float(len(obs_x))

    # sort the model data
    model_x = np.sort(model_df[model_val_name])
    model_y = np.arange(1, len(model_x) + 1) / float(len(model_x))

    # plot the observed cdf
    ax.plot(obs_x, obs_y, color="black", label="ERA5")

    # plot the model cdf
    ax.plot(model_x, model_y, color="red", label="HadGEM3-GC31-MM")

    # Set the x-axis label
    plt.xlabel("Sum")

    # Set the y-axis label
    plt.ylabel("CDF")

    # Set the title
    # plt.title("CDFs of the model and bias corrected data")

    # quantify the two sample ks test
    ks_stat, ks_p = ks_2samp(obs_df[obs_val_name], model_df[model_val_name])

    # print the ks stat and the ks p
    print(f"The ks stat is {ks_stat}")
    print(f"The ks p is {ks_p}")

    upper_alpha = 0.05
    lower_alpha = 0.01
    c_upper_alpha = 1.36
    c_lower_alpha = 1.63

    # quantify the critical value D_alpha
    d_alpha_upper = c_upper_alpha * np.sqrt(
        (len(obs_df[obs_val_name]) + len(model_df[model_val_name]))
        / (len(obs_df[obs_val_name]) * len(model_df[model_val_name]))
    )
    d_alpha_lower = c_lower_alpha * np.sqrt(
        (len(obs_df[obs_val_name]) + len(model_df[model_val_name]))
        / (len(obs_df[obs_val_name]) * len(model_df[model_val_name]))
    )

    # print the critical values
    print(f"at alpha = {upper_alpha}, the critical value is {d_alpha_upper}")
    print(f"at alpha = {lower_alpha}, the critical value is {d_alpha_lower}")

    # if the statistic value is higher than the critical value
    if ks_stat > d_alpha_lower:
        print("The null hypothesis is rejected at the 1% level")
        print("The two samples are not drawn from the same distribution")
    elif ks_stat > d_alpha_upper:
        print("The null hypothesis is rejected at the 5% level")
        print("The two samples are not drawn from the same distribution")
    else:
        print("The null hypothesis is not rejected")

    # if the ks_p is smaller than the alpha
    if ks_p < lower_alpha:
        print("The null hypothesis is rejected at the 1% level")
        print("The two samples are not drawn from the same distribution")
    elif ks_p < upper_alpha:
        print("The null hypothesis is rejected at the 5% level")
        print("The two samples are not drawn from the same distribution")
    else:
        print("The null hypothesis is not rejected")

    # set the current time
    now = datetime.now()

    # set the current date
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_{date}.pdf"))

    return


# Define a function to plot the quantile quantile plot
# comparing the two distributions
def plot_qq(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    save_prefix: str = "qq",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the quantile-quantile plot comparing the observations and model data.

    Parameters
    ==========

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

    save_prefix: str
        The prefix to use when saving the plots. Default is "qq".

    save_dir: str
        The directory to save the plots to. Default is the current directory.

    Returns
    =======

    None

    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # set up the quantiles
    quantiles = np.linspace(0, 1, 100)

    quantiles_x = np.linspace(0, 10, 1000)

    # set up qunatiles y for the 1 to 1 line
    quantiles_y = np.linspace(0, 10, 1000)

    # if the datsets contain NaNs
    if (
        obs_df[obs_val_name].isnull().values.any()
        or model_df[model_val_name].isnull().values.any()
    ):
        print("The datasets contain NaNs")

        # Drop NaNs from the observation and model data
        obs_df_clean = obs_df.dropna(subset=[obs_val_name])
        model_df_clean = model_df.dropna(subset=[model_val_name])

        # Set up the quantiles
        obs_quantiles = np.quantile(obs_df_clean[obs_val_name], quantiles)

        # Set up the quantiles
        model_quantiles = np.quantile(model_df_clean[model_val_name], quantiles)
    else:
        # set up the quantiles for the obs data
        obs_quantiles = np.quantile(obs_df[obs_val_name], quantiles)

        # set up the quantiles for the model data
        model_quantiles = np.quantile(model_df[model_val_name], quantiles)

    # # plot the 1:1 line
    # ax.plot(quantiles_x, quantiles_y, color="black", linestyle="--")

    # round down from the value of the lowest quantile
    min_val = np.floor(min(min(obs_quantiles), min(model_quantiles)))

    # round up from the value of the highest quantile
    max_val = np.ceil(max(max(obs_quantiles), max(model_quantiles)))

    # create a x = y line
    x = np.linspace(min_val, max_val, 100)

    # plot the quantile-quantile plot
    ax.plot(x, x, color="black", linestyle="--")

    # plot the obs and model quantiles
    ax.plot(obs_quantiles, model_quantiles, color="red", linestyle="None", marker="o")

    # set the x-axis label
    ax.set_xlabel("ERA5")

    # set the y-axis label
    ax.set_ylabel("HadGEM3-GC31-MM")

    # # set the title
    # ax.set_title("Quantile-Quantile Plot")

    # set the current time
    now = datetime.now()

    # set the current date
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_{date}.pdf"))

    return


# Define a function to plot the composite SLP patterns for the obs
def plot_composite_obs(
    obs_df: pd.DataFrame,
    obs_val_name: str,
    percentile: float,
    title: str,
    calc_anoms: bool = False,
    months: list[int] = [10, 11, 12, 1, 2, 3],
    climatology_period: list[int] = [1990, 2020],
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    psl_variable: str = "msl",
    freq: str = "Amon",
    obs_time_name: str = "time",
    save_prefix: str = "composite_obs",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
    regrid_file: str = "/gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1_gn_196011-197103.nc",
) -> None:
    """

    Plots the composite SLP patterns for the observations.

    Args:
        obs_df (pd.DataFrame): The DataFrame containing the observations with columns for the
        observation value and the observation time.

        obs_val_name (str): The name of the observation value column.

        percentile (float): The percentile to use for the composite. E.g. 0.95 for the 95th percentile.

        climatology_period (list[int]): The period to use for the climatology. Default is [1990, 2020].

        lat_bounds (list): The latitude bounds to use for the composite. Default is [30, 80].

        lon_bounds (list): The longitude bounds to use for the composite. Default is [-90, 30].

        psl_variable (str): The name of the variable to use for the composite. Default is "msl".

        freq (str): The frequency of the data. Default is "Amon".

        save_prefix (str): The prefix to use when saving the plots. Default is "composite_obs".

        save_dir (str): The directory to save the plots to. Default is the current directory.

    Returns:
        None
    """

    # Set up the regrid ERA5 path
    regrid_era5_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

    # Work out the percentile threshold for the obs data
    obs_threshold = np.percentile(obs_df[obs_val_name], percentile)

    # print the len of the full obs_df
    print(f"The length of the obs df is {len(obs_df)}")

    # Apply a boolean to the df to where values are beneath
    # this threshold
    obs_df_composite = obs_df[obs_df[obs_val_name] < obs_threshold]

    # the percentile is print
    print(f"The {percentile}th percentile is {obs_threshold}")

    # Print the len of the obs df composite
    print(f"The length of the obs df composite is {len(obs_df_composite)}")

    # print the head of the obs df composite
    print(obs_df_composite.head())

    # # Load the regridded ERA5 data
    # ds = xr.open_mfdataset(
    #     regrid_era5_path,
    #     combine="by_coords",
    #     parallel=False,
    #     engine="netcdf4",
    #     coords="minimal",
    # )

    # # If expver is present in the observations
    # if "expver" in ds.coords:
    #     # Combine the first two expver variables
    #     ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # # convert ds to a cube
    # cube = ds.to_iris()

    # # load in the ERA5 data with iris
    cube = iris.load_cube(regrid_era5_path, psl_variable)

    # load the sample file
    model_cube_example = iris.load_cube(regrid_file)

    # regrid the cube to the model cube
    cube = cube.regrid(model_cube_example, iris.analysis.Linear())

    # print the regrid cube
    print(cube)

    # subset the cube to the correct grid
    cube = cube.intersection(longitude=(-180, 180), latitude=(-90, 90))

    # print the cube
    print(cube)

    # print the lats
    print(cube.coord("latitude").points)

    # print the lons
    print(cube.coord("longitude").points)

    # prtint the lat bounds
    print(lat_bounds)

    # print the lon bounds
    print(lon_bounds)

    # Select the data for expver=1 and expver=5
    cube_expver1 = cube.extract(iris.Constraint(expver=1))
    cube_expver5 = cube.extract(iris.Constraint(expver=5))

    # # Merge the two cubes
    # cube = iris.cube.CubeList([cube_expver1, cube_expver5]).concatenate()

    # print the cube
    print(cube_expver1)

    # assuming that this has most of the data
    cube = cube_expver1

    # if calc anoms is true
    if calc_anoms:
        print("Calculating the climatology for the observations")

        # assert that the months are integers
        assert all(
            isinstance(month, int) for month in months
        ), "Months must be integers"

        # subset the data to the region
        cube_clim = cube.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # print the months
        print(months)

        # set up the months constraint
        months_constraint = iris.Constraint(
            time=lambda cell: cell.point.month in months
        )

        # subset the data to the months
        cube_clim = cube_clim.extract(months_constraint)

        # set up the years constraint
        years_constraint = iris.Constraint(
            time=lambda cell: cell.point.year in climatology_period
        )

        # Select the years
        cube_clim = cube_clim.extract(years_constraint)

        # print cube clim
        print(cube_clim)

        # print the type of cub clime
        print(type(cube_clim))

        # Calculate the climatology
        cube_clim = cube_clim.collapsed("time", iris.analysis.MEAN)

    ds_list = []

    # Set up an empty list
    for i, time in enumerate(obs_df_composite[obs_time_name]):
        # Subset the data to the region
        cube_subset = cube.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # Subset the data to the time
        cube_subset = cube_subset.extract(iris.Constraint(time=time))

        # # Add a new coordinate 'number' to the cube
        # number_coord = iris.coords.AuxCoord(i, long_name="number", units="1")

        # # add this as a dimensioned coordinate
        # cube_subset.add_dim_coord(number_coord, 0)

        # # print the cube subset
        # print(cube_subset)

        # append the cube to the list
        ds_list.append(cube_subset)

    # print ds_list
    print(ds_list)

    # make sure ds_list is an iris cube list
    ds_list = iris.cube.CubeList(ds_list)

    # remove the attributes which don't match up
    removed_attributes = equalise_attributes(ds_list)

    # Concatenate the list with a time dimension
    ds_composite = ds_list.merge_cube()

    # print ds copmosite
    print(ds_composite)

    # print the type of ds_compopsite
    print(type(ds_composite))

    # take the mean over the time dimension
    ds_composite = ds_composite.collapsed("time", iris.analysis.MEAN)

    # Etract the lat and lon points
    lats = ds_composite.coord("latitude").points
    lons = ds_composite.coord("longitude").points

    # if calc_anoms is True
    if calc_anoms:
        # Calculate the anomalies
        field = (ds_composite.data - cube_clim.data) / 100  # convert to hPa
    else:
        # Extract the data values
        field = ds_composite.data / 100  # convert to hPa

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    # Set up the num events
    num_events = len(obs_df_composite)

    # include a textbox in the top left
    ax.text(
        0.02,
        0.95,
        f"N = {num_events}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    # Set up the current date time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    return


# Formatting functions for 4 significant figures and 1 decimal point
def format_func(
    x: float,
    pos: int,
):
    """
    Formats the x-axis ticks as significant figures.

    Args:
        x (float): The tick value.
        pos (int): The position of the tick.

    Returns:
        str: The formatted tick value.
    """
    return f"{x:.4g}"


def format_func_one_decimal(
    x: float,
    pos: int,
):
    """
    Formats the x-axis ticks to one decimal point.

    Args:
        x (float): The tick value.
        pos (int): The position of the tick.

    Returns:
        str: The formatted tick value.
    """
    return f"{x:.1f}"


# define a function to plot the composite SLP events
# for the model
def plot_composite_model(
    model_df: pd.DataFrame,
    model_val_name: str,
    percentile: float,
    title: str,
    model: str = "HadGEM3-GC31-MM",
    psl_variable: str = "psl",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    calc_anoms: bool = False,
    months: list[int] = [10, 11, 12, 1, 2, 3],
    climatology_period: list[int] = [1990, 2020],
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
    save_prefix: str = "composite_model",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the composite SLP events for the model data.

    Args:
        model_df (pd.DataFrame): The DataFrame containing the model data with columns for the model value.
        model_val_name (str): The name of the model value column.
        percentile (float): The percentile to use for the composite.
        model (str, optional): The name of the model. Defaults to "HadGEM3-GC31-MM".
        psl_variable (str, optional): The name of the sea level pressure variable. Defaults to "psl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        experiment (str, optional): The name of the experiment. Defaults to "dcppA-hindcast".
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.
        months (list[int], optional): The months to include in the composite. Defaults to [10, 11, 12, 1, 2, 3].
        climatology_period (list[int], optional): The period to use for climatology. Defaults to [1990, 2020].
        lat_bounds (list, optional): The latitude bounds for the composite. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude bounds for the composite. Defaults to [-90, 30].
        files_loc_path (str, optional): The path to the file location. Defaults to "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv".

    Returns:
        None
    """

    # Work out the percentile threshold for the model data
    model_threshold = np.percentile(model_df[model_val_name], percentile)

    # Print the full len of the model df
    print(f"The length of the model df is {len(model_df)}")

    # Apply a boolean to the df to where values are beneath
    model_df_composite = model_df[model_df[model_val_name] < model_threshold]

    # Print the percentile value
    print(f"The {percentile}th percentile is {model_threshold}")

    # Print the len of the model df composite
    print(f"The length of the model df composite is {len(model_df_composite)}")

    # print the head of the model df composite
    print(model_df_composite.head())

    # extract the unique members
    unique_members = np.unique(model_df_composite["member"])

    # assert that the files loc path exists
    assert os.path.exists(files_loc_path), "The files loc path does not exist"

    # Load the files location
    files_loc = pd.read_csv(files_loc_path)

    # print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"freq: {freq}")
    print(f"psl_variable: {psl_variable}")

    # # extract the model_path_var
    # model_path_var = files_loc.loc[
    #     (files_loc["model"] == model)
    #     & (files_loc["experiment"] == experiment)
    #     & (files_loc["frequency"] == freq)
    #     & (files_loc["variable"] == sf_variable)
    # ]["path"].values[0]

    # Extract the path for the given model, experiment, freq, and variable
    model_path_psl = files_loc.loc[
        (files_loc["model"] == model)
        & (files_loc["experiment"] == experiment)
        & (files_loc["frequency"] == freq)
        & (files_loc["variable"] == psl_variable)
    ]["path"].values[0]

    # asser that the model path psl exists
    assert os.path.exists(model_path_psl), "The model path psl does not exist"

    # extract the model path root
    model_path_root_psl = model_path_psl.split("/")[1]

    # Set up an empty list of files
    files_list = []

    # Extract unique (init_year, member) pairs
    unique_year_member_pairs = model_df_composite[
        ["init_year", "member"]
    ].drop_duplicates()
    unique_year_member_pairs = list(
        unique_year_member_pairs.itertuples(index=False, name=None)
    )

    # print(unique_year_member_pairs)

    # loop over the unique year member pairs
    for year, member in unique_year_member_pairs:
        # find the leads to extract
        # leads_ym = model_df_composite.loc[
        #     (model_df_composite["init_year"] == year)
        #     & (model_df_composite["member"] == member)
        # ]["lead"].values

        # # print the year and member
        # print(f"year: {year}, member: {member}")

        # # print the lead
        # print(f"leads: {leads_ym}")

        if model_path_root_psl == "work":
            raise NotImplementedError("work path not implemented yet")
        elif model_path_root_psl == "gws":
            # Create the path
            path = f"{model_path_psl}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

            # glob this path
            files = glob.glob(path)

            # assert that files has length 1
            assert len(files) == 1, f"files has length {len(files)}"

            # extract the file
            file = files[0]

            # append the file to the files list
            files_list.append(file)
        elif model_path_root_psl == "badc":
            # Form the path to the files
            year_path = f"{model_path_psl}/s{year}-r*i?p?f?/{freq}/{psl_variable}/g?/files/d????????/*.nc"

            # Glob the files in the directory containing the initialisation year
            files = glob.glob(year_path)

            # assert that files has len of more than zero
            assert len(files) > 0, f"files has length {len(files)}"

            # Extend the files to the aggregated files list
            files_list.extend(files)
        else:
            raise ValueError(f"Unknown model path root {model_path_root_psl}")

    # # print the files
    # print(f"The files are {files_list}")

    # print the len of files
    print(f"The length of files is {len(files_list)}")

    # # print the unique year member pairs
    # print(f"The unique year member pairs are {unique_year_member_pairs}")

    # print the len of the unique year member paris
    print(
        f"The length of the unique year member pairs is {len(unique_year_member_pairs)}"
    )

    # create an empty list for the files
    dss = []

    # if the variable is psl
    if psl_variable == "psl":
        # loop over the files and year members
        for idx, (file, (year, member)) in tqdm(
            enumerate(zip(files_list, unique_year_member_pairs)), total=len(files_list)
        ):
            # Your existing code here
            # find the leads to extract
            leads_ym = model_df_composite.loc[
                (model_df_composite["init_year"] == year)
                & (model_df_composite["member"] == member)
            ]["lead"].values

            # load the file
            ds = xr.open_dataset(file)

            # Format the lead as an int
            ds = set_integer_time_axis(
                xro=ds,
                frequency=freq,
            )

            # loop over the leads
            for lead in leads_ym:
                # select the leads from the time variable
                ds = ds.sel(time=leads_ym)

                # Add a new coordinate 'number' with a unique value
                ds = ds.expand_dims({"number": [idx]})

                # Append the ds to the dss list
                dss.append(ds[psl_variable])
    elif psl_variable == "zg":

        # make sure there are no duplicates in the files_list
        files_list = list(set(files_list))

        # loop over the year members
        for idx, (year, member) in tqdm(
            enumerate(unique_year_member_pairs), total=len(unique_year_member_pairs)
        ):
            # Find the leads to extract
            leads_ym = model_df_composite.loc[
                (model_df_composite["init_year"] == year)
                & (model_df_composite["member"] == member)
            ]["lead"].values

            # Find the files containing the year and member
            files = [file for file in files_list if f"s{year}-r{member}i" in file]

            # print the len of files
            print(f"The length of files is {len(files)}")

            # print the files
            print(files)

            # FIXME: Fix this
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

    # Concatenate all datasets along the 'number' dimension
    combined_ds = xr.concat(dss, dim="number")

    # print the shape of the combined ds
    print(f"The shape of the combined ds is {combined_ds.shape}")

    # Take the mean over the 'number' dimension
    mean_ds = combined_ds.mean(dim="number")

    # convert to a cube
    cube_psl = mean_ds.to_iris()

    # # print the cube psl
    # print(cube_psl)

    # # print the lats and the lons
    # print(cube_psl.coord("latitude").points)
    # print(cube_psl.coord("longitude").points)

    # subset to region of interest
    cube_psl = cube_psl.intersection(longitude=(-180, 180), latitude=(-90, 90))

    # subset to the actual region of interest
    cube_psl = cube_psl.intersection(
        longitude=(lon_bounds[0], lon_bounds[1]),
        latitude=(lat_bounds[0], lat_bounds[1]),
    )

    # if calc anoms is true
    if calc_anoms:

        # set up a save directory for the climatologies
        save_dir_clim = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_clim"

        # assert that this directory exists
        assert os.path.exists(save_dir_clim), "The save directory does not exist"

        # set up the fname
        fname = f"climatology_{model}_{experiment}_{freq}_{psl_variable}_{climatology_period[0]}-{climatology_period[1]}_{lat_bounds[0]}-{lat_bounds[1]}_{lon_bounds[0]}-{lon_bounds[1]}_{months[0]}-{months[-1]}.nc"

        # set up the full climatology path
        climatology_path = os.path.join(save_dir_clim, fname)

        # if the climatology path exists
        if os.path.exists(climatology_path):
            print("The climatology file exists")
            print("Loading the climatology file")

            # load the file using iris
            cube_clim = iris.load_cube(climatology_path)

        else:
            print("The climatology file does not exist")
            print("Calculating the climatology")

            # Set up a list for the full ds's
            clim_dss = []
            # Loop over the years
            for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
                member_list = []
                for member in unique_members:
                    start_time = time.time()

                    path = f"{model_path_psl}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                    # glob this path
                    glob_start = time.time()
                    files = glob.glob(path)
                    glob_end = time.time()

                    # assert
                    assert (
                        len(files) == 1
                    ), f"files has length {len(files)} for year {year} and member {member} and path {path}"

                    # open all of the files
                    # open_start = time.time()
                    member_ds = xr.open_mfdataset(
                        files[0],
                        combine="nested",
                        concat_dim="time",
                        preprocess=lambda ds: preprocess(
                            ds=ds,
                            year=year,
                            variable=psl_variable,
                            months=months,
                        ),
                        parallel=True,
                        engine="netcdf4",
                        coords="minimal",  # expecting identical coords
                        data_vars="minimal",  # expecting identical vars
                        compat="override",  # speed up
                    ).squeeze()
                    # open_end = time.time()

                    # id init year == climatology_period[0]
                    # and member == unique_members[0]
                    # set_time_start = time.time()
                    if year == climatology_period[0] and member == unique_members[0]:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                            first_month_attr=True,
                        )
                    else:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                        )
                    # set_time_end = time.time()

                    # take the mean over the time axis
                    # mean_start = time.time()
                    member_ds = member_ds.mean(dim="time")
                    # mean_end = time.time()

                    # append the member_ds to the member_list
                    member_list.append(member_ds)

                    # end_time = time.time()

                    # # Print timing information
                    # print(f"Year: {year}, Member: {member}")
                    # print(f"  Total time: {end_time - start_time:.2f} seconds")
                    # print(f"  glob time: {glob_end - glob_start:.2f} seconds")
                    # print(f"  open_mfdataset time: {open_end - open_start:.2f} seconds")
                    # print(f"  set_integer_time_axis time: {set_time_end - set_time_start:.2f} seconds")
                    # print(f"  mean time: {mean_end - mean_start:.2f} seconds")

                # Concatenate with a new member dimension using xarray
                member_ds = xr.concat(member_list, dim="member")
                # append the member_ds to the init_year_list
                clim_dss.append(member_ds)
            # Concatenate the init_year list along the init dimension
            # and rename as lead time
            ds = xr.concat(clim_dss, "init")

            # print ds
            print(ds)

            # set up the members
            ds["member"] = unique_members
            ds["init"] = np.arange(climatology_period[0], climatology_period[1] + 1)

            # extract the variable
            ds_var = ds[psl_variable]

            # # take the mean over lead dimension
            # ds_clim = ds_var.mean(dim="lead")

            # take the mean over member dimension
            ds_clim = ds_var.mean(dim="member")

            # take the mean over init dimension
            ds_clim = ds_clim.mean(dim="init")

            # convert to a cube
            cube_clim = ds_clim.to_iris()

            # # regrid the model data to the obs grid
            # cube_clim_regrid = cube_clim.regrid(cube_obs, iris.analysis.Linear())

            # subset to the correct grid
            cube_clim = cube_clim.intersection(
                longitude=(-180, 180), latitude=(-90, 90)
            )

            # subset to the region of interest
            cube_clim = cube_clim.intersection(
                latitude=(lat_bounds[0], lat_bounds[1]),
                longitude=(lon_bounds[0], lon_bounds[1]),
            )

            # if the climatology file does not exist
            print("Saving the climatology file")

            # save the cube_clim
            iris.save(cube_clim, climatology_path)

    # extract the lats and lons
    lats = cube_psl.coord("latitude").points
    lons = cube_psl.coord("longitude").points

    # if calc_anoms is True
    if calc_anoms:
        field = (cube_psl.data - cube_clim.data) / 100  # convert to hPa
    else:
        field = cube_psl.data / 100

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    # set up the num events
    num_events = len(model_df_composite)

    # include a textbox in the top left
    ax.text(
        0.02,
        0.95,
        f"N = {num_events}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    # Set up the current date time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    return


# Define a function for preprocessing the model data
def preprocess_boilerplate(
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


# define a function for preprocessing
def preprocess(
    ds: xr.Dataset,
    year: int,
    variable: str,
    months: list[int] = [11, 12, 1, 2, 3],
) -> xr.Dataset:
    """
    Preprocesses the model data by subsetting to the months of interest.

    Args:
        ds (xr.Dataset): The model dataset to be preprocessed.
        year (int): The year of the data.
        variable (str): The variable to be preprocessed.
        months (list[int], optional): The months to be used for the preprocessing. Defaults to [11, 12, 1, 2, 3].

    Returns:
        xr.Dataset: The preprocessed model dataset.
    """

    # if year is not an int, format as an int
    if not isinstance(year, int):
        year = int(year)

    # if the variable is not in the ds
    if variable not in ds:
        raise ValueError(f"Cannot find the variable {variable} in the ds")

    # # Set up the times to extract
    # start_date_this = cftime.datetime.strptime(
    #     f"{year}-{months[0]}-01", "%Y-%m-%d", calendar="360_day"
    # )
    # end_date_this = cftime.datetime.strptime(
    #     f"{year + 1}-{months[-1]}-30", "%Y-%m-%d", calendar="360_day"
    # )

    # # slice between the start and end dates
    # ds = ds.sel(time=slice(start_date_this, end_date_this))

    # extract the specific months
    ds = ds.sel(time=ds["time.month"].isin(months))

    return ds


# define a function to plot both composites for the obs and the model
def plot_composite_obs_model(
    obs_df: pd.DataFrame,
    obs_val_name: str,
    obs_time_name: str,
    model_df: pd.DataFrame,
    model_val_name: str,
    percentile: float,
    variable: str,
    nboot: int = 1000,
    obs_variable: str = "msl",
    model: str = "HadGEM3-GC31-MM",
    psl_variable: str = "psl",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    calc_anoms: bool = False,
    months: list[int] = [10, 11, 12, 1, 2, 3],
    climatology_period: list[int] = [1990, 2018],
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
    save_prefix: str = "composite_obs_model",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/unseen",
    regrid_file: str = "/gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1_gn_196011-197103.nc",
) -> None:
    """
    Plots the composite SLP events for both the observations and the model data.

    Args:
        obs_df (pd.DataFrame): The DataFrame containing the observation data with columns for the observation value and the observation time.
        obs_val_name (str): The name of the observation value column.
        obs_time_name (str): The name of the observation time column.
        model_df (pd.DataFrame): The DataFrame containing the model data with columns for the model value.
        model_val_name (str): The name of the model value column.
        percentile (float): The percentile to use for the composite.
        title (str): The title of the plot.
        obs_variable (str, optional): The name of the observation variable. Defaults to "msl".
        model (str, optional): The name of the model. Defaults to "HadGEM3-GC31-MM".
        psl_variable (str, optional): The name of the sea level pressure variable. Defaults to "psl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        experiment (str, optional): The name of the experiment. Defaults to "dcppA-hindcast".
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.
        months (list[int], optional): The months to include in the composite. Defaults to [10, 11, 12, 1, 2, 3].
        climatology_period (list[int], optional): The period to use for climatology. Defaults to [1990, 2018].
        lat_bounds (list, optional): The latitude bounds for the composite. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude bounds for the composite. Defaults to [-90, 30].
        files_loc_path (str, optional): The path to the file location. Defaults to "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv".
        save_prefix (str, optional): The prefix to use for the saved plot. Defaults to "composite_obs_model".
        save_dir (str, optional): The directory to save the plot in. Defaults to "/gws/nopw/j04/canari/users/benhutch/plots".

    Returns:
        None
    """

    # Set up the regrid ERA5 path
    regrid_era5_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

    # if the months list is not [10, 11, 12, 1, 2, 3]
    # then subset the obs_df to these months
    if months != [10, 11, 12, 1, 2, 3]:
        print("Subsetting the obs_df to the months of interest: ", months)

        # assert that time is a datetime
        assert isinstance(obs_df[obs_time_name].iloc[0], pd.Timestamp)

        # subset the obs_df to the months
        obs_df = obs_df[obs_df[obs_time_name].dt.month.isin(months)]

    # Work out the percentile threshold for the obs data
    obs_threshold = np.percentile(obs_df[obs_val_name], percentile)

    # print the len of the full obs_df
    print(f"The length of the obs df is {len(obs_df)}")

    # print the head of the obs_df for checking
    print(obs_df.head())

    # Apply a boolean to the df to where values are beneath
    # this threshold
    obs_df_composite = obs_df[obs_df[obs_val_name] < obs_threshold]

    # Set up the number of obs events
    num_obs_events = len(obs_df_composite)

    # the percentile is print
    print(f"The {percentile}th percentile is {obs_threshold}")

    # Print the len of the obs df composite
    print(f"The length of the obs df composite is {len(obs_df_composite)}")

    # print the head of the obs df composite
    print(obs_df_composite.head())

    # # Load the regridded ERA5 data
    # ds = xr.open_mfdataset(
    #     regrid_era5_path,
    #     combine="by_coords",
    #     parallel=False,
    #     engine="netcdf4",
    #     coords="minimal",
    # )

    # # If expver is present in the observations
    # if "expver" in ds.coords:
    #     # Combine the first two expver variables
    #     ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # # convert ds to a cube
    # cube = ds.to_iris()

    # # load in the ERA5 data with iris
    cube = iris.load_cube(regrid_era5_path, obs_variable)

    # load the sample file
    model_cube_example = iris.load_cube(regrid_file)

    # regrid the cube to the model cube
    cube = cube.regrid(model_cube_example, iris.analysis.Linear())

    # print the regrid cube
    print(cube)

    # subset the cube to the correct grid
    cube = cube.intersection(longitude=(-180, 180), latitude=(-90, 90))

    # print the cube
    print(cube)

    # print the lats
    print(cube.coord("latitude").points)

    # print the lons
    print(cube.coord("longitude").points)

    # prtint the lat bounds
    print(lat_bounds)

    # print the lon bounds
    print(lon_bounds)

    # Select the data for expver=1 and expver=5
    cube_expver1 = cube.extract(iris.Constraint(expver=1))
    cube_expver5 = cube.extract(iris.Constraint(expver=5))

    # # Merge the two cubes
    # cube = iris.cube.CubeList([cube_expver1, cube_expver5]).concatenate()

    # print the cube
    print(cube_expver1)

    # assuming that this has most of the data
    cube = cube_expver1

    # if calc anoms is true
    if calc_anoms:
        print("Calculating the climatology for the observations")

        # assert that the months are integers
        assert all(
            isinstance(month, int) for month in months
        ), "Months must be integers"

        # subset the data to the region
        cube_clim = cube.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # print the months
        print(months)

        # set up the months constraint
        months_constraint = iris.Constraint(
            time=lambda cell: cell.point.month in months
        )

        # subset the data to the months
        cube_clim = cube_clim.extract(months_constraint)

        # set up the years constraint
        years_constraint = iris.Constraint(
            time=lambda cell: cell.point.year in climatology_period
        )

        # Select the years
        cube_clim = cube_clim.extract(years_constraint)

        # print cube clim
        print(cube_clim)

        # print the type of cub clime
        print(type(cube_clim))

        # Calculate the climatology
        cube_clim = cube_clim.collapsed("time", iris.analysis.MEAN)

    ds_list = []

    # Set up an empty list
    for i, time in enumerate(obs_df_composite[obs_time_name]):
        # Subset the data to the region
        cube_subset = cube.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # Subset the data to the time
        cube_subset = cube_subset.extract(iris.Constraint(time=time))

        # # Add a new coordinate 'number' to the cube
        # number_coord = iris.coords.AuxCoord(i, long_name="number", units="1")

        # # add this as a dimensioned coordinate
        # cube_subset.add_dim_coord(number_coord, 0)

        # # print the cube subset
        # print(cube_subset)

        # append the cube to the list
        ds_list.append(cube_subset)

    # print ds_list
    print(ds_list)

    # make sure ds_list is an iris cube list
    ds_list = iris.cube.CubeList(ds_list)

    # remove the attributes which don't match up
    removed_attributes = equalise_attributes(ds_list)

    # Concatenate the list with a time dimension
    ds_composite_full = ds_list.merge_cube()

    # print ds copmosite
    print(ds_composite_full)

    # print the type of ds_compopsite
    print(type(ds_composite_full))

    # take the mean over the time dimension
    ds_composite = ds_composite_full.collapsed("time", iris.analysis.MEAN)

    # # Etract the lat and lon points
    # lats = ds_composite.coord("latitude").points
    # lons = ds_composite.coord("longitude").points

    # Set up the leads
    leads = np.arange(1, 10 + 1)

    # if the months list is not [10, 11, 12, 1, 2, 3]
    # then subset the obs_df to these months
    if months != [10, 11, 12, 1, 2, 3]:
        print("Subsetting the model_df to the months of interest: ", months)

        # intialize an empty list of leads
        leads_sel = []

        # loop over the leads
        for l in leads:
            # if months is [10, 11, 12]
            if months == [10, 11, 12]:  # OND
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l), (12 * l) + 1, (12 * l) + 2])
            elif months == [11, 12]:
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 1, (12 * l) + 2])
            elif months == [11, 12, 1]:  # NDJ
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 1, (12 * l) + 2, (12 * l) + 3])
            elif months == [12]:
                # append the leads to the leads_sel
                leads_sel.append((12 * l) + 2)
            elif months == [12, 1, 2]:  # DJF
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 2, (12 * l) + 3, (12 * l) + 4])
            elif months == [1, 2, 3]:  # JFM
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 3, (12 * l) + 4, (12 * l) + 5])
            elif months == [2, 3]:
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 4, (12 * l) + 5])
            elif months == [1, 2]:
                # append the leads to the leads_sel
                leads_sel.extend([(12 * l) + 3, (12 * l) + 4])
            else:
                raise ValueError(f"Unknown months {months}")

        # print the leds sel
        print(leads_sel)

        # subset the model_df to the leads_sel
        model_df = model_df[model_df["lead"].isin(leads_sel)]

    # print the head of the model df for checking
    print(model_df.head())

    # print the tail
    print(model_df.tail())

    # Work out the percentile threshold for the model data
    model_threshold = np.percentile(model_df[model_val_name], percentile)

    # Print the full len of the model df
    print(f"The length of the model df is {len(model_df)}")

    # Apply a boolean to the df to where values are beneath
    model_df_composite = model_df[model_df[model_val_name] < model_threshold]

    # print the model df composite
    print(model_df_composite)

    # # set up the numer of events
    # num_obs_events = len(model_df_composite)

    # Print the percentile value
    print(f"The {percentile}th percentile is {model_threshold}")

    # Print the len of the model df composite
    print(f"The length of the model df composite is {len(model_df_composite)}")

    num_model_events = len(model_df_composite)

    # print the head of the model df composite
    print(model_df_composite.head())

    # extract the unique members
    unique_members = np.unique(model_df_composite["member"])

    # assert that the files loc path exists
    assert os.path.exists(files_loc_path), "The files loc path does not exist"

    # Load the files location
    files_loc = pd.read_csv(files_loc_path)

    # print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"freq: {freq}")
    print(f"psl_variable: {psl_variable}")

    # # extract the model_path_var
    # model_path_var = files_loc.loc[
    #     (files_loc["model"] == model)
    #     & (files_loc["experiment"] == experiment)
    #     & (files_loc["frequency"] == freq)
    #     & (files_loc["variable"] == sf_variable)
    # ]["path"].values[0]

    # Extract the path for the given model, experiment, freq, and variable
    model_path_psl = files_loc.loc[
        (files_loc["model"] == model)
        & (files_loc["experiment"] == experiment)
        & (files_loc["frequency"] == freq)
        & (files_loc["variable"] == psl_variable)
    ]["path"].values[0]

    # asser that the model path psl exists
    assert os.path.exists(model_path_psl), "The model path psl does not exist"

    # extract the model path root
    model_path_root_psl = model_path_psl.split("/")[1]

    # Set up an empty list of files
    files_list = []

    # Extract unique (init_year, member) pairs
    unique_year_member_pairs = model_df_composite[
        ["init_year", "member"]
    ].drop_duplicates()
    unique_year_member_pairs = list(
        unique_year_member_pairs.itertuples(index=False, name=None)
    )

    # print(unique_year_member_pairs)

    # loop over the unique year member pairs
    for year, member in unique_year_member_pairs:
        # find the leads to extract
        # leads_ym = model_df_composite.loc[
        #     (model_df_composite["init_year"] == year)
        #     & (model_df_composite["member"] == member)
        # ]["lead"].values

        # # print the year and member
        # print(f"year: {year}, member: {member}")

        # # print the lead
        # print(f"leads: {leads_ym}")

        if model_path_root_psl == "work":
            raise NotImplementedError("work path not implemented yet")
        elif model_path_root_psl == "gws":
            # Create the path
            path = f"{model_path_psl}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

            # glob this path
            files = glob.glob(path)

            # assert that files has length 1
            assert len(files) == 1, f"files has length {len(files)}"

            # extract the file
            file = files[0]
        elif model_path_root_psl == "badc":
            raise NotImplementedError("home path not implemented yet")
        else:
            raise ValueError(f"Unknown model path root {model_path_root_psl}")

        # append the file to the files list
        files_list.append(file)

    # # print the files
    # print(f"The files are {files_list}")

    # print the len of files
    print(f"The length of files is {len(files_list)}")

    # # print the unique year member pairs
    # print(f"The unique year member pairs are {unique_year_member_pairs}")

    # print the len of the unique year member paris
    print(
        f"The length of the unique year member pairs is {len(unique_year_member_pairs)}"
    )

    # # FIXME: limit to first 30 for testing purposes
    # print("WARNING: Limiting to first 30 for testing purposes")
    # files_list = files_list[:30]
    # unique_year_member_pairs = unique_year_member_pairs[:30]
    # print("WARNING: Limiting to first 30 for testing purposes")

    # create an empty list for the files
    dss = []

    # loop over the files and year members
    for idx, (file, (year, member)) in tqdm(
        enumerate(zip(files_list, unique_year_member_pairs)), total=len(files_list)
    ):
        # Your existing code here
        # find the leads to extract
        leads_ym = model_df_composite.loc[
            (model_df_composite["init_year"] == year)
            & (model_df_composite["member"] == member)
        ]["lead"].values

        # load the file
        ds = xr.open_dataset(file)

        # Format the lead as an int
        ds = set_integer_time_axis(
            xro=ds,
            frequency=freq,
        )

        # Loop over the leads
        for i, lead in enumerate(leads_ym):
            # select the leads from the time variable
            ds_lead = ds.sel(time=lead)

            # Add a new coordinate 'number' with a unique value
            ds_lead = ds_lead.expand_dims({"number": [idx]})

            # Append the ds to the dss list
            dss.append(ds_lead[psl_variable])

    # Concatenate all datasets along the 'number' dimension
    combined_ds = xr.concat(dss, dim="number")

    # # print combined_ds
    # print("The combined_ds is", combined_ds)

    # print the shape of the combined ds
    print(f"The shape of the combined ds is {combined_ds.shape}")

    # convert to a cube
    cube_psl = combined_ds.to_iris()

    # subset to teh correct grid
    cube_psl = cube_psl.intersection(longitude=(-180, 180), latitude=(-90, 90))

    # subset to the region of interest
    cube_psl = cube_psl.intersection(
        longitude=(lon_bounds[0], lon_bounds[1]),
        latitude=(lat_bounds[0], lat_bounds[1]),
    )

    # # print the lats and lons
    # print(cube_psl.coord("latitude").points)
    # print(cube_psl.coord("longitude").points)

    # # print obs lats and lons
    # print(lats)
    # print(lons)

    # print the combined ds
    print("The cube_psl is", cube_psl)

    # regrid the obs data to the same as the model data
    print("Regridding the obs data to the model data")
    cube_obs = ds_composite.regrid(cube_psl, iris.analysis.Linear())
    cube_obs_full = ds_composite_full.regrid(cube_psl, iris.analysis.Linear())
    cube_clim = cube_clim.regrid(cube_psl, iris.analysis.Linear())

    # regrid ds_composite full (which has all the obs events in)
    print("Regridding the obs data to the model data")
    ds_composite_full_regrid = ds_composite_full.regrid(
        cube_psl, iris.analysis.Linear()
    )

    # assert that combined_ds lats array == lats
    assert np.allclose(
        cube_psl.coord("latitude").points, cube_obs.coord("latitude").points
    ), "The lats do not match"

    # assert that combined_ds lons array == lons
    assert np.allclose(
        cube_psl.coord("longitude").points, cube_obs.coord("longitude").points
    ), "The lons do not match"

    # if calc_anoms is True
    if calc_anoms:
        # Calculate the anomalies
        field_obs = (cube_obs.data - cube_clim.data) / 100  # convert to hPa

        # calculate the anomalies
        field_obs_full = (cube_obs_full.data - cube_clim.data) / 100  # convert to hPa
    else:
        # Extract the data values
        field_obs = cube_obs.data / 100  # convert to hPa

        # Extract the data values
        field_obs_full = cube_obs_full.data / 100  # convert to hPa

    # print the shape of field_obs_full
    print(f"The shape of field_obs_full is {field_obs_full.shape}")

    # print the values of the field_obs_full
    print(f"The values of field_obs_full are {field_obs_full}")

    # extract combined_ds as an array
    combined_ds_arr = cube_psl.data

    # set up the nrows for the data
    nrows = combined_ds_arr.shape[0]

    # print the nrows
    print(f"The number of rows is {nrows}")

    # set up the shapes of the arrays to be filled
    nlats = len(cube_psl.coord("latitude").points)
    nlons = len(cube_psl.coord("longitude").points)

    # print the num obs events
    print(f"The number of obs events is {num_obs_events}")

    # set up an empty array of zeros to be filled
    model_boot = np.zeros([nboot, num_obs_events, nlats, nlons])

    # Set up the array for the p-values
    p_values = np.zeros([nlats, nlons])

    # Take the mean over the 'number' dimension
    mean_ds = combined_ds.mean(dim="number")

    # convert to a cube
    cube_psl = mean_ds.to_iris()

    # # print the cube psl
    # print(cube_psl)

    # # print the lats and the lons
    # print(cube_psl.coord("latitude").points)
    # print(cube_psl.coord("longitude").points)

    # subset to region of interest
    cube_psl = cube_psl.intersection(longitude=(-180, 180), latitude=(-90, 90))

    # subset to the actual region of interest
    cube_psl = cube_psl.intersection(
        longitude=(lon_bounds[0], lon_bounds[1]),
        latitude=(lat_bounds[0], lat_bounds[1]),
    )

    # if calc anoms is true
    if calc_anoms:

        # set up a save directory for the climatologies
        save_dir_clim = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_clim"

        # assert that this directory exists
        assert os.path.exists(save_dir_clim), "The save directory does not exist"

        # set up the fname
        fname = f"climatology_{model}_{experiment}_{freq}_{psl_variable}_{climatology_period[0]}-{climatology_period[1]}_{lat_bounds[0]}-{lat_bounds[1]}_{lon_bounds[0]}-{lon_bounds[1]}_{months[0]}-{months[-1]}.nc"

        # set up the full climatology path
        climatology_path = os.path.join(save_dir_clim, fname)

        # if the climatology path exists
        if os.path.exists(climatology_path):
            print("The climatology file exists")
            print("Loading the climatology file")

            # load the file using iris
            cube_clim = iris.load_cube(climatology_path)

        else:
            print("The climatology file does not exist")
            print("Calculating the climatology")

            # Set up a list for the full ds's
            clim_dss = []

            # Loop over the years
            for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
                member_list = []
                for member in unique_members:
                    start_time = time.time()

                    path = f"{model_path_psl}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                    # glob this path
                    glob_start = time.time()
                    files = glob.glob(path)
                    glob_end = time.time()

                    # assert
                    assert (
                        len(files) == 1
                    ), f"files has length {len(files)} for year {year} and member {member} and path {path}"

                    # open all of the files
                    # open_start = time.time()
                    member_ds = xr.open_mfdataset(
                        files[0],
                        combine="nested",
                        concat_dim="time",
                        preprocess=lambda ds: preprocess(
                            ds=ds,
                            year=year,
                            variable=psl_variable,
                            months=months,
                        ),
                        parallel=True,
                        engine="netcdf4",
                        coords="minimal",  # expecting identical coords
                        data_vars="minimal",  # expecting identical vars
                        compat="override",  # speed up
                    ).squeeze()
                    # open_end = time.time()

                    # id init year == climatology_period[0]
                    # and member == unique_members[0]
                    # set_time_start = time.time()
                    if year == climatology_period[0] and member == unique_members[0]:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                            first_month_attr=True,
                        )
                    else:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                        )
                    # set_time_end = time.time()

                    # take the mean over the time axis
                    # mean_start = time.time()
                    member_ds = member_ds.mean(dim="time")
                    # mean_end = time.time()

                    # append the member_ds to the member_list
                    member_list.append(member_ds)

                    # end_time = time.time()

                    # # Print timing information
                    # print(f"Year: {year}, Member: {member}")
                    # print(f"  Total time: {end_time - start_time:.2f} seconds")
                    # print(f"  glob time: {glob_end - glob_start:.2f} seconds")
                    # print(f"  open_mfdataset time: {open_end - open_start:.2f} seconds")
                    # print(f"  set_integer_time_axis time: {set_time_end - set_time_start:.2f} seconds")
                    # print(f"  mean time: {mean_end - mean_start:.2f} seconds")

                # Concatenate with a new member dimension using xarray
                member_ds = xr.concat(member_list, dim="member")
                # append the member_ds to the init_year_list
                clim_dss.append(member_ds)
            # Concatenate the init_year list along the init dimension
            # and rename as lead time
            ds = xr.concat(clim_dss, "init")

            # print ds
            print(ds)

            # set up the members
            ds["member"] = unique_members
            ds["init"] = np.arange(climatology_period[0], climatology_period[1] + 1)

            # extract the variable
            ds_var = ds[psl_variable]

            # # take the mean over lead dimension
            # ds_clim = ds_var.mean(dim="lead")

            # take the mean over member dimension
            ds_clim = ds_var.mean(dim="member")

            # take the mean over init dimension
            ds_clim = ds_clim.mean(dim="init")

            # convert to a cube
            cube_clim = ds_clim.to_iris()

            # # regrid the model data to the obs grid
            # cube_clim_regrid = cube_clim.regrid(cube_obs, iris.analysis.Linear())

            # subset to the correct grid
            cube_clim = cube_clim.intersection(
                longitude=(-180, 180), latitude=(-90, 90)
            )

            # subset to the region of interest
            cube_clim = cube_clim.intersection(
                latitude=(lat_bounds[0], lat_bounds[1]),
                longitude=(lon_bounds[0], lon_bounds[1]),
            )

            # if the climatology file does not exist
            print("Saving the climatology file")

            # save the cube_clim
            iris.save(cube_clim, climatology_path)

    # # extract the lats and lons
    # lats = cube_psl.coord("latitude").points
    # lons = cube_psl.coord("longitude").points

    # if calc_anoms is True
    if calc_anoms:
        field_model = (combined_ds_arr - cube_clim.data) / 100  # convert to hPa

        field_model_full = (combined_ds_arr - cube_clim.data) / 100  # convert to hPa
    else:
        field_model = cube_psl.data / 100

        field_model_full = cube_psl.data / 100

    # print the shape of field_model_full
    print(f"The shape of field_model_full is {field_model_full.shape}")

    # print the shape of the field obs full
    print(f"The shape of the field obs full is {field_obs_full.shape}")

    # print the values of the field model full
    print(f"The values of the field model full are {field_model_full}")

    # priont the values of the field obs full
    print(f"The values of the field obs full are {field_obs_full}")

    # Loop over the bootstraps
    for iboot in tqdm(range(nboot)):
        # Create an array of randomly selected rows with length num_obs_events
        random_rows = np.random.choice(nrows, num_obs_events, replace=True)

        # Extract the data for these rows
        model_boot[iboot, :, :, :] = field_model_full[random_rows, :, :]

    # take the bootstrapped mean
    model_boot_mean = np.mean(model_boot, axis=0)

    # # print the shapes of the arrays
    # print(f"The shape of the model_boot is {model_boot.shape}")
    # print(f"The shape of the model_boot_mean is {model_boot_mean.shape}")

    # # print the values of model boot mean
    # print(f"The values of model_boot_mean are {model_boot_mean}")

    # # print the values of ds_compsoite_full_regrid.data
    # print(f"The values of ds_composite_full_regrid.data are {ds_composite_full_regrid.data}")

    # # print the shape of the ds_composite_full.data
    # print(
    #     f"The shape of the ds_composite_full.data is {ds_composite_full_regrid.data.shape}"
    # )

    # print the shape of field obs full
    print(f"The shape of field obs full is {field_obs_full.shape}")

    # print the shape of model boot mean
    print(f"The shape of model boot mean is {model_boot_mean.shape}")

    # Calculate the p-values
    _, p_values = stats.ttest_ind(field_obs_full, model_boot_mean, axis=0)

    # print the p-values
    print(f"The p-values are {p_values}")

    # print the shape of the p-values
    print(f"The shape of the p-values is {p_values.shape}")

    # print the values of field model
    print(f"The values of field model are {field_model}")

    # print the values of field obs
    print(f"The values of field obs are {field_obs}")

    # Set up the lons
    lons = cube_psl.coord("longitude").points
    lats = cube_psl.coord("latitude").points

    # print the shape of field model
    print(f"The shape of field model is {field_model.shape}")

    # print the shape of field obs
    print(f"The shape of field obs is {field_obs.shape}")

    # set up a figure as two subplots (one row, two columns)
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the observed basemap on the first axis
    mymap_obs = axs[0].contourf(
        lons,
        lats,
        field_obs,
        clevs,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend="both",
    )

    # PLot the observed contours on the first axis
    contours_obs = axs[0].contour(
        lons,
        lats,
        field_obs,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    # plot the model basemap on the second axis
    mymap_model = axs[1].contourf(
        lons,
        lats,
        np.mean(field_model, axis=0),
        clevs,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend="both",
    )

    # Plot the model contours on the second axis
    contours_model = axs[1].contour(
        lons,
        lats,
        np.mean(field_model, axis=0),
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    if calc_anoms:
        # Set the labels for the contours
        axs[0].clabel(
            contours_obs, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

        # Set the labels for the contours
        axs[1].clabel(
            contours_model,
            clevs,
            fmt="%.4g",
            fontsize=8,
            inline=True,
            inline_spacing=0.0,
        )

    else:
        # Set the labels for the contours
        axs[0].clabel(
            contours_obs, clevs, fmt="%d", fontsize=8, inline=True, inline_spacing=0.0
        )

        # Set the labels for the contours
        axs[1].clabel(
            contours_model, clevs, fmt="%d", fontsize=8, inline=True, inline_spacing=0.0
        )

    num_events = [
        num_obs_events,
        num_model_events,
    ]

    iterations = [0, 1]

    # Add coastlines
    for ax, num_event, i in zip(axs, num_events, iterations):
        ax.coastlines()

        # format the gridlines and labels
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
        )
        gl.xlabels_top = False
        gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {"size": 7, "color": "black"}
        gl.ylabels_right = False
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {"size": 7, "color": "black"}

        # get rid of te top labels
        gl.top_labels = False

        # get rid of the right labels
        gl.right_labels = False

        # if the iteration is 1
        if i == 1:
            gl.left_labels = False

        # include a textbox in the top left
        ax.text(
            0.02,
            0.95,
            f"N = {num_event}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="black",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
            alpha=0.8,
        )

    # Print the range of the p-values
    print(f"The range of the p-values is {np.min(p_values)} to {np.max(p_values)}")

    # print the shape of the p-values
    print(f"The shape of the p-values is {p_values.shape}")

    # # prnt the shape of mymap
    # print(f"The shape of mymap is {mymap_obs.shape}")

    # print the shape of the obsfield
    print(f"The shape of the obsfield is {field_obs.shape}")

    # mask the pfield to be NaN where the p-value is greater than 0.05
    pfield_masked = np.ma.masked_where(p_values > 0.05, p_values)

    # print the pfield masked
    print(f"The pfield masked is {pfield_masked}")

    # plot this on the first axis
    axs[1].contourf(
        lons,
        lats,
        pfield_masked,
        transform=ccrs.PlateCarree(),
        hatches=["..."],
        alpha=0.0,
    )

    if calc_anoms:
        cbar = fig.colorbar(
            mymap_obs,
            ax=axs,
            orientation="horizontal",
            shrink=0.9,
            pad=0.08,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours_obs)
    else:
        # add colorbar
        cbar = fig.colorbar(
            mymap_obs,
            ax=axs,
            orientation="horizontal",
            shrink=0.9,
            pad=0.08,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours_obs)

    # set up the titles
    axs[0].set_title(
        f"Observed {percentile}th percentile of {variable} events (ERA5)",
        fontsize=8,
        fontweight="bold",
    )

    # set up the titles
    axs[1].set_title(
        f"Model {percentile}th percentile of {variable} events ({model})",
        fontsize=8,
        fontweight="bold",
    )

    # Set up the tickparams
    cbar.ax.tick_params(labelsize=7, length=0)
    cbar.set_ticks(ticks)

    # # Make the plot look nice
    # plt.tight_layout()

    # Set up the current date time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    return


# Function for calculating autocorrelation of the obs data
# Octobers with Novembers etc.
def calc_autocorr_obs(
    obs_df: pd.DataFrame,
    obs_val_name: str,
    months: List[int],
    obs_time_name: str = "time",
    fname_prefix: str = "autocorr_obs",
    save_dir: str = "/home/users/benhutch/unseen_multi_year/dfs",
) -> None:
    """

    Calculates the autocorrelation in the observed time series by
    splitting the dataframe into month subsets and calculating the
    correlation between the subsets.

    Parameters
    ----------

    obs_df : pd.DataFrame
        The dataframe containing the observed data

    obs_val_name : str
        The name of the column containing the observed data

    months : List[int]
        The months to calculate the autocorrelation for

    obs_time_name : str
        The name of the column containing the observed time data

    fname_prefix : str
        The prefix to use for the filename

    save_dir : str
        The directory to save the output to

    Returns
    -------

    None

    """

    # Set up an empty list for the list of monthly subsets
    monthly_subsets = []

    # Loop over the months
    for month in months:
        # Subset the data to the month
        obs_df_month = obs_df[obs_df[obs_time_name].dt.month == month]

        # Extract the values as an array
        obs_vals = obs_df_month[obs_val_name].values

        # Append the values to the list
        monthly_subsets.append(obs_vals)

    # Set up a dataframe with len(months) columns each containing the monthly subset
    df = pd.DataFrame(monthly_subsets).T

    # name the columns
    df.columns = months

    # Compute the correlations
    corrs = df.corr()

    # print the correlations
    print(corrs)

    # Set up the current date time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the correlations
    corrs.to_csv(os.path.join(save_dir, f"{fname_prefix}_{date}.csv"))

    return


# Define a function for plotting the chance of event
# Where event is the chance of one month within a given winter
# exceeding it's most extreme conditions


def plot_chance_of_event_return_levels(
    obs_df: pd.DataFrame,
    model_df_ondjfm: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    months: List[int],
    num_samples: int = 1000,
    save_prefix: str = "chance_of_event_return_levels",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """

    Plots the chance of an event being worse than a specific year using return levels. Based on Figure 4 from Thompson et al., 2017 (doi: 10.1038/s41467-017-00275-3).

    Parameters
    ==========

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the
        observation value and the observation time.

    model_df_ondjfm: pd.DataFrame
        The DataFrame containing the model data with columns for the
        model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    months: List[int]
        The months to use for the event. For example, [10, 11, 12, 1, 2] for ONDJF.

    num_samples: int
        The number of samples to use for bootstrapping. Default is 1000.

    save_prefix: str
        The prefix to use when saving the plots. Default is "chance_of_event_return_levels".

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots".

    Returns
    =======

    None

    """

    fig, ax = plt.subplots(figsize=(6, 6))

    probs = 1 / np.arange(1.1, 1000, 0.1) * 100
    years = np.arange(1.1, 1000, 0.1)
    lead_years = np.arange(1, 11)
    months = [10, 11, 12, 1, 2, 3]
    n_samples = num_samples

    model_df_perc_change = pd.DataFrame()

    for m, month in enumerate(months):
        obs_month = obs_df[obs_df["time"].dt.month == month]
        obs_month_min = np.min(obs_month[obs_val_name])
        leads_this_month = [(ly * 12) + m for ly in lead_years]
        model_month = model_df_ondjfm[model_df_ondjfm["lead"].isin(leads_this_month)]
        model_month[f"{model_val_name}_perc_change"] = (
            (model_month[model_val_name] - obs_month_min) / obs_month_min * 100
        )
        model_df_perc_change = pd.concat([model_df_perc_change, model_month])

    print(model_df_perc_change[f"{model_val_name}_perc_change"].min())
    print(model_df_perc_change[f"{model_val_name}_perc_change"].max())

    model_df_rl = empirical_return_level(
        model_df_perc_change[f"{model_val_name}_perc_change"].values
    )
    print(model_df_rl.shape)
    print(model_df_rl.head())
    print(model_df_perc_change.shape)
    print(model_df_perc_change.head())

    model_vals = np.zeros([n_samples, len(model_df_perc_change)])
    params_model = [gev.fit(model_df_perc_change[f"{model_val_name}_perc_change"])]

    for i in tqdm(range(n_samples)):
        model_vals_this = np.random.choice(
            model_df_perc_change[f"{model_val_name}_perc_change"],
            size=len(model_df_rl["sorted"]),
            replace=True,
        )
        model_df_rl_sample = empirical_return_level(model_vals_this)
        model_vals[i, :] = model_df_rl_sample["sorted"]

    levels_model = gev.ppf(1 / years, *params_model[0])
    levels_model = np.array(levels_model)

    plt.plot(levels_model, probs, "b-")
    plt.plot(model_df_rl["sorted"], model_df_rl["probability"], color="red")

    print(model_vals.shape)

    model_vals_025 = np.percentile(model_vals, 2.5, axis=0)
    model_vals_975 = np.percentile(model_vals, 97.5, axis=0)

    plt.fill_betweenx(
        model_df_rl["probability"],
        model_vals_025,
        model_vals_975,
        color="red",
        alpha=0.2,
    )

    # model_vals_mean = np.mean(model_vals, axis=0)

    ax.set_ylim(0, 20)
    plt.axhline(1, color="black", linestyle="--")
    x_points = np.array([20, 10, 5, 2, 1, 0.5, 0.2, 0.1])
    ax.set_xlim(1, -1)
    ax.set_yscale("log")
    ax.set_yticks(x_points)
    ax.set_ylim(0.1, 10)
    ax.set_ylabel("Chance of event (%)")
    ax.set_xlabel(f"% change relative to lowest observed value")

    # Set up the current datetime
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.png"),
        dpi=600,
        bbox_inches="tight",
    )

    return


def plot_monthly_boxplots(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    variable: str,
    months: list,
    month_names: list,
    lead_years: np.ndarray,
    season: str,
    first_year: int,
    last_year: int,
    country: str,
    model: str = "HadGEM3-GC31-MM",
    experiment: str = "dcppA-hindcast",
    freq: str = "day",
    figsize: tuple = (10, 5),
    save_prefix: str = "monthly_boxplots",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """Plots boxplots for the different months.

    Parameters
    ==========

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

    variable: str
        The variable name to be plotted.

    months: list
        List of months to be plotted.

    month_names: list
        List of month names for the x-axis.

    lead_years: np.ndarray
        Array of lead years.

    season: str
        The season to be plotted.

    first_year: int
        The first year of the data.

    last_year: int
        The last year of the data.

    country: str
        The country name for the title.

    model: str
        The model name for the title.

    experiment: str
        The experiment name for the title.

    freq: str
        The frequency of the data for the title.

    figsize: tuple
        The size of the figure.

    Returns
    =======

    None
    """

    # Set up the figure size
    plt.figure(figsize=figsize)

    # Set up the ylabel
    plt.ylabel(f"{variable}")

    # Assert that months are recognized
    assert months == [10, 11, 12, 1, 2, 3], "Months not recognized"

    # Loop over the months
    for i, month in enumerate(months):
        # Subset to the month
        obs_df_month = obs_df[obs_df["time"].dt.month == month]

        leads_this_month = []

        # Find the leads to extract
        for j, ly in enumerate(lead_years):
            # Set up the leads
            leads_this_month = np.arange(
                331 + (j * 360) + (i * 30), 331 + 30 + (j * 360) + (i * 30)
            )

        # Subset to the leads
        model_df_month = model_df[model_df["lead"].isin(leads_this_month)]

        # Calculate the upper quartile (75th percentile)
        obs_lower_quartile = np.percentile(obs_df_month[obs_val_name], 25)
        obs_upper_quartile = np.percentile(obs_df_month[obs_val_name], 75)
        model_lower_quartile = np.percentile(model_df_month[model_val_name], 25)

        # Calculate the obs min value for the month
        obs_min = np.min(obs_df_month[obs_val_name])
        obs_max = np.max(obs_df_month[obs_val_name])

        # Plot the observed data in black
        obs_box = plt.boxplot(
            obs_df_month[obs_val_name],
            positions=[i + 1],
            widths=0.3,
            showfliers=False,
            boxprops=dict(color="black"),
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            flierprops=dict(markerfacecolor="black", marker="o"),
            medianprops=dict(color="black"),
            whis=[0, 100],  # the 0th and 100th percentiles (i.e. min and max)
            patch_artist=True,
        )

        # Set the face color for the observed data box
        for box in obs_box["boxes"]:
            box.set(facecolor="grey")

        # Plot the model data in red
        model_box = plt.boxplot(
            model_df_month[model_val_name],
            positions=[i + 1.5],
            widths=0.3,
            showfliers=False,
            boxprops=dict(color="black"),
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            flierprops=dict(markerfacecolor="black", marker="o"),
            medianprops=dict(color="black"),
            whis=[0, 100],  # the 0th and 100th percentiles (i.e. min and max)
            patch_artist=True,
        )

        # Set the face color for the model data box
        for box in model_box["boxes"]:
            box.set(facecolor="salmon")

        # add scatter points for obs values beneath the lower quartile
        obs_below_lower_quartile = obs_df_month[obs_val_name][
            obs_df_month[obs_val_name] < obs_lower_quartile
        ]
        obs_above_upper_quartile = obs_df_month[obs_val_name][
            obs_df_month[obs_val_name] > obs_upper_quartile
        ]

        plt.scatter(
            [i + 1] * len(obs_above_upper_quartile),
            obs_above_upper_quartile,
            color="black",
            marker="_",
            s=15,
            zorder=10,
        )

        # # add scatter points for model values beneath the lower quartile
        # model_below_lower_quartile = model_df_month[model_val_name][model_df_month[model_val_name] < model_lower_quartile]
        # plt.scatter(
        #     [i + 1.5] * len(model_below_lower_quartile),
        #     model_below_lower_quartile,
        #     color="red",
        #     marker="_",
        #     s=20,
        # )

        # add red dots for the points which are lower than the obs min
        model_below_obs_min = model_df_month[model_val_name][
            model_df_month[model_val_name] < obs_min
        ]
        model_above_obs_max = model_df_month[model_val_name][
            model_df_month[model_val_name] > obs_max
        ]

        # plot the model data
        plt.scatter(
            [i + 1.5] * len(model_above_obs_max),
            model_above_obs_max,
            color="red",
            edgecolor="black",
            marker="o",
            s=15,
            zorder=10,
        )

    # include gridlines
    plt.grid(axis="y")

    # set the xticks
    plt.xticks(ticks=np.arange(1, 7), labels=month_names)

    # set the title
    plt.title(
        f"Boxplots of {variable} for {country} {season} {first_year}-{last_year} HadGEM3-GC31-MM {experiment} {freq}"
    )

    # Set up the current datetime
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")

    try:
        # Save the plot
        plt.savefig(
            os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
            dpi=600,
            bbox_inches="tight",
        )
    except Exception as e:
        print(f"Error saving plot: {e}")

    return


# Define a function to plot the return period of extremes
def plot_rp_extremes(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    model_time_name: str,
    ylim: tuple = (0, 120),
    percentile: int = 99,
    months: list = [10, 11, 12, 1, 2, 3],
    lead_years: np.ndarray = np.arange(1, 11),
    n_samples: int = 1000,
    years_period: tuple = (1960, 2028),
    high_values_rare: bool = True,
    save_prefix: str = "rp_extremes",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the return period of extremes.

    Parameters
    ==========

    obs_df: pd.DataFrame
        The DataFrame containing the observations with columns for the

    model_df: pd.DataFrame
        The DataFrame containing the model data with columns for the model value and the model time.

    obs_val_name: str
        The name of the observation value column.

    model_val_name: str
        The name of the model value column.

    obs_time_name: str
        The name of the observation time column.

    model_time_name: str
        The name of the model time column.

    ylim: tuple
        The y-axis limits. Default is (0, 120).

    percentile: int
        The percentile to use for the return period calculation. Default is 99.

    months: list
        List of months to be plotted.

    lead_years: np.ndarray
        Array of lead years.

    n_samples: int
        The number of samples to use for bootstrapping. Default is 1000.

    years_period: tuple
        The period of years to use for the return period calculation. Default is (1960, 2028).

    high_values_rare: bool
        Whether high values are rare. Default is True.

    save_prefix: str
        The prefix to use when saving the plots. Default is "rp_extremes".

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots".

    Returns
    =======

    None

    """

    # if the time column is not datetime
    if not isinstance(obs_df[obs_time_name].values[0], np.datetime64):
        obs_df[obs_time_name] = pd.to_datetime(obs_df[obs_time_name])

    # if the time column is not datetime
    if not isinstance(model_df[model_time_name].values[0], np.datetime64):
        model_df[model_time_name] = pd.to_datetime(model_df[model_time_name])

    # Subset the data to the years period
    obs_df_subset = obs_df[
        (obs_df[obs_time_name].dt.year >= years_period[0])
        & (obs_df[obs_time_name].dt.year <= years_period[1])
    ]

    # Subset the data to the years period
    model_df_subset = model_df[
        (model_df[model_time_name].dt.year >= years_period[0])
        & (model_df[model_time_name].dt.year <= years_period[1])
    ]

    # Print the head of the obs_df
    print(obs_df_subset.head())
    print(obs_df_subset.tail())

    # Print the head of the model_df
    print(model_df_subset.head())
    print(model_df_subset.tail())

    # Set up the probabilities and years
    probs = 1 / np.arange(1.1, 1000, 0.1) * 100
    years = np.arange(1.1, 1000, 0.1)

    # Quantify the empirical return levels
    model_df_rl = empirical_return_level(
        data=model_df_subset[model_val_name].values,
        high_values_rare=high_values_rare,
    )

    # reverse the order of the rows
    # FIXME: May not be correct for low values rare
    model_df_rl_inverse = model_df_rl.iloc[::-1]

    # Create an array to store the return levels
    model_rl = np.zeros([n_samples, len(model_df_rl)])
    obs_rl = np.zeros([n_samples, len(obs_df_subset)])

    # Set up the model params
    model_params = []
    obs_params = []
    model_params_first = []

    model_params_first.append(
        gev.fit(
            model_df_subset[model_val_name].values,
        )
    )

    # Loop over the no. samples
    for i in tqdm(range(n_samples)):
        # Sample the model data
        model_vals_this = np.random.choice(
            model_df_subset[model_val_name].values,
            size=len(model_df_rl["sorted"]),
            replace=True,
        )

        # set up the obs vals this
        obs_vals_this = np.random.choice(
            obs_df_subset[obs_val_name].values,
            size=len(obs_df_subset),
            replace=True,
        )

        # Quantify the empirical return levels
        model_df_rl_this = empirical_return_level(
            data=model_vals_this,
            high_values_rare=True,
        )

        # Quantify the return levels using the gev
        model_params.append(
            gev.fit(
                model_vals_this,
            )
        )

        # Set up the obs return levels
        obs_params.append(
            gev.fit(
                obs_vals_this,
            )
        )

        # Store the model return levels
        model_rl[i, :] = model_df_rl_this["sorted"]

    levels_model = []
    levels_obs = []

    # loop over the num_samples
    for i in range(n_samples):
        # Generate the ppf fit
        levels_model.append(
            np.array(
                gev.ppf(
                    1 - 1 / years,
                    *model_params[i],
                )
            )
        )

        # Generate the ppf fit
        levels_obs.append(
            np.array(
                gev.ppf(
                    1 - 1 / years,
                    *obs_params[i],
                )
            )
        )

    # # Generate the ppf fit
    levels_model_first = np.array(
        gev.ppf(
            1 - 1 / years,
            *model_params_first[0],
        )
    )

    # Convert probs to the return level in years
    return_years = 1 / (probs / 100)

    # Convert model params to an array
    model_params = np.array(model_params)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # plot the observed return levels
    _ = ax.fill_between(
        return_years,
        np.quantile(levels_obs, 0.025, axis=0).T,
        np.quantile(levels_obs, 0.975, axis=0).T,
        color="gray",
        alpha=0.5,
        label="ERA5",
    )

    # plot the model return levels
    _ = ax.fill_between(
        return_years,
        np.quantile(levels_model, 0.025, axis=0).T,
        np.quantile(levels_model, 0.975, axis=0).T,
        color="red",
        alpha=0.5,
        label="HadGEM3-GC31-MM",
    )

    # Set up a logarithmic x-axis
    ax.set_xscale("log")

    # Limit to between 10 and 1000 years
    ax.set_xlim(10, 1000)

    # Set the xticks at 10, 20, 50, 100, 200, 500, 1000
    plt.xticks(
        [10, 20, 50, 100, 200, 500, 1000],
        ["10", "20", "50", "100", "200", "500", "1000"],
    )

    # Set the ylim
    ax.set_ylim(ylim)

    # Set the ylabel
    ax.set_ylabel("No. exceedance days", fontsize=12)

    # set the xlabel
    ax.set_xlabel("Return period (years)", fontsize=12)

    # include the value of the worst obs event with a horizontal line
    # ax.axhline(
    #     np.max(obs_df[obs_val_name]),
    #     color="blue",
    #     linestyle="-",
    # )

    # include the value of the 90th percentile of the obs
    # with a horizontal line
    ax.axhline(
        np.percentile(obs_df[obs_val_name], percentile),
        color="blue",
        linestyle="-",
    )

    # Include text on this line for the value
    # ax.text(
    #     11,
    #     np.max(obs_df[obs_val_name]) + 2,
    #     f"{round(np.max(obs_df[obs_val_name]))} days",
    #     color="blue",
    #     fontsize=12,
    # )

    # Include text on this line for the value
    ax.text(
        11,
        np.percentile(obs_df[obs_val_name], percentile) + 2,
        f"{round(np.percentile(obs_df[obs_val_name], percentile))} days",
        color="blue",
        fontsize=12,
    )

    # Set up the obs event
    bad_obs_event = np.percentile(obs_df[obs_val_name], percentile)
    worst_obs_event = np.max(obs_df[obs_val_name])

    # Quantify the return level for the worst obs event
    model_est_worst_obs = estimate_period(
        return_level=bad_obs_event,
        loc=model_params_first[0][1],
        scale=model_params_first[0][2],
        shape=model_params_first[0][0],
    )

    # Same but for the 2.5th percentile
    model_est_worst_obs_025 = estimate_period(
        return_level=bad_obs_event,
        loc=np.percentile(model_params[:, 1], 2.5),
        scale=np.percentile(model_params[:, 2], 2.5),
        shape=np.percentile(model_params[:, 0], 2.5),
    )

    # Same but for the 97.5th percentile
    model_est_worst_obs_975 = estimate_period(
        return_level=bad_obs_event,
        loc=np.percentile(model_params[:, 1], 97.5),
        scale=np.percentile(model_params[:, 2], 97.5),
        shape=np.percentile(model_params[:, 0], 97.5),
    )

    # print these values
    print(f"Model estimate for obs {percentile}th %tile event: {model_est_worst_obs}")
    print(
        f"Model estimate for obs {percentile}th %tile event 2.5th percentile: {model_est_worst_obs_025}"
    )
    print(
        f"Model estimate for obs {percentile}th %tile event 97.5th percentile: {model_est_worst_obs_975}"
    )

    # process into estiates
    worst_event = 1 - (model_est_worst_obs / 100)
    worst_event_025 = 1 - (model_est_worst_obs_025 / 100)
    worst_event_975 = 1 - (model_est_worst_obs_975 / 100)

    # Calculate the return period
    rp_worst_event = 1 / worst_event
    rp_worst_event_025 = 1 / worst_event_025
    rp_worst_event_975 = 1 / worst_event_975

    # print these values
    print(f"Return period for obs {percentile}th %tile event: {rp_worst_event}")
    print(
        f"Return period for obs {percentile}th %tile event 2.5th percentile: {rp_worst_event_025}"
    )
    print(
        f"Return period for obs {percentile}th %tile event 97.5th percentile: {rp_worst_event_975}"
    )

    central_95 = abs(rp_worst_event_975 - rp_worst_event_025) / 2

    # include a textbox in the top right with the return period of the worst observed event
    ax.text(
        0.95,
        0.02,
        f"Obs {percentile}th %tile RP: {round(rp_worst_event)} +/- {round(central_95)} years",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=10,
        # bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    # include a legend in the top left
    ax.legend(fontsize=10, loc="upper left")

    # Set a title with the years
    ax.set_title(
        f"RP of extreme {obs_val_name} events {years_period[0]}-{years_period[1]}",
        fontsize=10,
    )

    # Set up the current datetime
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{save_prefix}_{date}.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    return


# Set up the sigmoid fit
def sigmoid(x, L, x0, k, b):
    """
    Computes the sigmoid function.

    Parameters:
    x (float or array-like): The input value(s) for which to compute the sigmoid function.
    L (float): The curve's maximum value.
    x0 (float): The x-value of the sigmoid's midpoint.
    k (float): The steepness of the curve.
    b (float): The value to shift the curve vertically.

    Returns:
    float or array-like: The computed sigmoid value(s).
    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


# Set up a function for the dot plot
def dot_plot(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    model_time_name: str,
    ylabel: str,
    obs_label: str = "Observed",
    model_label: str = "Modelled",
    very_bad_label: str = "unseen events",
    bad_label: str = "extreme events",
    normal_label: str = "events",
    ylims: tuple = (0, 120),
    dashed_quant: float = 0.8,
    solid_line: Callable[[np.ndarray], float] = np.max,
    figsize: tuple = (10, 5),
    save_prefix: str = "dot_plot",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
) -> None:
    """
    Plots the dotplot for e.g. no. exceedance days.

    Parameters
    ==========

    obs_df: pd.DataFrame
        The DataFrame for the obs

    model_df: pd.DataFrame
        The DataFrame for the model

    obs_val_name: str
        The name of the obs value column

    model_val_name: str
        The name of the model value column

    model_time_name: str
        The name of the model time column

    ylabel: str
        The y-axis label for the figure

    obs_label: str
        The label for the obs data. Default is "Observed".

    model_label: str
        The label for the model data. Default is "Modelled".

    very_bad_label: str
        The label for the very bad events. Default is "unseen events".

    bad_label: str
        The label for the bad events. Default is "extreme events".

    normal_label: str
        The label for the normal events. Default is "events".

    ylims: tuple
        The y-axis limits. Default is (0, 120).

    dashed_quant: float
        The quantile to use for the dashed line. Default is 0.8.

    solid_line: Callable[[np.ndarray], float]
        The function to use for the solid line. Default is np.max.

    figsize: tuple
        The figure size. Default is (10, 5).

    save_prefix: str
        The prefix to use when saving the plots. Default is "dot_plot".

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots".

    Returns
    =======

    None

    """

    # Assert that the index of the obs df is a datetime in years
    assert isinstance(obs_df.index, pd.DatetimeIndex), "Index  of obs must be a datetime"

    # Set up the figure
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [8, 1]},
    )

    # add a horizontal line for the 0.8 quantile of the observations
    axs[0].axhline(
        np.quantile(obs_df[obs_val_name], dashed_quant),
        color="blue",
        linestyle="--",
    )

    # for the max value of the obs
    axs[0].axhline(
        solid_line(obs_df[obs_val_name]),
        color="blue",
        linestyle="-.",
    )

    # plot the scatter points for the obs
    axs[0].scatter(
        obs_df.index,
        obs_df[obs_val_name],
        color="blue",
        marker="x",
        label=obs_label,
        zorder=2,
    )

    # if solid line is np.max and dahsed line is above 0.5
    if solid_line == np.max and dashed_quant > 0.5:
        print("Bad events have high values")

        # Separate model data by threshold
        very_bad_events = df_model_exceedance_dt[
            model_df[model_val_name] > solid_line(obs_df[obs_val_name])
        ]

        # Model data above 80th percentile
        bad_events = df_model_exceedance_dt[
            (model_df[model_val_name] > np.quantile(obs_df[obs_val_name], dashed_quant))
            & (model_df[model_val_name] < solid_line(obs_df[obs_val_name]))
        ]

        # Model data below 80th percentile
        events = df_model_exceedance_dt[
            model_df[model_val_name] < np.quantile(obs_df[obs_val_name], dashed_quant)
        ]

    else:
        print("Bad events have low values")

        # assert that solid_line is np.min
        assert solid_line == np.min, "Solid line must be np.min"

        # assert that dashed_quant is below 0.5
        assert dashed_quant < 0.5, "Dashed quantile must be below 0.5"

        # Separate model data by threshold
        very_bad_events = df_model_exceedance_dt[
            model_df[model_val_name] < solid_line(obs_df[obs_val_name])
        ]

        # Model data above 80th percentile
        bad_events = df_model_exceedance_dt[
            (model_df[model_val_name] < np.quantile(obs_df[obs_val_name], dashed_quant))
            & (model_df[model_val_name] > solid_line(obs_df[obs_val_name]))
        ]

        # Model data below 80th percentile
        events = df_model_exceedance_dt[
            model_df[model_val_name] > np.quantile(obs_df[obs_val_name], dashed_quant)
        ]

    # Plot the points below the minimum of the obs
    axs[0].scatter(
        very_bad_events[model_time_name]
        very_bad_events[model_val_name]
        color="red",
        alpha=0.8,
        label=very_bad_label,
    )

    # Plot the points below the 20th percentile
    axs[0].scatter(
        bad_events[model_time_name],
        bad_events[model_val_name],
        color="orange",
        alpha=0.8,
        label=bad_label,
    )

    # Plot the points above the 20th percentile
    axs[0].scatter(
        events[model_time_name],
        events[model_val_name],
        color="grey",
        alpha=0.8,
        label=normal_label,
    )

    # include the legend
    axs[0].legend(fontsize=12)

    # label the y-axis
    axs[0].set_ylabel(ylabel, fontsize=14)

    # set up the x-axis

    # increase the size of the value labels
    axs[0].tick_params(axis="x", labelsize=12)

    # same for the y-axis
    axs[0].tick_params(axis="y", labelsize=12)

    # set up the ylims
    axs[0].set_ylim(ylims)

    # do the events plots for the no. exceedance days on the second plot
    axs[1].boxplot(
        obs_df[obs_val_name],
        colors="black",
        lineoffsets=0,
        linelengths=0.5,
        orientation="vertical",
        linewidths=1,
        label="obs",
    )

    # plot the model data
    axs[1].boxplot(
        model_df[model_val_name],
        colors="red",
        lineoffsets=1,
        linelengths=0.5,
        orientation="vertical",
        linewidths=0.5,
        label="model",
    )

    # # set up the xlabels for the second plot
    # xlabels = ["Observed", "Model"]

    # # add the xlabels to the second subplot
    # axs[1].set_xticks(xlabels)

    # # remove the xlabel
    # axs[1].set_xlabel("")

    # # remove the yticks and axis lin
    # axs[1].set_yticks([])

    # remove the xticks from the second plot
    axs[1].set_xticks([])

    # specify a tight layout
    plt.tight_layout()

    # set up a fname for the plot
    fname = f"obs-{obs_val_name}_model-{model_val_name}_quantile-{dashed_quant}_solid-{solid_line.__name__}_{save_prefix}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pdf"

    # form the savepath
    savepath = os.path.join(save_dir, fname)

    if not os.path.exists(savepath):
        print(f"Saving plot to {savepath}")
        # save the plot
        plt.savefig(savepath, bbox_inches="tight", dpi=800)

        # print that we have saved the plot
        print(f"Saved plot to {savepath}")

    return
