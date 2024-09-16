# Functions for loading DePreSys data

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
import iris
import shapely.geometry
import cartopy.io.shapereader as shpreader
from datetime import datetime, timedelta
import xesmf as xe
import matplotlib.cm as cm
import cftime

# Import types
from typing import Any, Callable, Union, List
# from iris import Cube

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
    frequency: str = "Amon",
    engine: str = "netcdf4",
    parallel: bool = False,
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

    # SET UP THE PATHS TO NIAMHS CSV FILE
    # Set up the path to the csv file
    csv_path = None # FIXME: set the path to the csv file

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
        # print(f"processing init year {init_year}")
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