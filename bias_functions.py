"""
Functions for quantifying and adjusting this biases in the
dcppA-hindcast data.

Utilizing similar methods to Dawkins and Rushby 2021.
"""

import os
import sys
import glob
import time
import re

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import xesmf as xe
from tqdm import tqdm

# Import specifically from functions
from functions import set_integer_time_axis, regrid_ds, select_gridbox

# Import dictionaries
import unseen_dictionaries as dicts


# Define a function that will load the dcppA hindcast data
# For a given model, variable, and lead time
# E.g. for HadGEM3-GC31-MM, tas, and 1 year lead time
# As HadGEM3-GC31-MM is initialized in November, this would be
# The monthly data from the first November to the following October
def load_dcpp_data_lead(
    model: str,
    variable: str,
    lead_time: int,
    init_years: list[int],
    experiment: str = "dcppA-hindcast",
    frequency: str = "Amon",
    engine: str = "netcdf4",
    parallel: bool = False,
    csv_fpath: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
) -> xr.Dataset:
    """
    Load the dcppA hindcast data for a given model, variable, and lead time.

    Parameters
    ----------
    model : str
        The name of the model to load data for.
    variable : str
        The name of the variable to load data for.
    lead_time : int
        The lead time to load data for.
    init_years : list[int]
        The initialization years to load data for.
    experiment : str, optional
        The name of the experiment to load data for, by default 'dcppA-hindcast'.
    frequency : str, optional
        The frequency of the data to load, by default 'Amon'.
    engine : str, optional
        The engine to use when loading the data, by default 'netcdf4'.
    parallel : bool, optional
        Whether to load the data in parallel, by default False.
    csv_fpath : str, optional
        The path to the csv file containing the paths to the data, by default "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv".
    Returns
    -------
    xr.Dataset
        The data for the given model, variable, and lead time.
    """

    # Check that the csv file exists
    if not os.path.exists(csv_fpath):
        raise FileNotFoundError(f"Cannot find the file {csv_fpath}")

    # Load in the csv file
    csv_data = pd.read_csv(csv_fpath)

    # Extract the path for the given model, experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == variable)
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
            year_path = f"{model_path}/s{init_year}-r*i?p?f?/{frequency}/{variable}/g?/files/d????????/*.nc"

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
        re.split(r"_s....-", agg_files.split("/")[-1].split("_g")[0])[1]
        for agg_files in files
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
            variant_files = [
                file for file in agg_files if f"s{init_year}-{variant}" in file
            ]

            # Open all leads for the specified variant
            member_ds = xr.open_mfdataset(
                variant_files,
                chunks={"time": 10},
                combine="nested",
                concat_dim="time",
                preprocess=lambda ds: preprocess_lead(
                    ds=ds,
                    lead_time=lead_time,
                    frequency=frequency,
                ),
                parallel=parallel,
                engine=engine,
                coords="minimal",  # explicitly set coords to minimal
                data_vars="minimal",  # explicitly set data_vars to minimal
                compat="override",  # override the default behaviour
            ).squeeze()  # remove any dimensions of length 1

            # If this is the first year and ensemble member
            if init_year == init_years[0] and variant == variants[0]:
                # Set the time axis to be an integer
                member_ds = set_integer_time_axis(
                    xro=member_ds,
                    frequency=frequency,
                    first_month_attr=True,
                )
            else:
                # Set the time axis to be an integer
                member_ds = set_integer_time_axis(
                    xro=member_ds,
                    frequency=frequency,
                    first_month_attr=False,
                )
            # Append the member_ds to the member_list
            member_list.append(member_ds)
        # Concatenate the member_list along the member dimension
        member_ds = xr.concat(member_list, dim="member")
        # Append the member_ds to the init_year_list
        init_year_list.append(member_ds)
    # Concatenate the init_year_list along the init_year dimension
    ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

    # Set up the members
    ds["member"] = variants
    ds["init"] = np.arange(init_years[0], init_years[-1] + 1)

    # Return the data
    return ds


# Define a function to preprocess the data
def preprocess_lead(
    ds: xr.Dataset,
    lead_time: int,
    frequency: str,
):
    """
    Preprocess the data by constraining the data to the lead time and frequency
    specified. E.g. for HadGEM3-GC31-MM, lead time 1, frequency 'Amon', this
    would extract the first November to the following October (indices 0 to 11).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to preprocess.

    lead_time : int
        The lead time of the data to be extracted.

    frequency : str
        The frequency of the data to be extracted.

    Returns
    -------
    xr.Dataset
        The preprocessed dataset.
    """

    # if the frequency is 'Amon'
    if frequency == "Amon":
        # Set up the indices to extract
        indices = np.arange(0, lead_time * 12)

        # Extract the data
        ds = ds.isel(time=indices)
    elif frequency == "day":
        raise NotImplementedError("Daily data not yet implemented.")
    else:
        raise ValueError(f"Frequency {frequency} not recognised.")

    # Return the preprocessed dataset
    return ds


# Define a function to load the observed data
def load_and_rg_obs(
    model_ds: xr.Dataset,
    obs_variable: str,
    obs_path: str,
    init_years: list[int],
    lead_time: int,
    rg_algo: str = "bilinear",
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    periodic: bool = True,
    parallel: bool = False,
) -> xr.Dataset:
    """
    Load the dcppA hindcast data for a given model, variable, and lead time.

    Parameters
    ----------
    model_ds : xr.Dataset
        The dataset to regrid the obs to.
    obs_variable : str
        The name of the variable to load data for.
    obs_path : str
        The path to the observed data.
    init_years : list[int]
        The initialization years to load data for.
    lead_time : int
        The lead time to load data for.
    rg_algo : str, optional
        The regridding algorithm to use, by default 'bilinear'.
    grid_bounds : list[float], optional
        The grid bounds to use when regridding, by default [-180.0, 180.0, -90.0, 90.0].
    periodic : bool, optional
        Whether the grid is periodic, by default True.
    parallel : bool, optional
        Whether to load the data in parallel, by default False.

    Returns
    -------
    xr.Dataset
        The loaded observed data.
    """

    # Calculate the resolution of the input dataset
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

    # Load the observed data
    obs_ds = xr.open_mfdataset(
        obs_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=parallel,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in obs_ds.coords:
        # Combine the first two expver variables
        obs_ds = obs_ds.sel(expver=1).combine_first(obs_ds.sel(expver=5))

    # Set up the start and end years
    start_year = init_years[0]
    end_year = init_years[-1] + lead_time

    # Set up the first month
    # Which is the same month as the first month of the model data
    first_month = model_ds["time"].dt.month[0].values

    # Set up the last month
    last_month = first_month - 1  # e.g. for november (11), this would be october (10)

    # restrict to between the start and end years
    obs_ds = obs_ds.sel(
        time=slice(
            f"{start_year}-{first_month:02d}-01", f"{end_year}-{last_month:02d}-30"
        )
    )

    # Convert the lat and lon to 1D
    ds_out["lon"] = ds_out["lon"].mean(dim="y")
    ds_out["lat"] = ds_out["lat"].mean(dim="x")

    # Set up the regridder
    regridder = xe.Regridder(
        obs_ds,
        ds_out,
        rg_algo,
        periodic=periodic,
    )

    # Regrid the data
    obs_rg = regridder(obs_ds[obs_variable])

    # return the regridded data
    return obs_rg

# define a main function for testing
def main():
    # Start a timer
    start = time.time()

    # Define the model, variable, and lead time
    model = "HadGEM3-GC31-MM"
    variable = "tas"
    obs_variable = "t2m"
    lead_time = 1
    init_years = [1960]
    # init_years = np.arange(1960, 1970 + 1)
    experiment = "dcppA-hindcast"
    frequency = "Amon"
    engine = "netcdf4"
    parallel = False

    test_file = "/gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/tas/HadGEM3-GC31-MM/merged_files/tas_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_196011-197103.nc"

    obs_fpath = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

    # test the load data function
    ds = load_dcpp_data_lead(
        model=model,
        variable=variable,
        lead_time=lead_time,
        init_years=init_years,
        experiment=experiment,
        frequency=frequency,
        engine=engine,
        parallel=parallel,
    )

    # regrid the data
    ds = regrid_ds(
        ds=ds,
        variable=variable,
    )

    # Select the gridbox
    ds = select_gridbox(
        ds=ds,
        grid=dicts.eu_grid,
        calc_mean=False,
    )

    # Load the test ds
    test_ds = xr.open_dataset(test_file)

    # Test the load and regrid obs function
    obs = load_and_rg_obs(
        model_ds=test_ds,
        obs_variable=obs_variable,
        obs_path=obs_fpath,
        init_years=init_years,
        lead_time=lead_time,
        rg_algo="bilinear",
        grid_bounds=[-180.0, 180.0, -90.0, 90.0],
        periodic=True,
        parallel=False,
    )

    # select the gridbox for the obs
    obs = select_gridbox(
        ds=obs,
        grid=dicts.eu_grid,
        calc_mean=False,
    )

    # print the obs
    print(f"Obs: {obs}")

    # print the ds
    print(f"DS: {ds}")

    # End the timer
    end = time.time()

    # Print the time taken
    print(f"Time taken: {end - start:.2f} seconds.")

    # Print that we are exiting the main function
    print("Exiting main function.")
    sys.exit()


if __name__ == "__main__":
    main()
