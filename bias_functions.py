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
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xesmf as xe
from tqdm import tqdm
from dask.diagnostics import ProgressBar

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
        The initialization years to load data for.O
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
            files = glob.glob(os.path.join(model_path, f"*s{init_year}*"))

            # # print the len of the files
            # print(f"Number of files: {len(files)}")

            # Assert that there are files
            assert len(files) > 0, f"No files found for {init_year} in {model_path}"

            # # print the len of files for each year
            # print(f"Number of files: {len(files)} for {init_year}")

            # # if the length of the files is greater than 10
            # if len(files) > 10:
            #     # print the first 10 files
            #     print(f"files: {files}")

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
        if lead_time == 1:
            # Extract the indices
            indices = np.arange(0, 12)
        else:
            # Extract the indices
            indices = np.arange(((lead_time - 1) * 12), lead_time * 12)

        # print the indices
        print(f"indices: {indices}")

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

# Create a function for saving the data
def save_data(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    model: str,
    experiment: str,
    frequency: str,
    variable: str,
    init_years: list[int],
    lead_time: int,
    save_dir: str = "/work/scratch-nopw2/benhutch/test_nc/",
):
    """
    Save the model and observed data to netcdf files.
    
    Parameters
    ----------
    model_ds : xr.Dataset
        The model dataset to save.
    obs_ds : xr.Dataset
        The observed dataset to save.
    model : str
        The model to save.
    experiment : str
        The experiment to save.
    frequency : str
        The frequency to save.
    variable : str
        The variable to save.
    init_years : list[int]
        The initialization years to save.
    lead_time : int
        The lead time to save.
    save_dir : str, optional
        The directory to save the files to, by default "/work/scratch-nopw2/benhutch/test_nc/".
    
    Returns
    -------

    """

    # If the save directory does not exist
    if not os.path.exists(save_dir):
        # Make the directory
        os.makedirs(save_dir)

    # Set up the current date, hour and minute
    current_time = time.strftime("%Y%m%dT%H%M%S")

    # Set up the model file name
    model_fname = f"{model}_{experiment}_{variable}_s{init_years[0]}-{init_years[-1]}_lead{lead_time}_{frequency}_{current_time}.nc"

    # Set up the obs file name
    obs_fname = f"obs_{variable}_s{init_years[0]}-{init_years[-1]}_{frequency}_{current_time}.nc"

    # Set up the model file path
    model_fpath = os.path.join(save_dir, model_fname)

    # Set up the obs file path
    obs_fpath = os.path.join(save_dir, obs_fname)

    # Set up the delayed model object
    delayed_model = model_ds.to_netcdf(model_fpath, compute=False)

    # Set up the delayed obs object
    delayed_obs = obs_ds.to_netcdf(obs_fpath, compute=False)

    try:
        # Compute the delayed model object
        with ProgressBar():
            print("Saving model data.")
            results = delayed_model.compute()
    except:
        print("Model data not saved.")

    try:
        # Compute the delayed obs object
        with ProgressBar():
            print("Saving obs data.")
            results = delayed_obs.compute()
    except:
        print("Obs data not saved.")

    print(f"Model data saved to {model_fpath}")
    print(f"Obs data saved to {obs_fpath}")

    return None
        


# Define a function to calculate and plot the bias
# saving the output
def calc_and_plot_bias(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    month_idx: int,
    lead_time: int,
    init_years: list[int],
    variable: str,
    month_name: str,
    figsize: tuple = (12, 6),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
    save: bool = True,
):
    """
    Calculate and plot the bias between the model and observed data.
    for the mean and sigma. For a single variable and lead time.

    Parameters
    ----------

    model_ds : xr.Dataset
        The model dataset to calculate the bias for.
    obs_ds : xr.Dataset
        The observed dataset to calculate the bias for.
    month_idx : int
        The index of the month to calculate the bias for.
    lead_time : int
        The lead time to calculate the bias for.
    variable : str
        The variable to calculate the bias for.
    month_name : str
        The name of the month to calculate the bias for.
    figsize : tuple, optional
        The size of the figure to plot, by default (12, 6).
    save_dir : str, optional
        The directory to save the plots to, by default "/gws/nopw/j04/canari/users/benhutch/plots/".
    save : bool, optional
        Whether to save the plots, by default True.

    Returns
    -------

    """

    # Select the month idx from the model data
    assert month_idx != 0, "Month index cannot be 0."

    # Find the mean(dim=y) values of lon, which are not nan
    lons = model_ds["lon"]

    # print the lons
    print(f"Lon min: {lons.min().values}, Lon max: {lons.max().values}")

    # Find the mean(dim=x) values of lat, which are not nan
    lats = obs_ds["lat"]
    lons = obs_ds["lon"]

    # pritn the shape of lats
    print(f"Shape of lats: {np.shape(lats)}")
    print(f"Shape of lons: {np.shape(lons)}")

    # Select the month from the model data
    model_month = model_ds.sel(lead=month_idx)

    # Find the month at the correct month_idx in the obs
    obs_month = obs_ds.time.dt.month[month_idx - 1]

    # Select the month from the obs data
    obs_month_data = obs_ds.where(obs_ds.time.dt.month == obs_month, drop=True)

    # Calculate the bias as model_mean - obs_mean
    mean_bias = model_month.mean(
        dim=["init", "member"], skipna=True
    ) - obs_month_data.mean(dim="time")

    # Calculate the bias as model_sigma - obs_sigma
    sigma_bias = model_month.std(
        dim=["init", "member"], skipna=True
    ) - obs_month_data.std(dim="time")

    # Find the indices of x and y where the mean bias is not nan
    mean_bias_idcs = np.where(~np.isnan(mean_bias.values))

    # apply the first mean_bias_idcs to the lats
    mean_bias_lats = lats[mean_bias_idcs[0]]

    # apply the second mean_bias_idcs to the lons
    mean_bias_lons = lons[mean_bias_idcs[1]]

    lats_values = mean_bias_lats.values

    # extract only the unique values
    lats_values = np.unique(lats_values)

    # print the lons
    lons_values = mean_bias_lons.values

    # extract only the unique values
    lons_values = np.unique(lons_values)

    # drop the nan values
    mean_bias = mean_bias.dropna("y", how="all").dropna("x", how="all")

    # drop the nan values
    sigma_bias = sigma_bias.dropna("y", how="all").dropna("x", how="all")

    # add the y coordinates as the lats_values
    mean_bias["y"] = lats_values
    mean_bias["x"] = lons_values

    # add the y coordinates as the lats_values
    sigma_bias["y"] = lats_values
    sigma_bias["x"] = lons_values

    # Set up the figure
    fig, ax = plt.subplots(1, 2, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot the mean bias
    # ax[0].imshow(mean_bias.values, cmap="bwr", vmin=-10, vmax=10, transform=ccrs.PlateCarree(), interpolation='none')
    # ax[0].set_title('Mean Bias')

    # # Plot the sigma bias
    # ax[1].imshow(sigma_bias.values, cmap="bwr", vmin=-5, vmax=5, transform=ccrs.PlateCarree(), interpolation='none')
    # ax[1].set_title('Sigma Bias')

    # set up the contour levels
    # clevs = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
    clevs = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # set up the ticks
    ticks_mean = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

    # Plot the mean bias
    # Plot the mean bias
    contour_mean = mean_bias.plot.contourf(
        ax=ax[0],
        cmap="bwr",
        levels=clevs,
        add_colorbar=False,
        vmin=-10,
        vmax=10,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(contour_mean, ax=ax[0], ticks=ticks_mean)
    cbar.set_label('Mean Bias')

    # Set up the contour levels
    # clevs = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    clevs = np.array([-5, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Set the ticks manually
    ticks_std = np.array([-5, -4, -3, -2, 0, 2, 3, 4, 5])

    # Plot the sigma bias
    contour_std = sigma_bias.plot.contourf(
        ax=ax[1],
        cmap="bwr",
        levels=clevs,
        add_colorbar=False,
        vmin=-5,
        vmax=5,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(contour_std, ax=ax[1], ticks=ticks_std)
    cbar.set_label('Sigma Bias')
    
    # add coastlines
    ax[0].coastlines()

    # add coastlines
    ax[1].coastlines()

    # Set the title
    ax[0].set_title(
        f"mean bias"
    )

    # Set the title
    ax[1].set_title(
        f"sigma Bias"
    )

    # Set up the super title
    fig.suptitle(
        f"{variable} bias for {month_name} lead {lead_time} years {init_years[0]}-{init_years[-1]}"
    )

    # Set the xlabel
    ax[0].set_xlabel("Longitude")

    # Set the xlabel
    ax[1].set_xlabel("Longitude")

    # Set the ylabel
    ax[0].set_ylabel("Latitude")

    # Set up the current time
    current_time = time.strftime("%Y%m%dT%H%M%S")

    # Set up the fname
    fname = f"{variable}_bias_{month_name}_lead{lead_time}_init{init_years[0]}-{init_years[-1]}_{current_time}.pdf"

    # if the save_dir does not exist
    if not os.path.exists(save_dir):
        # make the directory
        os.makedirs(save_dir)

    # If save is True
    if save and not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving figure to {os.path.join(save_dir, fname)}")
        # Save the figure
        fig.savefig(os.path.join(save_dir, fname))

    # Show the plot
    plt.show()

    # return None
    return None

# Calculate and plot 6 rows x 2 columns for the mean or std for each lead month
def calc_and_plot_bias_all_months(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    lead_time: int,
    init_years: list[int],
    variable: str,
    month_names: list[str],
    mean_or_std: str = "mean",
    figsize: tuple = (12, 6),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
    save: bool = True,
):
    """
    Calculate and plot the bias between the model and observed data.
    for the mean or sigma. For a single variable and lead time.

    Parameters
    ----------

    model_ds : xr.Dataset
        The model dataset to calculate the bias for.
    obs_ds : xr.Dataset
        The observed dataset to calculate the bias for.
    lead_time : int
        The lead time to calculate the bias for.
    init_years : list[int]
        The initialization years to calculate the bias for.
    variable : str
        The variable to calculate the bias for.
    month_names : list[str]
        The names of the months to calculate the bias for.
    mean_or_std : str, optional
        Whether to calculate the mean or standard deviation, by default "mean".
        Only takes values of "mean" or "std".
    figsize : tuple, optional
        The size of the figure to plot, by default (12, 6).
    save_dir : str, optional
        The directory to save the plots to, by default "/gws/nopw/j04/canari/users/benhutch/plots/".
    save : bool, optional
        Whether to save the plots, by default True.

    Returns
    -------

    """

    # Extract the lats and lons from the obs
    lats = obs_ds["lat"]
    lons = obs_ds["lon"]

    # Set up the figure
    fig, ax = plt.subplots(6, 2, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})

    # Loop over the month names
    for i, month_name in enumerate(month_names):
        print(f"plotting {month_name} at index {i}")

        # Select the month idx from the model data
        model_ds_month = model_ds.sel(lead=i + 1)

        # Find the month at the correct month_idx in the obs
        obs_month = obs_ds.time.dt.month[i - 1]

        # Select the month from the obs data
        obs_month_data = obs_ds.where(obs_ds.time.dt.month == obs_month, drop=True)

        # if the mean_or_std is mean
        if mean_or_std == "mean":
            # Calculate the bias as model_mean - obs_mean
            bias = model_ds_month.mean(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.mean(dim="time")

            # Set up the contour levels
            clevs = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])

            # Set up the ticks
            ticks = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        elif mean_or_std == "std":
            # Calculate the bias as model_sigma - obs_sigma
            bias = model_ds_month.std(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.std(dim="time")

            # Set up the contour levels
            clevs = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

            # Set up the ticks
            ticks = np.array([-5, -4, -3, -2, 0, 2, 3, 4, 5])
        else:
            raise ValueError(f"mean_or_std {mean_or_std} not recognised.")
        
        # Find the indices of x and y where the mean bias is not nan
        bias_idcs = np.where(~np.isnan(bias.values))

        # apply the first bias_idcs to the lats
        bias_lats = lats[bias_idcs[0]]

        # apply the second bias_idcs to the lons
        bias_lons = lons[bias_idcs[1]]

        lats_values = bias_lats.values

        # extract only the unique values
        lats_values = np.unique(lats_values)

        # print the lons
        lons_values = bias_lons.values

        # extract only the unique values
        lons_values = np.unique(lons_values)

        # drop the nan values
        bias = bias.dropna("y", how="all").dropna("x", how="all")

        # add the y coordinates as the lats_values
        bias["y"] = lats_values

        # add the y coordinates as the lats_values
        bias["x"] = lons_values

        # Plot the bias
        contour = bias.plot.contourf(
            ax=ax[i // 2, i % 2],
            cmap="bwr",
            levels=clevs,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )

        cbar = plt.colorbar(contour, ax=ax[i // 2, i % 2], ticks=ticks, shrink=0.8)
        cbar.set_label(f"{mean_or_std.capitalize()} Bias")

        # add coastlines
        ax[i // 2, i % 2].coastlines()

        # Set the title
        ax[i // 2, i % 2].set_title(
            f"{month_name}"
        )

        # Set the xlabel
        ax[i // 2, i % 2].set_xlabel("Lon")

        # Set the ylabel
        ax[i // 2, i % 2].set_ylabel("Lat")

    # Set up the super title
    fig.suptitle(
        f"{variable} {mean_or_std.capitalize()} bias for lead {lead_time} years {init_years[0]}-{init_years[-1]}"
    )

    # Set up the current time
    current_time = time.strftime("%Y%m%dT%H%M%S")

    # Set up the fname
    fname = f"{variable}_{mean_or_std}_bias_lead{lead_time}_init{init_years[0]}-{init_years[-1]}_{current_time}.pdf"

    # if the save_dir does not exist
    if not os.path.exists(save_dir):
        # make the directory
        os.makedirs(save_dir)

    # If save is True
    if save and not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving figure to {os.path.join(save_dir, fname)}")
        # Save the figure
        fig.savefig(os.path.join(save_dir, fname))

    # Show the plot
    plt.show()

    # return None
    return None

# define a main function for testing
def main():
    # Start a timer
    start = time.time()

    # Set up the parser
    parser = argparse.ArgumentParser(description="Test the bias functions.")

    # Add the arguments
    parser.add_argument(
        "--model",
        type=str,
        help="The model to load data for.",
        default="HadGEM3-GC31-MM",
    )

    parser.add_argument(
        "--variable",
        type=str,
        help="The variable to load data for.",
        default="tas",
    )

    parser.add_argument(
        "--obs_variable",
        type=str,
        help="The observed variable to load data for.",
        default="t2m",
    )

    parser.add_argument(
        "--lead_time",
        type=int,
        help="The lead time to load data for.",
        default=1,
    )

    parser.add_argument(
        "--start_year",
        type=int,
        help="The start year to load data for.",
    )

    parser.add_argument(
        "--end_year",
        type=int,
        help="The end year to load data for.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        help="The experiment to load data for.",
        default="dcppA-hindcast",
    )

    parser.add_argument(
        "--frequency",
        type=str,
        help="The frequency to load data for.",
        default="Amon",
    )

    parser.add_argument(
        "--engine",
        type=str,
        help="The engine to use when loading the data.",
        default="netcdf4",
    )

    parser.add_argument(
        "--parallel",
        type=bool,
        help="Whether to load the data in parallel.",
        default=False,
    )

    # Set up the args
    args = parser.parse_args()

    # print the args
    print(f"Model: {args.model}")
    print(f"Variable: {args.variable}")
    print(f"Obs Variable: {args.obs_variable}")
    print(f"Lead Time: {args.lead_time}")
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")
    print(f"Experiment: {args.experiment}")
    print(f"Frequency: {args.frequency}")
    print(f"Engine: {args.engine}")
    print(f"Parallel: {args.parallel}")

    # Set up the init years
    init_years = np.arange(args.start_year, args.end_year + 1)

    # Set the variables
    model = args.model
    variable = args.variable
    obs_variable = args.obs_variable
    lead_time = args.lead_time
    experiment = args.experiment
    frequency = args.frequency
    engine = args.engine
    parallel = args.parallel

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

    # Save the data
    save_data(
        model_ds=ds,
        obs_ds=obs,
        model=model,
        experiment=experiment,
        frequency=frequency,
        variable=variable,
        init_years=init_years,
        lead_time=lead_time,
        save_dir="/work/scratch-nopw2/benhutch/test_nc/",
    )

    # # Test the plot bias function
    # calc_and_plot_bias(
    #     model_ds=ds,
    #     obs_ds=obs,
    #     month_idx=1,
    #     lead_time=lead_time,
    #     init_years=init_years,
    #     variable=variable,
    #     month_name="November",
    #     figsize=(12, 6),
    #     save_dir="/gws/nopw/j04/canari/users/benhutch/plots/",
    #     save=True,
    # )

    # End the timer
    end = time.time()

    # Print the time taken
    print(f"Time taken: {end - start:.2f} seconds.")
        
    # Print that we are exiting the main function
    print("Exiting main function.")
    sys.exit()


if __name__ == "__main__":
    main()
