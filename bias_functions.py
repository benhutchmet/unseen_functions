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
import cftime
import netCDF4

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

    # Extract the unique variants
    variants = list(set(variants))

    # Print the unique variants
    print(f"Unique variants: {variants}")

    # print the shape of the agg_files
    print(f"Shape of agg_files: {len(agg_files)}")

    # Set up the init_year list
    init_year_list = []

    # Loop over the initialisation years
    for init_year in tqdm(init_years, desc="Loading data"):
        # member list for the xr objects
        member_list = []

        # Load the data by looping over the unique variants
        for variant in variants:
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

    # If the frequency is 'day'
    # We have to extract the correct leads after the data has been loaded
    if frequency == "day":
        if ds.attrs["time_axis_type"] == "Datetime360Day":
            print(
                "Converting the lead times to days assuming datetime360 day frequency."
            )

            # Set up the dates list
            dates = []

            # Set up the first day
            first_day = pd.to_datetime(ds.attrs["first_month"])

            # Loop over the lead times
            for i in range(len(ds["lead"])):
                # Calculate the number of years, months, and days to add
                years_to_add = i // 360
                months_to_add = (i % 360) // 30
                days_to_add = (i % 360) % 30

                # Calculate the new lead time
                new_year = first_day.year + years_to_add
                new_month = first_day.month + months_to_add
                new_day = first_day.day + days_to_add

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
            raise ValueError(
                f"Time axis type {ds.attrs['time_axis_type']} not recognised."
            )

        # print the first date
        print(f"First date: {dates[0]}")

        # Print the last date
        print(f"Last date: {dates[-1]}")

        # loop over and print all of the attributes
        # print theb time units
        first_file = agg_files[0]

        # Open the NetCDF file
        dataset = netCDF4.Dataset(first_file)

        # Access the time variable
        time = dataset.variables["time"]

        # Print the characteristics of time
        print(f"bounds: {time.bounds}")
        print(f"units: {time.units}")
        print(f"calendar: {time.calendar}")
        print(f"axis: {time.axis}")
        print(f"long_name: {time.long_name}")
        print(f"standard_name: {time.standard_name}")

        # extract time.units as a string
        time_units = str(time.units)

        # extract the calendar as string
        calendar = str(dates[0].calendar)

        # # print dates[0].calendar
        # print(f"dates[0].calendar: {dates[0].calendar}")
        # # prtin time_units
        # print(f"time_units: {time_units}")

        # print the first value of dates
        print(f"First value of time: {dataset.variables['time'][0]}")

        # Set up the first day to constrain
        first_day = cftime.date2num(dates[0], time_units, calendar, has_year_zero=True)

        if lead_time != 1:
            print("Lead time is not 1")
            # Add the lead time to the first day
            add_time = (
                cftime.date2num(dates[0], time_units, calendar, has_year_zero=True)
                + (lead_time - 1) * 360
            )

        else:
            print("Lead time is 1")
            add_time = cftime.date2num(
                dates[0], time_units, calendar, has_year_zero=True
            )

        # Set up the end time
        end_time = (
            cftime.date2num(dates[0], time_units, calendar, has_year_zero=True)
            - 1
            + (lead_time * 360)
        )

        # Convert from num to date
        add_time_date = cftime.num2date(
            add_time, time_units, calendar, has_year_zero=True
        )

        # num 2 date for the end time
        end_time_date = cftime.num2date(
            end_time, time_units, calendar, has_year_zero=True
        )

        # print the first day
        print(f"First day: {add_time_date}")
        print(f"Last day: {end_time_date}")

        # Convert dates to a numpy array
        dates_array = np.array(dates)

        # Extract the indices of these dates in the dates list
        indices = np.where(
            (dates_array >= add_time_date) & (dates_array <= end_time_date)
        )[0]

        # Use these indices to extract the data for lead at these indices
        # Extract the values of lead
        lead_values = ds["lead"].values

        # Extract the values of lead at the indices
        lead_values = lead_values[indices]

        # Extract the indices of these dates in the dates list
        ds = ds.isel(lead=lead_values)

        # Add the first day and last day as an attribute
        ds.attrs["first_day"] = add_time_date
        ds.attrs["last_day"] = end_time_date

        # print ds
        print(f"ds: {ds}")

        # Set up the lead values
        ds["lead"] = np.arange(1, len(ds["lead"]) + 1)

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

    # Not sure how best to implement this yet
    # elif frequency == "day":
    #     raise NotImplementedError("Daily data not yet implemented.")
    # else:
    #     raise ValueError(f"Frequency {frequency} not recognised.")

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
    fig, ax = plt.subplots(
        1, 2, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot the mean bias
    # ax[0].imshow(mean_bias.values, cmap="bwr", vmin=-10, vmax=10, transform=ccrs.PlateCarree(), interpolation='none')
    # ax[0].set_title('Mean Bias')

    # # Plot the sigma bias
    # ax[1].imshow(sigma_bias.values, cmap="bwr", vmin=-5, vmax=5, transform=ccrs.PlateCarree(), interpolation='none')
    # ax[1].set_title('Sigma Bias')

    # set up the contour levels
    # clevs = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
    clevs = np.array(
        [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

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
    cbar.set_label("Mean Bias")

    # Set up the contour levels
    # clevs = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    clevs = np.array(
        [
            -5,
            -4.5,
            -4.0,
            -3.5,
            -3.0,
            -2.5,
            -2.0,
            -1.5,
            -1.0,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
        ]
    )

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
    cbar.set_label("Sigma Bias")

    # add coastlines
    ax[0].coastlines()

    # add coastlines
    ax[1].coastlines()

    # Set the title
    ax[0].set_title(f"mean bias")

    # Set the title
    ax[1].set_title(f"sigma Bias")

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
    freq: str = "Amon",
    mean_or_std: str = "mean",
    figsize: tuple = (12, 6),
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
    save: bool = True,
    constrained_grid: dict = dicts.eu_grid,
    vmin_set: float = -9999.0,
    vmax_set: float = 9999.0,
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
    freq : str, optional
        The frequency of the data, by default "Amon".
    mean_or_std : str, optional
        Whether to calculate the mean or standard deviation, by default "mean".
        Only takes values of "mean" or "std".
    figsize : tuple, optional
        The size of the figure to plot, by default (12, 6).
    save_dir : str, optional
        The directory to save the plots to, by default "/gws/nopw/j04/canari/users/benhutch/plots/".
    save : bool, optional
        Whether to save the plots, by default True.
    constrained_grid : dict, optional
        The dictionary containing the constrained grid, by default dicts.eu_grid.
    vmin_set : float, optional
        The minimum value for the colorbar, by default -9999.0.
    vmax_set : float, optional
        The maximum value for the colorbar, by default 9999.0.

    Returns
    -------

    """

    # Extract the lats and lons from the obs
    lats = obs_ds["lat"]
    lons = obs_ds["lon"]

    # Set up the figure
    fig, ax = plt.subplots(
        4,
        3,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    # # Create a colorbar axes
    # cbar_ax = fig.add_axes([0.1, 0.2, 0.8, 0.02])

    # if the variable is rsds
    if variable == "rsds" and freq == "Amon":
        print("Convert rsds to W/m^2 from J/m^2")

        # Divide the obs data by 86400 (to convert from J/m^2 to W/m^2 in days)
        obs_ds = obs_ds / 86400

    # Initialize global vmin and vmax
    global_vmin = np.inf
    global_vmax = -np.inf

    # Iterate over all data to find global vmin and vmax
    for i, month_name in tqdm(enumerate(month_names), desc="Calculating vmin and vmax"):

        # Select the month idx from the model data
        model_ds_month = model_ds.sel(lead=i + 1)

        # Find the month at the correct month_idx in the obs
        obs_month = obs_ds.time.dt.month[i - 1]

        # Select the month from the obs data
        obs_month_data = obs_ds.where(obs_ds.time.dt.month == obs_month, drop=True)

        # Calculate the bias
        if mean_or_std == "mean":
            bias = model_ds_month.mean(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.mean(dim="time")
        elif mean_or_std == "std":
            bias = model_ds_month.std(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.std(dim="time")

        # Update global vmin and vmax
        vmin = np.floor(bias.min())
        vmax = np.ceil(bias.max())
        global_vmin = min(global_vmin, vmin)
        global_vmax = max(global_vmax, vmax)

        if variable == "tas":
            if mean_or_std == "mean":
                # set up clev_int
                clev_int = 1

                # ticks
                ticks_int = 2
            else:
                # set up clev_int
                clev_int = 0.25

                # ticks
                ticks_int = 1
        elif variable == "sfcWind":
            if mean_or_std == "mean":
                # set up clev_int
                clev_int = 0.25

                # ticks
                ticks_int = 2
            elif mean_or_std == "std":
                # set up clev_int
                clev_int = 0.0625

                # ticks
                ticks_int = 0.5
        elif variable == "rsds":
            if mean_or_std == "mean":
                # set up clev_int
                clev_int = 10

                # # ticks
                ticks_int = 20
            elif mean_or_std == "std":
                # set up clev_int
                clev_int = 2

                # # ticks
                ticks_int = 4

    if abs(global_vmin) > global_vmax:
        # Then set vmax to abs vmin
        global_vmax = abs(global_vmin)
    else:
        # Set vmin to -vmax
        global_vmin = -global_vmax

    # if vmin and vmax are not -9999 and 9999
    if vmin_set != -9999.0 and vmax_set != 9999.0:
        # Set the vmin and vmax
        global_vmin = vmin_set
        global_vmax = vmax_set

    # print the global vmin and vmax
    print(f"Global vmin: {global_vmin}, Global vmax: {global_vmax}")

    # Set up the contour levels using global vmin and vmax
    clevs = np.arange(global_vmin, global_vmax + clev_int, clev_int)

    # print the clevs
    print(f"clevs: {clevs}")

    # Set up the ticks
    ticks = np.arange(global_vmin, global_vmax + ticks_int, ticks_int)

    # print the ticks
    print(f"ticks: {ticks}")

    # Loop over the month names
    for i, month_name in tqdm(enumerate(month_names), desc="Plotting"):
        print(f"plotting {month_name} at index {i}")

        # Select the month idx from the model data
        model_ds_month = model_ds.sel(lead=i + 1)

        # Find the month at the correct month_idx in the obs
        obs_month = obs_ds.time.dt.month[i - 1]

        # Select the month from the obs data
        obs_month_data = obs_ds.where(obs_ds.time.dt.month == obs_month, drop=True)

        # print that we have selected months
        print(f"Selected months for {month_name}")

        # if the mean_or_std is mean
        if mean_or_std == "mean":
            print(f"Calculating mean bias for {month_name}")
            # Calculate the bias as model_mean - obs_mean
            bias = model_ds_month.mean(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.mean(dim="time")
            print(f"Calculated mean bias for {month_name}")

            # # Set up the contour levels
            # #         clevs = np.array(
            # #     [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
            # # )

            # # # Set up the ticks
            # # ticks = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

            # # print(f"Calculating vmin and vmax for {month_name}")
            # # Set the vmin and vmax
            # vmin = np.floor(bias.min())
            # vmax = np.ceil(bias.max())

            # # if abs vmin > vmax
            # if abs(vmin) > vmax:
            #     # Then set vmax to abs vmin
            #     vmax = abs(vmin)
            # else:
            #     # Set vmin to -vmax
            #     vmin = -vmax

            # # print the vmin and vmax
            # print(f"vmin: {vmin}, vmax: {vmax} for {month_name} and {mean_or_std}")

            # # Set up the interval depending on the range
            # if variable == "tas":
            #     # set up clev_int
            #     clev_int = 1

            #     # ticks
            #     ticks = np.arange(vmin, vmax + 1, 2)

            # elif variable == "sfcWind":
            #     # set up clev_int
            #     clev_int = 0.25

            #     # ticks
            #     ticks = np.arange(vmin, vmax + 1, 2)
            # elif variable == "rsds":
            #     # set up clev_int
            #     clev_int = 5

            #     # # ticks
            #     # ticks = np.arange(vmin, vmax + 1, 10)

            # # # print the clev_int
            # # print(f"clev_int: {clev_int} for {month_name} and {mean_or_std}")

            # # Set up the contour levels
            # clevs = np.arange(vmin, vmax + clev_int, clev_int)

            # # Set up the ticks
            # ticks = np.arange(vmin, vmax + 1, 2)

        elif mean_or_std == "std":
            print(f"Calculating std bias for {month_name}")
            # Calculate the bias as model_sigma - obs_sigma
            bias = model_ds_month.std(
                dim=["init", "member"], skipna=True
            ) - obs_month_data.std(dim="time")
            print(f"Calculated std bias for {month_name}")

            # print the clev_int
            print(f"clev_int: {clev_int} for {month_name} and {mean_or_std}")

            # # Set up the contour levels
            # clevs = np.arange(vmin, vmax + clev_int, clev_int)

            # # Set up the ticks
            # ticks = np.arange(vmin, vmax + 0.25, 0.5)
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
            ax=ax[i // 3, i % 3],
            cmap="bwr",
            levels=clevs,
            vmin=global_vmin,
            vmax=global_vmax,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )

        # Set the x and y limits to the desired domain
        ax[i // 3, i % 3].set_xlim([constrained_grid["lon1"], constrained_grid["lon2"]])
        ax[i // 3, i % 3].set_ylim([constrained_grid["lat1"], constrained_grid["lat2"]])

        # cbar = plt.colorbar(contour, ax=ax[i // 2, i % 2], ticks=ticks, shrink=0.8)
        # cbar.set_label(f"{mean_or_std.capitalize()} Bias")

        # add coastlines
        ax[i // 3, i % 3].coastlines()

        # Set the title
        ax[i // 3, i % 3].set_title(f"{month_name}")

        # Set the xlabel
        ax[i // 3, i % 3].set_xlabel("Lon")

        # Set the ylabel
        ax[i // 3, i % 3].set_ylabel("Lat")

    # Create a colorbar for the whole figure
    cbar = fig.colorbar(
        contour, ax=ax, orientation="vertical", ticks=ticks, shrink=0.6, extend="both"
    )
    cbar.set_label(f"{mean_or_std.capitalize()} Bias")

    # fig.subplots_adjust(bottom=0.25)

    # Set up the super title
    fig.suptitle(
        f"{variable} {mean_or_std.capitalize()} bias for lead {lead_time} years {init_years[0]}-{init_years[-1]} and {freq}"
    )

    # Set up the current time
    current_time = time.strftime("%Y%m%dT%H%M%S")

    # Set up the fname
    fname = f"{variable}_{mean_or_std}_bias_lead{lead_time}_init{init_years[0]}-{init_years[-1]}_freq{freq}_{current_time}.pdf"

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


# define a function to calculate the bias correction coefficients
# for each point of the grid
# and then output them to a netcdf file
def calc_and_save_bias_coeffs(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    lead_time: int,
    month: int,
    init_years: list[int],
    variable: str,
    model_name: str,
    save_flag: bool = False,
    save_dir: str = "/work/scratch-nopw2/benhutch/test_nc/",
):
    """
    Calculates the bias correction coefficients for the provided data using the
    method from Luo et al. (2018) and saves them to a netcdf file.

    Parameters
    ----------

    model_ds : xr.Dataset
        The model dataset to calculate the bias for.
    obs_ds : xr.Dataset
        The observed dataset to calculate the bias for.
    lead_time : int
        The lead time to calculate the bias for.
    month : int
        The index of the month to calculate the bias for.
    init_years : list[int]
        The initialization years to calculate the bias for.
    variable : str
        The variable to calculate the bias for.
    model_name : str
        The name of the model to calculate the bias for.
    save_flag : bool, optional
        Whether to save the bias coefficients, by default False.
    save_dir : str, optional
        The directory to save the files to, by default "/work/scratch-nopw2/benhutch/test_nc/".

    Returns
    -------

    """

    if month in [11, 12]:
        # Set up the leads to select
        leads_sel = np.arange(((month - 11) * 30) + 1, ((month - 11) * 30) + 31)
    elif month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Set up the leads to select
        leads_sel = np.arange(60 + ((month - 1) * 30) + 1, 60 + ((month - 1) * 30) + 31)
    else:
        raise ValueError(f"Month {month} not recognised.")

    # print the leads to sel
    print(f"Leads to select: {leads_sel}")

    # Select the leads
    model_ds_month = model_ds.sel(lead=leads_sel)

    # Select the obs data for the given month
    obs_month = obs_ds.sel(time=obs_ds.time.dt.month == month)

    # print the obs month
    print(f"Obs month: {obs_month}")

    # print the model month
    print(f"Model month: {model_ds_month}")

    # extract the lats and lons
    lats = obs_month["lat"].values
    lons = obs_month["lon"].values

    # # print the lats and lons
    # print(f"Lats: {lats}")
    # print(f"Lons: {lons}")

    # Extract the model_ds_month as numpy array
    model_ds_month_np = model_ds_month.values

    # Extract the obs_month as numpy array
    obs_month_np = obs_month.values

    # print the shape of the model_ds_month_np
    print(f"Shape of model_ds_month_np: {np.shape(model_ds_month_np)}")

    # print the shape of the obs_month_np
    print(f"Shape of obs_month_np: {np.shape(obs_month_np)}")

    # set up a list to store the bias corrected model data
    bc_model_data_month = np.zeros(
        [
            np.shape(model_ds_month_np)[0],
            np.shape(model_ds_month_np)[1],
            np.shape(model_ds_month_np)[2],
            np.shape(model_ds_month_np)[3],
            np.shape(model_ds_month_np)[4],
        ]
    )

    # print the shape of the bc_model_data_month
    print(f"Shape of bc_model_data_month: {np.shape(bc_model_data_month)}")

    # Loop over the lats
    for x, lon in tqdm(enumerate(lons), desc="Calculating bias coefficients lon"):
        for y, lat in enumerate(lats):

            # Select the obs data for the given lat and lon
            obs_month_np_point = obs_month_np[:, y, x]

            # Select the model data for the given lat and lon
            model_month_np_point = model_ds_month_np[:, :, :, y, x].flatten()

            # if all obs values and model values are nan, then continue
            if np.all(np.isnan(obs_month_np_point)) and np.all(
                np.isnan(model_month_np_point)
            ):
                # Reshape the model_month_np_point to the original shape in Nans
                bc_model_data_month[:, :, :, y, x] = np.reshape(
                    model_month_np_point,
                    (
                        np.shape(model_ds_month_np)[0],
                        np.shape(model_ds_month_np)[1],
                        np.shape(model_ds_month_np)[2],
                    ),
                )
                continue

            # # print the values of model_month_np_point
            # print(f"Model month np point: {model_month_np_point}")

            # Apply the linear scaling method to the mean
            model_month_np_point_ls = model_month_np_point + (
                np.mean(obs_month_np_point) - np.mean(model_month_np_point)
            )

            # normalise the model_month_np_point_ls to a zero mean
            model_month_np_point_norm = model_month_np_point_ls - np.mean(
                model_month_np_point_ls
            )

            # Scale the variance
            model_month_np_point_norm = (
                np.std(obs_month_np_point) / np.std(model_month_np_point_norm)
            ) * model_month_np_point_norm

            # Add the mean back
            model_month_np_point_norm = model_month_np_point_norm + np.mean(
                model_month_np_point_ls
            )

            # if the data has been flattened to shape (300,)
            # but we want to get it back to the original shape (1, 10, 30)
            # then append the data to the bc_model_data_month
            bc_model_data_month[:, :, :, y, x] = np.reshape(
                model_month_np_point_norm,
                (
                    np.shape(model_ds_month_np)[0],
                    np.shape(model_ds_month_np)[1],
                    np.shape(model_ds_month_np)[2],
                ),
            )

    # PRINT THE SHAPE OF THE BC_MODEL_DATA_MONTH
    print(f"Shape of bc_model_data_month: {np.shape(bc_model_data_month)}")

    # print the bc_model_data_month
    print(f"BC Model Data Month: {bc_model_data_month}")

    # if save_flag is True
    if save_flag:

        # if the save_dir does not exist
        if not os.path.exists(save_dir):
            # make the directory
            os.makedirs(save_dir)

        # Set up the filename
        fname = f"{variable}_bias_correction_{model_name}_lead{lead_time}_month{month}_init{init_years[0]}-{init_years[-1]}.nc"

        # Set up the array as a xarray dataset
        bc_model_data_month_ds = xr.DataArray(
            bc_model_data_month,
            dims=["init", "member", "lead", "lat", "lon"],
            coords={
                "lead": model_ds_month.lead.values,
                "init": model_ds_month.init.values,
                "member": model_ds_month.member.values,
                "lat": lats,
                "lon": lons,
            },
        )

        # Set up the path
        save_path = os.path.join(save_dir, fname)

        # if save_path exists
        if os.path.exists(save_path):
            # print that the file already exists
            print(f"File {save_path} already exists.")        
        else:
            # save the file
            delayed_obj = bc_model_data_month_ds.to_netcdf(save_path, compute=False)

            try:
                with ProgressBar():
                    print("Saving bias corrected data")
                    results = delayed_obj.compute()
            except e as Exception:
                print(f"Error saving file: {e}")


    return bc_model_data_month


# Define a function for plotting and comparing the mean and bc data
def verify_bc_plot(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    bc_model_data: np.ndarray,
    lead_time: int,
    month: int,
    init_years: list[int],
    variable: str,
    model_name: str,
    mean_or_std: str = "mean",
    mean_clevs: tuple = (-5, 5.5, 0.5),
    mean_ticks: tuple = (-5, 5.5, 1),
    std_clevs: tuple = (-3, 3.5, 0.5),
    std_ticks: tuple = (-3, 3.5, 1),
    save_dir: str = "/work/scratch-nopw2/benhutch/test_nc/",
):
    """

    Plots the spatial bias as the model - obs for the mean or standard deviation
    for both the original data (left) and the bias corrected data (right).

    Parameters
    ----------

    model_ds : xr.Dataset
        The model dataset to calculate the bias for.
    obs_ds : xr.Dataset
        The observed dataset to calculate the bias for.
    bc_model_data : np.ndarray
        The bias corrected model data.
    lead_time : int
        The lead time to calculate the bias for.
    month : int
        The index of the month to calculate the bias for.
    init_years : list[int]
        The initialization years to calculate the bias for.
    variable : str
        The variable to calculate the bias for.
    model_name : str
        The name of the model to calculate the bias for.
    mean_or_std : str, optional
        Whether to calculate the mean or standard deviation, by default "mean".
        Only takes values of "mean" or "std".
    mean_clevs : tuple, optional
        The contour levels for the mean bias, by default (-5, 5.5, 0.5).
    mean_ticks : tuple, optional
        The ticks for the mean bias, by default (-5, 5.5, 1).
    std_clevs : tuple, optional
        The contour levels for the standard deviation bias, by default (-3, 3.5, 0.5).
    std_ticks : tuple, optional
        The ticks for the standard deviation bias, by default (-3, 3.5, 1).
    save_dir : str, optional
        The directory to save the files to, by default "/work/scratch-nopw2/benhutch/test_nc/".

    Returns
    -------

    """

    # Select the leads to plot
    if month in [11, 12]:
        # Set up the leads to select
        leads_sel = np.arange(((month - 11) * 30) + 1, ((month - 11) * 30) + 31)
    elif month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Set up the leads to select
        leads_sel = np.arange(60 + ((month - 1) * 30) + 1, 60 + ((month - 1) * 30) + 31)
    else:
        raise ValueError(f"Month {month} not recognised.")

    # print the leads to sel
    print(f"Leads to select: {leads_sel}")

    # Select the leads
    model_ds_month = model_ds.sel(lead=leads_sel)

    # Select the obs data for the given month
    obs_month = obs_ds.sel(time=obs_ds.time.dt.month == month)

    # print the obs month
    print(f"Obs month: {obs_month}")

    # print the model month
    print(f"Model month: {model_ds_month}")

    # extract the lats and lons
    lats = obs_month["lat"].values
    lons = obs_month["lon"].values

    # extract the values
    obs_month_np = obs_month.values

    # print the min and max of the obs_month_np
    print(f"Min obs_month_np: {obs_month_np.min()}, Max obs_month_np: {obs_month_np.max()}")
    # print the mean of the obs_month_np
    print(f"Mean obs_month_np: {np.nanmean(obs_month_np)}")

    # extract the values
    model_month_np = model_ds_month.values

    # print the min and max of the model_month_np
    print(f"Min model_month_np: {model_month_np.min()}, Max model_month_np: {model_month_np.max()}")
    # print the mean of the model_month_np
    print(f"Mean model_month_np: {np.nanmean(model_month_np)}")

    # print the shape
    print(f"Shape of model_month_np: {np.shape(model_month_np)}")

    # print the shape
    print(f"Shape of obs_month_np: {np.shape(obs_month_np)}")

    # assert that not all of the values are Nan
    assert not np.all(np.isnan(model_month_np)), "All values of model_month_np are nan"

    # assert that not all of the values are Nan
    assert not np.all(np.isnan(obs_month_np)), "All values of obs_month_np are nan"

    # # if the mean_or_std is mean
    if mean_or_std == "mean":
        # Calculate the bias as model_mean - obs_mean
        bias = np.nanmean(model_month_np, axis=(0, 1, 2)) - np.nanmean(obs_month_np, axis=0)

        # calculate the bias corrected bias
        bc_bias = np.nanmean(bc_model_data, axis=(0, 1, 2)) - np.nanmean(obs_month_np, axis=0)

    elif mean_or_std == "std":
        # Calculate the bias as model_sigma - obs_sigma
        bias = np.nanstd(model_month_np, axis=(0, 1, 2)) - np.nanstd(obs_month_np, axis=0)

        # calculate the bias corrected bias
        bc_bias = np.nanstd(bc_model_data, axis=(0, 1, 2)) - np.nanstd(obs_month_np, axis=0)
    else:
        raise ValueError(f"mean_or_std {mean_or_std} not recognised.")
    
    # print the shapes of the bias and bc_bias
    print(f"Shape of bias: {np.shape(bias)}")

    # print the shapes of the bias and bc_bias
    print(f"Shape of bc_bias: {np.shape(bc_bias)}")

    # assert that not all of the values are Nan
    assert not np.all(np.isnan(bias)), "All values of bias are nan"

    # assert that not all of the values are Nan
    assert not np.all(np.isnan(bc_bias)), "All values of bc_bias are nan"

    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # print the min and max of the bias
    print(f"Min bias: {bias.min()}, Max bias: {bias.max()}")

    # print the min and max of the bc_bias
    print(f"Min bc_bias: {bc_bias.min()}, Max bc_bias: {bc_bias.max()}")

    if mean_or_std == "mean":
        # Set up the contour levels
        clevs = np.arange(mean_clevs[0], mean_clevs[1], mean_clevs[2])
        # Set up the ticks
        ticks = np.arange(mean_ticks[0], mean_ticks[1], mean_ticks[2])
    else:
        # Set up the contour levels
        clevs = np.arange(std_clevs[0], std_clevs[1], std_clevs[2])
        # Set up the ticks
        ticks = np.arange(std_ticks[0], std_ticks[1], std_ticks[2])

    # remove 0 from the clevs
    clevs = clevs[clevs != 0]

    # remove 0 from the ticks
    ticks = ticks[ticks != 0]

    # Set up the first contourf object
    contour_bias = axs[0].contourf(
        lons,
        lats,
        bias,
        clevs,
        transform=ccrs.PlateCarree(),
        cmap="bwr",
        extend="both",
    )

    # set the x and y lims to the desired domain
    axs[0].set_xlim([dicts.eu_grid["lon1"], dicts.eu_grid["lon2"]])
    axs[0].set_ylim([dicts.eu_grid["lat1"], dicts.eu_grid["lat2"]])

    # Set up the second contourf object
    contour_bc = axs[1].contourf(
        lons,
        lats,
        bc_bias,
        clevs,
        transform=ccrs.PlateCarree(),
        cmap="bwr",
        extend="both",
    )

    # set the x and y lims to the desired domain
    axs[1].set_xlim([dicts.eu_grid["lon1"], dicts.eu_grid["lon2"]])
    axs[1].set_ylim([dicts.eu_grid["lat1"], dicts.eu_grid["lat2"]])

    # Add coastlines
    axs[0].coastlines()

    # Add coastlines
    axs[1].coastlines()

    # Set the title
    axs[0].set_title(f"Mean Bias {model_name}")

    # Set the title
    axs[1].set_title(f"Mean Bias BC {model_name}")

    # set up the cbar
    cbar = plt.colorbar(contour_bias, ax=axs, ticks=ticks, shrink=0.8,orientation='horizontal')
    cbar.set_label(f"{mean_or_std.capitalize()} Bias")

    fig.subplots_adjust(bottom=0.3)

    # set up the super title
    fig.suptitle(
        f"{variable} {mean_or_std.capitalize()} bias for lead {lead_time} years {init_years[0]}-{init_years[-1]} month {month}"
    )

    # move the subtitle closer to the plot
    plt.subplots_adjust(top=0.9)

    # show the plot
    plt.show()




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

    parser.add_argument(
        "--month_bc",
        type=int,
        help="The month to calculate the bias correction for.",
        default=11, # November, the first month
    )

    # Set up the args
    args = parser.parse_args()

    # sbatch submit_bias_process.bash HadGEM3-GC31-MM tas t2m 2 1960 1960 dcppA-hindcast Amon 12

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
    print(f"Month BC: {args.month_bc}")

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
    month_bc = args.month_bc

    # Test file for monthtly data
    # test_file = "/gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/tas/HadGEM3-GC31-MM/merged_files/tas_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_196011-197102.nc"

    if variable in ["tas", "t2m"]:
        test_file = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1960-r1i1p1f2/day/tas/gn/files/d20200417/tas_day_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_19601101-19601230.nc"
    elif variable in ["sfcWind", "si10"]:
        test_file = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1960-r1i1p1f2/day/sfcWind/gn/files/d20200417/sfcWind_day_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_19601101-19601230.nc"
    elif variable in ["psl"]:
        test_file = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1960-r1i1p1f2/day/psl/gn/files/d20200417/psl_day_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_19601101-19601230.nc"
    elif variable in ["rsds"]:
        test_file = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1960-r1i1p1f2/day/rsds/gn/files/d20200417/rsds_day_HadGEM3-GC31-MM_dcppA-hindcast_s1960-r1i1p1f2_gn_19601101-19601230.nc"
    else:
        raise ValueError(f"Variable {variable} not recognised.")

    # depending on the variable set up the obs_fpath
    if variable in ["tas", "t2m"]:
        # Set up the obs fpath
        obs_fpath = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_t2m_daily_1960_2020.nc"
    elif variable in ["sfcWind", "si10"]:
        # Set up the obs fpath
        obs_fpath = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1960_2020.nc"
    elif variable in ["rsds"]:
        # Set up the obs fpath
        obs_fpath = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_rsds_daily_1960_2020.nc"
    elif variable in ["psl"]:
        # Set up the obs fpath
        obs_fpath = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_msl_daily_1960_2020_daymean.nc"
    else:
        raise ValueError(f"Variable {variable} not recognised.")

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

    # # Print the ds
    # print(f"DS: {ds}")

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

    # # print the obs
    # print(f"Obs: {obs}")

    # # print ds
    # print(f"DS: {ds}")

    # test the calc and plot bias function
    bc_data = calc_and_save_bias_coeffs(
        model_ds=ds,
        obs_ds=obs,
        lead_time=lead_time,
        month=month_bc,
        init_years=init_years,
        variable=variable,
        model_name=model,
        save_flag=True,
    )

    # print the shape of bc data
    print(f"Shape of bc_data: {np.shape(bc_data)}")

    # End the timer
    end = time.time()

    # Print the time taken
    print(f"Time taken: {end - start:.2f} seconds.")

    # # Print that we are exiting the main function
    # print("Exiting main function.")
    # sys.exit()


if __name__ == "__main__":
    main()
