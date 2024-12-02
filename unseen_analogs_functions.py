#!/usr/bin/env python

"""
unseen_analogs_functions.py

Script which contains functions to perform the UNSEEN-analogs method for 
exploring long duration (multi-month to multi-year) wind droughts in the
European region.

UNSEEN method based on Kay et al. (2023):
https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/asl.1158

Analogs on some of the methods from Stringer, Thornton, and Knight (2020):
https://journals.ametsoc.org/view/journals/apme/59/2/jamc-d-19-0094.1.xml

Conversion to capacity factors (external to this script) based on Bloomfield models from CLEARHEADS:
https://essd.copernicus.org/articles/14/2749/2022/

Steps:

1. Using the DePreSys output which has been bias corrected to ERA5 (for msl).
2. Grab the daily (bias corrected) daily psl field
3. Find the mean squared difference between the model day and all days from ERA5
4. Select the day from ERA5 with the lowest mean squared differences for msl field.
5. Aggregate observed 'dates' over a month to get a sequence of analog days
(representative of the variability of the model).
6. Extract the 100m wind speed field for the analog days.
7. Pass the bias corrected 100m wind field through the CLEARHEADS model.
8. Get the hourly time series of onshore/offshore wind capacity factors for the ERA5 day.
9. Repeat for all of the days of the DePreSys data.
"""

# Import local modules
import os
import sys
import glob
import time
import argparse

# Import external modules
import numpy as np
import xarray as xr
import pandas as pd
import iris
import cftime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Specific imports
from tqdm import tqdm
from ncdata.iris_xarray import cubes_to_xarray, cubes_from_xarray

# Import dictionaries
sys.path.append("/home/users/benhutch/unseen_functions")
import unseen_dictionaries as udicts


# Define a function to regrid the observed data to the model grid
def regrid_obs_to_model(
    obs_path: str,
    model_path: str,
    grid_bounds: dict = udicts.eu_grid,
    model_var_name: str = "__xarray_dataarray_variable__",
    member_name: str = "r10i1p1f2",
    rename_var: str = "psl",
) -> iris.cube.Cube:
    """
    Regrid the observed data to the model grid.

    Parameters
    ----------
    obs_path : str
        Path to the observed data.
    model_path : str
        Path to the model data.
    grid_bounds : dict
        Dictionary containing the grid bounds for the model data.
    model_var_name : str
        Name of the variable in the model data.
    member_name : str
        Name of the member in the model data.
    rename_var : str
        Name to rename the variable to.

    Returns
    -------
    iris.cube.Cube
        Regridded observed data.
    """
    # Load the observed data
    obs_ds = xr.open_dataset(obs_path)

    # Load the model data
    model_ds = xr.open_dataset(model_path)

    # Subset the obs_ds to the EU grid
    obs_ds = obs_ds.sel(
        longitude=slice(grid_bounds["lon1"], grid_bounds["lon2"]),
        latitude=slice(grid_bounds["lat2"], grid_bounds["lat1"]),
    )

    # Subset the model data to the EU grid
    model_ds = model_ds.sel(
        lon=slice(grid_bounds["lon1"], grid_bounds["lon2"]),
        lat=slice(grid_bounds["lat1"], grid_bounds["lat2"]),
    )

    # if member is a dimension coordinate
    if "member" in model_ds.coords:
        # Select the variable and a member from the model ds
        model_data_member = model_ds[model_var_name].sel(member=member_name)
    else:
        # Select the variable from the model ds
        model_data_member = model_ds[model_var_name]

    # Rename the variable to something acceptable for iris
    model_data_member = model_data_member.rename(rename_var)

    # Convert the model data to an iris cube
    member_cube = model_data_member.squeeze().to_iris()

    # print the member cube
    print("Member cube:", member_cube)

    # if lat and lon are dimensions, then rename them
    if "lat" in member_cube.coords():
        if "lon" in member_cube.coords():
            # Rename the latitude and longitude coordinates
            member_cube.coord("lat").rename("latitude")
            member_cube.coord("lon").rename("longitude")

    # Convert obs data to an iris cube
    obs_cube = cubes_from_xarray(obs_ds)

    # Make sure the units and attributes are the same
    member_cube.coord("latitude").units = obs_cube[0].coord("latitude").units
    member_cube.coord("longitude").units = obs_cube[0].coord("longitude").units

    # and for the attributes
    member_cube.coord("latitude").attributes = obs_cube[0].coord("latitude").attributes
    member_cube.coord("longitude").attributes = (
        obs_cube[0].coord("longitude").attributes
    )

    # Regrid the observed data to the model grid
    obs_cube_rg = obs_cube[0].regrid(member_cube, iris.analysis.Linear())

    return obs_cube_rg


# Define a function to load the model cube and find the analogs
def create_analogs_df(
    model_path: str,
    init_year: int,
    init_month: int,
    obs_cube_rg: iris.cube.Cube,
    grid_bounds: dict = udicts.eu_grid,
    model_var_name: str = "__xarray_dataarray_variable__",
    rename_var: str = "psl",
    df_save_dir: str = "/home/users/benhutch/unseen_functions/save_dfs",
    df_save_fname: str = "analogs_df.csv",
) -> pd.DataFrame:
    """
    Load the model data and find the analogs.

    Parameters
    ----------
    model_path : str
        Path to the model data.
    init_year : int
        Initialisation year to find the analogs for.
    init_month : int
        Initialisation month to find the analogs for.
    obs_cube_rg : iris.cube.Cube
        Regridded observed data.
    grid_bounds : dict
        Dictionary containing the grid bounds for the model data.
    model_var_name : str
        Name of the variable in the model data.
    rename_var : str
        Name to rename the variable to.
    df_save_dir : str
        Directory to save the analogs DataFrame.
    df_save_fname : str
        Filename to save the analogs DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the analogs.
    """
    # Load the model data
    model_ds = xr.open_dataset(model_path)

    # # print the model ds min and max lat
    # print(f"Model ds min lat: {model_ds.lat.min().values}")
    # print(f"Model ds max lat: {model_ds.lat.max().values}")
    # print(f"Model ds min lon: {model_ds.lon.min().values}")
    # print(f"Model ds max lon: {model_ds.lon.max().values}")

    # sys.exit()

    # select the init from the model ds
    model_ds = model_ds.sel(init=init_year)

    # Subset the model data to the EU grid
    model_ds = model_ds.sel(
        lon=slice(grid_bounds["lon1"], grid_bounds["lon2"]),
        lat=slice(grid_bounds["lat1"], grid_bounds["lat2"]),
    )

    # reset the member variable to be integers
    model_ds["member"] = model_ds["member"].str[1:-6].astype(int)

    # Rename the variable to something acceptable for iris
    model_ds = model_ds[model_var_name].rename(rename_var)

    # Convert the model data to an iris cube
    model_cube = model_ds.squeeze().to_iris()

    # if the init_month is in [10, 11, 12], then the subset obs data to months 10, 11, 12
    if init_month in [10, 11, 12]:  # Early winter
        obs_cube_rg = obs_cube_rg.extract(
            iris.Constraint(time=lambda cell: cell.point.month in [10, 11, 12])
        )
    elif init_month in [1, 2, 3]:  # Late winter
        obs_cube_rg = obs_cube_rg.extract(
            iris.Constraint(time=lambda cell: cell.point.month in [1, 2, 3])
        )
    elif init_month in [4, 5, 6]:  # Early summer
        obs_cube_rg = obs_cube_rg.extract(
            iris.Constraint(time=lambda cell: cell.point.month in [4, 5, 6])
        )
    elif init_month in [7, 8, 9]:  # Late summer
        obs_cube_rg = obs_cube_rg.extract(
            iris.Constraint(time=lambda cell: cell.point.month in [7, 8, 9])
        )
    else:
        raise ValueError(
            "init_month must be in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
        )

    # Extract the members
    model_members = model_cube.coord("member").points
    model_leads = model_cube.coord("lead").points
    model_inits = model_cube.coord("init").points

    # extract the obs times
    obs_times = obs_cube_rg.coord("time").points

    # Print these
    print(f"Members: {model_members}")
    print(f"Leads: {model_leads}")
    print(f"Inits: {model_inits}")

    # # print the obs times
    # print(f"Obs times: {obs_times}")

    # # start a timer
    # start = time.time()

    # subset the model cube to the init year
    model_cube = model_cube.extract(iris.Constraint(init=init_year))

    # Extract the data into arrays
    obs_array = obs_cube_rg.data
    model_array = model_cube.data

    # # end
    # end = time.time()

    # # Print the time taken
    # print(f"Time taken to extract the data: {end - start} seconds")

    # print the shape of the arrays
    print(f"Obs shape: {obs_array.shape}")
    print(f"Model shape: {model_array.shape}")

    # start a timer
    start = time.time()

    # if the save directory doesn't exist, create it
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir)

    # form the save path
    df_save_path = os.path.join(df_save_dir, df_save_fname)

    # if the save path exists, load the dataframe
    if os.path.exists(df_save_path):
        print(f"Loading analogs DataFrame from: {df_save_path}")
        analogs_df = pd.read_csv(df_save_path)
        return analogs_df, model_cube

    # Subset the obs array to the first 100 times for testing
    # -------------------------------------------------------
    # print("Subsetting the obs array to the first 100 times for testing...")
    # obs_array = obs_array[:100, :, :]
    # obs_times = obs_times[:100]
    # -------------------------------------------------------

    # set up a list to store the MSE values
    mse_array = np.zeros((len(model_members), len(model_leads), len(obs_times)))

    # Loop over the leads in model_cube
    for l, lead in tqdm(enumerate(model_leads)):
        # Loop over the members in model_cube
        for m, member in enumerate(model_members):
            # Subset the model data to the current lead and member
            model_data = model_array[m, l, :, :]

            # expand dimension of model data to match obs array
            model_data = np.expand_dims(model_data, axis=(0))

            # Calculate the mean squared error between the model and obs data
            mse = np.mean((model_data - obs_array) ** 2, axis=(1, 2))

            # Append the MSE to the list
            mse_array[m, l, :] = mse

    # print the length of the mse_list
    print(f"Length of mse_array: {np.shape(mse_array)}")

    # print the mse_list
    print(f"mse_array: {mse_array}")

    # end
    end = time.time()

    # Print the time taken
    print(f"Time taken to calculate the MSE: {end - start} seconds")

    # set up an empty dataframe to store the MSE
    mse_df = pd.DataFrame()

    # for each of the leads and members find the minimum MSE
    for l, lead in enumerate(model_leads):
        for m, member in enumerate(model_members):
            # find the minimum MSE for each member and lead
            min_mse = np.min(mse_array[m, l, :])

            # find the index of the minimum MSE
            min_mse_idx = np.argmin(mse_array[m, l, :])

            # find the date of the minimum MSE
            min_mse_date = obs_times[min_mse_idx]

            # append the results to the dataframe
            mse_df_this = pd.DataFrame(
                {
                    "lead": [lead],
                    "member": [member],
                    "min_mse": [min_mse],
                    "min_mse_time": [int(min_mse_date)],
                },
            )

            # concat the dataframes
            mse_df = pd.concat([mse_df, mse_df_this])

    # reset the index
    mse_df = mse_df.reset_index(drop=True)

    # if the save path doesn't exist, save the dataframe
    if not os.path.exists(df_save_path):
        print(f"Saving analogs DataFrame to: {df_save_path}")
        mse_df.to_csv(df_save_path)

    return mse_df, model_cube


# Define a function to perform some visual verification
# by plotting the model fields and then the macthing obs field
def plot_model_obs_fields(
    analogs_df: pd.DataFrame,
    model_cube: iris.cube.Cube,
    obs_cube_rg: iris.cube.Cube,
    lead: int = 1,
    members: list = [1, 2, 3, 4, 5],
    figsize: tuple = (12, 18),
    clevs: tuple = (988, 1024, 19),
    cmap: str = "viridis",
    save_dir: str = "/home/users/benhutch/unseen_functions/testing_plots",
) -> None:
    """
    Plot the model and observed fields for the analogs.

    Parameters
    ----------

    analogs_df : pd.DataFrame
        DataFrame containing the analogs.

    model_cube : iris.cube.Cube
        Model cube.

    obs_cube_rg : iris.cube.Cube
        Regridded observed data.

    lead : int
        Lead time to plot.

    members : list
        List of members to plot.

    figsize : tuple
        Figure size.

    clevs : tuple
        Contour levels.

    cmap : str
        Colormap.

    save_dir : str
        Directory to save the plots.

    Returns
    -------

    None

    """

    # Set up the figure
    fig, axs = plt.subplots(
        nrows=len(members),
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Loop over the members
    for i, member in enumerate(members):
        # Find the row with the correct lead and member
        row = analogs_df[
            (analogs_df["lead"] == lead) & (analogs_df["member"] == member)
        ]

        # Extract the time of the minimum MSE
        min_mse_time = row["min_mse_time"].values[0]

        # Extract the model cube for the lead and member
        model_cube_this = model_cube.extract(iris.Constraint(lead=lead, member=member))

        # Set up the time using cftime
        time_this = cftime.num2date(min_mse_time, "hours since 1900-01-01", "gregorian")

        # Extract the obs data for this time
        obs_cube_this = obs_cube_rg.extract(iris.Constraint(time=time_this))

        # Extract the model and obs field (as hPa)
        model_field = model_cube_this.data / 100
        obs_field = obs_cube_this.data / 100

        # Extract the lats and lons
        lats = obs_cube_this.coord("latitude").points
        lons = obs_cube_this.coord("longitude").points

        # Include coastlines
        axs[i, 0].coastlines()

        # Include the gridlines
        gl = axs[i, 0].gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False, linestyle="--"
        )

        # Set up the title
        axs[i, 0].set_title(f"Model: lead={lead}, member={member}", fontsize=8)

        # Plot the model field
        im = axs[i, 0].contourf(
            lons,
            lats,
            model_field,
            levels=np.linspace(*clevs),
            cmap=cmap,
            extend="both",
        )

        # Include the coastlines
        axs[i, 1].coastlines()

        # Include the gridlines
        gl = axs[i, 1].gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False, linestyle="--"
        )

        # subset time this to the first 10 characters
        time_this = str(time_this)[:10]

        # Set up the title
        axs[i, 1].set_title(f"Obs analog: time={time_this}", fontsize=8)

        # Plot the obs field
        im = axs[i, 1].contourf(
            lons, lats, obs_field, levels=np.linspace(*clevs), cmap=cmap, extend="both"
        )

    # Add a colorbar
    fig.colorbar(
        im, ax=axs, orientation="horizontal", label="hPa", pad=0.05, shrink=0.8
    )

    # set up the current time
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save the figure
    plt.savefig(
        os.path.join(save_dir, f"model_obs_fields_lead_{lead}_{current_time}.png")
    )

    # Close the figure
    plt.close()

    return None


# Main function for testing
def main():
    # start a timer
    start = time.time()

    # Define the paths to the observed and model data
    obs_path = (
        "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_msl_daily_1960_2020_daymean.nc"
    )

    # assert that the paths exist
    assert os.path.exists(obs_path), f"Observed data not found at: {obs_path}"

    # Create the parser
    parser = argparse.ArgumentParser(description="Process initialisation year and month.")

    # Add the arguments
    parser.add_argument('--init_year', type=int, required=True, help='The initialisation year.')
    parser.add_argument('--init_month', type=int, required=True, help='The initialisation month.')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use args.init_year and args.init_month in your code
    init_year = args.init_year
    init_month = args.init_month

    model_path = f"/work/scratch-nopw2/benhutch/test_nc/psl_bias_correction_HadGEM3-GC31-MM_lead1_month{init_month}_init1960-2018.nc"

    assert os.path.exists(model_path), f"Model data not found at: {model_path}"

    # Set up the save directory
    save_dir = "/gws/nopw/j04/canari/users/benhutch/unseen_analogs"
    save_fname = "ERA5_msl_daily_1960_2020_daymean_EU_grid_HadGEM_regrid.nc"

    # if the save directory doesn't exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # form the save path
    save_path = os.path.join(save_dir, save_fname)

    # if the save path exists, load the cube
    if os.path.exists(save_path):
        print(f"Loading regridded observed data from: {save_path}")
        obs_cube_rg = iris.load_cube(save_path)
    else:
        print(f"Regridding observed data to model grid...")
        # Regrid the observed data to the model grid
        obs_cube_rg = regrid_obs_to_model(obs_path, model_path)

        # if the save path doesn't exist, save the cube
        if not os.path.exists(save_path):
            print(f"Saving regridded observed data to: {save_path}")
            iris.save(obs_cube_rg, save_path)

    # # Print the cube
    print("Regridded observed data:", obs_cube_rg)

    # Create the analogs DataFrame
    mse_df, model_cube = create_analogs_df(
        model_path=model_path,
        init_year=init_year,
        init_month=init_month,
        obs_cube_rg=obs_cube_rg,
        df_save_dir="/gws/nopw/j04/canari/users/benhutch/unseen_analogs/analogs_dfs",
        df_save_fname=f"analogs_df_{init_year}_{init_month}_full_model.csv",
    )

    # # # Print the cube
    # print("mse_df:", mse_df)

    # # # Plot the model and observed fields for the analogs
    # plot_model_obs_fields(
    #     analogs_df=mse_df,
    #     model_cube=model_cube,
    #     obs_cube_rg=obs_cube_rg,
    #     lead=30,
    #     members=[1, 2, 3, 4, 5],
    #     figsize=(8, 15),
    # )

    # # end the timer
    end = time.time()

    # # Print the time taken
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    main()
