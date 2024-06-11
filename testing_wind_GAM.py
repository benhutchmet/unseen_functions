"""
test_wind_GAM.py
----------------

Functions for testing the developing of Generalised Additive Models (GAMs) for converting from 10m wind speed to 100m wind speed using smooth functions of other variables.

Arguments:
----------

    None

Returns:
--------

    None
"""

# Imports
import os
import sys
import glob
import argparse
import time

# Third-party libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Import the functions
from load_wind_functions import apply_country_mask

from functions_demand import (
    calc_spatial_mean,
    calc_hdd_cdd,
    calc_national_wd_demand,
    save_df,
)


# Define a main function
def main():
    
    # start a timer
    start_time = time.time()
    
    # TODO: This will have to be bias corrected for mean and variance
    # TODO: will also have to be regridded to the same grid as the demand data
    # like for the demand from temperature data
    # Set up hard coded variables
    wind_obs_path = "/home/users/benhutch/ERA5/ERA5_wind_daily_1960_2020.nc"
    country = "United Kingdom"
    country_name = "United_Kingdom"

    # assert that this file exist
    assert os.path.exists(wind_obs_path), f"File {wind_obs_path} does not exist."

    # load the file with xarray
    wind_obs = xr.open_dataset(wind_obs_path, chunks={"time":"auto", "latitude": "auto", "longitude": "auto"})
                                                      
    # print the wind obs
    print(wind_obs)

    # Select the data
    wind_obs_10m = wind_obs["si10"]
    wind_obs_100m_bc = wind_obs["si100_bc"]

    # # restrict to the first year
    # # For initial testing
    # wind_obs_10m = wind_obs_10m.sel(time=slice("1960-01-01", "1960-12-31"))
    # wind_obs_100m_bc = wind_obs_100m_bc.sel(time=slice("1960-01-01", "1960-12-31"))

    # apply the country mask for the UK
    wind_obs_10m_uk = apply_country_mask(
        ds=wind_obs_10m,
        country=country,
        lon_name="longitude",
        lat_name="latitude",
    )

    # Apply the country mask for the UK
    wind_obs_100m_bc_uk = apply_country_mask(
        ds=wind_obs_100m_bc,
        country=country,
        lon_name="longitude",
        lat_name="latitude",
    )

    # Calculate the spatial mean
    wind_obs_10m_uk = calc_spatial_mean(
        ds=wind_obs_10m_uk,
        country=country_name,
        variable="si10",
        variable_name="si10",
        convert_kelv_to_cel=False,
    )

    # Calculate the spatial mean
    wind_obs_100m_bc_uk = calc_spatial_mean(
        ds=wind_obs_100m_bc_uk,
        country=country_name,
        variable="si100_bc",
        variable_name="si100_bc",
        convert_kelv_to_cel=False,
    )

    # print the 10m wind speeds
    print(f"10m wind speeds for the UK:", wind_obs_10m_uk)

    # print the 100m wind speeds
    print(f"100m wind speeds for the UK:", wind_obs_100m_bc_uk)


    # # Select the first time step for si10
    # si10 = wind_obs["si10"].isel(time=0)

    # # print the values
    # print(si10.values)

    # # do the same but for si100_bc
    # si100_bc = wind_obs["si100_bc"].isel(time=0)

    # # print the values
    # print(si100_bc.values)

    # plot the 10m wind speeds against the 100m wind speeds for the year
    plt.figure()

    # subset the data by season
    # winter = DJF
    # autum = SON
    # spring = MAM
    # summer = JJA
    winter_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([12, 1, 2])]
    winter_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([12, 1, 2])]

    autumn_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([9, 10, 11])]
    autumn_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([9, 10, 11])]

    spring_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([3, 4, 5])]
    spring_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([3, 4, 5])]

    summer_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([6, 7, 8])]
    summer_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([6, 7, 8])]

    # plot the 10m wind speeds against the 100m wind speeds as a scatter plot
    # with different colours for the different seasons
    # autumn orange
    # spring green
    # summer blue
    # winter purple
    plt.scatter(winter_10m, winter_100m_bc, color="purple", label="Winter")
    plt.scatter(autumn_10m, autumn_100m_bc, color="orange", label="Autumn")
    plt.scatter(spring_10m, spring_100m_bc, color="green", label="Spring")
    plt.scatter(summer_10m, summer_100m_bc, color="blue", label="Summer")

    # set the title
    plt.title("10m vs 100m wind speeds for the UK")

    # set the x-axis label
    plt.xlabel("10m wind speed (m/s)")

    # set the y-axis label
    plt.ylabel("100m wind speed bc (m/s)")

    # include the legend in the top left
    plt.legend(loc="upper left")

    # set up the dir
    # the current dir + testing_plots
    plot_dir = os.path.join(os.getcwd(), "testing_plots")

    # if this does not exist, create it
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # set up the current time
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # save the plot
    plt.savefig(os.path.join(plot_dir, "scatter_plot_10m_100m_wind_speeds_" + current_time + ".png"))

    # end the timer
    print(f"Time taken: {time.time() - start_time} seconds.")

    # print that we are done
    print("Done.")
    sys.exit(0)

    return

if __name__ == "__main__":
    main()