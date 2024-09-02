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

# Import external modules
import numpy as np
import xarray as xr
import iris
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

    # Select the variable and a member from the model ds
    model_data_member = model_ds[model_var_name].sel(member=member_name)

    # Rename the variable to something acceptable for iris
    model_data_member = model_data_member.rename(rename_var)

    # Convert the model data to an iris cube
    member_cube = model_data_member.squeeze().to_iris()

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
    member_cube.coord("longitude").attributes = obs_cube[0].coord("longitude").attributes

    # Regrid the observed data to the model grid
    obs_cube_rg = obs_cube[0].regrid(member_cube, iris.analysis.Linear())

    return obs_cube_rg


# Main function for testing
def main():
    # start a timer
    start = time.time()
    
    # Define the paths to the observed and model data
    obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_msl_daily_1960_2020_daymean.nc"
    model_path = "/work/scratch-nopw2/benhutch/test_nc/psl_bias_correction_HadGEM3-GC31-MM_lead1_month11_init1960-1960.nc"

    # Regrid the observed data to the model grid
    obs_cube_rg = regrid_obs_to_model(obs_path, model_path)

    # # Print the cube
    print("Regridded observed data:", obs_cube_rg)

    # # end the timer
    end = time.time()

    # # Print the time taken
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main()