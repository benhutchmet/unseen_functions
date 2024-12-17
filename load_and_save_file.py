#!/usr/bin/env python
"""
load_and_save_file.py
=====================

This script is intended to be used to load in all of the DePreSys data
and save as an array for purposes of bootstrapping.

E.g. for s1960-s2019, 10 winters years, 10 members, the dimensions would be:

(60, 10, 10, 72, 144)

Where 60 is the number of years, 10 is the number of winter years/lead years,
10 is the number of ensemble members, and 72, 144 are the lat/lon dimensions.

Usage:
------
    python load_and_save_file.py <first_year> <last_year> <lead_years> <model>

    first_year: int
        The first year to process.
    last_year: int
        The last year to process.
    model: str
        The model to process.
    variable: str
        The variable to process.

Example:
--------
    python load_and_save_file.py 1960 2019 10 "HadGEM3-GC31-MM" "psl"

Output:
-------
    The processed np.array is saved in the directory:
    /gws/nopw/j04/canari/users/benhutch/saved_DePre/HadGEM3-GC31-MM/psl/

"""

# Local imports
import os
import sys
import time
import argparse
import calendar

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import iris

# Specific imports
from tqdm import tqdm
from datetime import datetime, timedelta

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
import functions as funcs
import bias_adjust as ba

# Function to get the last day of the month
def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


# Define a function to extract the numeric part
def extract_numeric(member_str):
    try:
        return int(member_str[1:-3])  # Adjust slicing to match the format 'rX'
    except ValueError:
        return np.nan  # Return NaN for non-standard entries

# Define the main function
def main():
    # Start the timer
    start = time.time()

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process UNSEEN data.")
    parser.add_argument(
        "--first_year", type=int, help="The first year of the period (e.g. 1960)."
    )
    parser.add_argument(
        "--last_year", type=int, help="The last year of the period (e.g. 2014)."
    )
    parser.add_argument("--model", type=str, help="The model name (e.g. HadGEM3-GC31-MM).")
    parser.add_argument("--variable", type=str, help="The variable name (e.g. tas).")
    parser.add_argument("--season", type=str, help="The season name (e.g. ONDJFM).")
    parser.add_argument("--lead_year", type=str, help="The lead year (e.g. 1-9).")

    # # set up the hard coded args
    # model = "HadGEM3-GC31-MM"
    experiment = "dcppA-hindcast"
    freq = "Amon" # go back to using monthly data

    # set up the save directory
    save_dir = "/gws/nopw/j04/canari/users/benhutch/saved_DePre"

    # if the save directory does not exist
    if not os.path.exists(save_dir):
        # make the directory
        os.makedirs(save_dir)

    # Parse the arguments
    args = parser.parse_args()

    # print the arguments
    print(f"First year: {args.first_year}")
    print(f"Last year: {args.last_year}")
    print(f"Model: {args.model}")
    print(f"Variable: {args.variable}")
    print(f"Season: {args.season}")

    if args.model in ["CanESM5", "BCC-CSM2-MR"]:
        # assert that if model is CanESM5, lead year is "1-9"
        assert args.lead_year == "1-9", "For CanESM5, lead year must be 1-9"
    elif args.model == "HadGEM3-GC31-MM":
        assert args.lead_year == "1-10", "For HadGEM3-GC31-MM, lead year must be 1-10"

    # Set up the months depending on the season
    if args.season == "DJF":
        months = [12, 1, 2]
    elif args.season == "NDJ":
        months = [11, 12, 1]
    elif args.season == "OND":
        months = [10, 11, 12]
    elif args.season == "JFM":
        months = [1, 2, 3]
    elif args.season == "MAM":
        months = [3, 4, 5]
    elif args.season == "JJA":
        months = [6, 7, 8]
    elif args.season == "SON":
        months = [9, 10, 11]
    elif args.season == "ONDJFM":
        months = [10, 11, 12, 1, 2, 3]
    elif args.season == "NDJFM":
        months = [11, 12, 1, 2, 3]
    else:
        raise ValueError("Season not recognised")

    # Load the model ensemble
    model_ds = funcs.load_model_data_xarray(
        model_variable=args.variable,
        model=args.model,
        experiment=experiment,
        start_year=args.first_year,
        end_year=args.last_year,
        first_fcst_year=int(args.first_year) + 1,
        last_fcst_year=int(args.first_year) + 2,
        months=months,
        frequency=freq,
        parallel=True,
    )

    # print that we have loaded the model data
    print("Loaded the model data")

    # # Get the size of the model data in bytes
    size_in_bytes = model_ds[args.variable].size * model_ds[args.variable].dtype.itemsize

    # # Convert bytes to gigabytes
    size_in_gb = size_in_bytes / (1024 ** 3)

    # # Print the size
    print(f"Model data size: {size_in_gb} GB")

    # print the model cube
    print(model_ds)

    # # Modify member coordiante before conbersion to iris
    # model_ds["member"] = model_ds["member"].str[1:-6].astype(int)

    # # Apply the function to the member array and filter out NaN values
    # model_ds["member"] = model_ds["member"].apply(extract_numeric).dropna().astype(int)

    # convert to an iris cube
    model_cube = model_ds[args.variable].squeeze().to_iris()

    # Set an intermediate timer
    intermediate = time.time()

    # Extract the data from the model
    model_data = model_cube.data
    lats = model_cube.coord("latitude").points
    lons = model_cube.coord("longitude").points
    members = model_cube.coord("member").points
    leads = model_cube.coord("lead").points
    years = model_cube.coord("init").points

    # stop the intermediate timer
    intermediate_time = intermediate - start

    # print the time taken
    print(f"Time taken to extract data: {intermediate_time} seconds")

    # print the model data shape
    print(f"Model data shape: {model_data.shape}")

    # print the model data values
    print(f"Model data values: {model_data}")

    # esnure that model data is an array not a masked array
    model_data = np.ma.filled(model_data, np.nan)

    # Set up the save directory
    save_dir = os.path.join(save_dir, args.model, args.variable, freq, args.season, f"{args.first_year}-{args.last_year}")

    # print the save directory
    print(f"Save directory: {save_dir}")

    # if the save directory does not exist
    if not os.path.exists(save_dir):
        # make the directory
        os.makedirs(save_dir)

    # Set up the fnames
    fname_model_data = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}.npy"
    fname_lats = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}_lats.npy"
    fname_lons = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}_lons.npy"
    fname_members = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}_members.npy"
    fname_leads = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}_leads.npy"
    fname_years = f"{args.model}_{args.variable}_{args.season}_{freq}_{args.first_year}-{args.last_year}_years.npy"

    # Save the model data
    with tqdm(total=model_data.size, desc="Saving model data") as pbar:
        np.save(os.path.join(save_dir, fname_model_data), model_data)
        pbar.update(model_data.size)

    # Save the lats
    with tqdm(total=lats.size, desc="Saving lats") as pbar:
        np.save(os.path.join(save_dir, fname_lats), lats)
        pbar.update(lats.size)

    # Save the lons
    with tqdm(total=lons.size, desc="Saving lons") as pbar:
        np.save(os.path.join(save_dir, fname_lons), lons)
        pbar.update(lons.size)

    # Save the members
    with tqdm(total=members.size, desc="Saving members") as pbar:
        np.save(os.path.join(save_dir, fname_members), members)
        pbar.update(members.size)

    # Save the leads
    with tqdm(total=leads.size, desc="Saving leads") as pbar:
        np.save(os.path.join(save_dir, fname_leads), leads)
        pbar.update(leads.size)

    # Save the years
    with tqdm(total=years.size, desc="Saving years") as pbar:
        np.save(os.path.join(save_dir, fname_years), years)
        pbar.update(years.size)

    # print that we have saved the model data
    print("Saved the model data")

    return

if __name__ == "__main__":
    main()