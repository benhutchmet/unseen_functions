#!/usr/bin/env python

"""
process_model_demand.py
=======================

Script to process the demand data for the model, over all months (11-10) for a given lead time.

Based on scripts of Hannah Bloomfield.

Usage:
------
    python process_model_demand.py <model> <variable> <first_year> <last_year> <lead_time> <country>

    model: str
        The model to process.
    variable: str
        The variable to process.
    first_year: int
        The first year to process.
    last_year: int
        The last year to process.
    lead_time: int
        The lead time to process.
    country: str
        The country to process.
        In format "United Kingdom" 
    
Example:
--------
    python process_model_demand.py --model "HadGEM3-GC31-MM" --variable "tas" --first_year 1960 --last_year 2018 --lead_time 1 --country "United Kingdom"

Output:

    The processed demand data for the model, variable, lead time and country specified is saved in the directory:
    ??????

"""

# Import local modules
import glob
import os
import sys
import re
import argparse
import time

# Import third-party modules
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Import the functions
from load_wind_functions import apply_country_mask

from functions_demand import (
    calc_spatial_mean,
    calc_hdd_cdd,
    calc_national_wd_demand,
    save_df,
)

# Import the dictionaries
import unseen_dictionaries as dicts


# set up the main function
def main():
    # Set up the hard coded variables
    base_dir = "/work/scratch-nopw2/benhutch/test_nc/"
    variable_saved = "__xarray_dataarray_variable__"
    months_list = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lon_name = "lon"
    lat_name = "lat"

    # start the timer
    start_time = time.time()

    # set up the parser
    parser = argparse.ArgumentParser(
        description="Process the demand data for the model, over all months (11-10) for a given lead time."
    )

    # add the arguments
    parser.add_argument("--model", type=str, help="The model to process.")

    parser.add_argument("--variable", type=str, help="The variable to process.")

    parser.add_argument("--first_year", type=int, help="The first year to process.")

    parser.add_argument("--last_year", type=int, help="The last year to process.")

    parser.add_argument("--lead_time", type=int, help="The lead time to process.")

    parser.add_argument("--country", type=str, help="The country to process.")

    # parse the arguments
    args = parser.parse_args()

    # If the arguments are not the type expected, raise an error
    if not isinstance(args.model, str):
        raise ValueError("model must be a string.")

    if not isinstance(args.variable, str):
        raise ValueError("variable must be a string.")

    if not isinstance(args.first_year, int):
        raise ValueError("first_year must be an integer.")

    if not isinstance(args.last_year, int):
        raise ValueError("last_year must be an integer.")

    if not isinstance(args.lead_time, int):
        raise ValueError("lead_time must be an integer.")

    if not isinstance(args.country, str):
        raise ValueError("country must be a string.")

    # if country contains a space, replace it with an underscore
    if " " in args.country:
        country_name = args.country.replace(" ", "_")
    else:
        country_name = args.country

    # Set up the empty dataframe
    combined_df = pd.DataFrame()

    # Loop over the months
    for month_idx in tqdm(months_list, desc="Loading months"):
        # Set up the fname
        fname = (
            f"{args.variable}_bias_corrected_{args.model}"
            f"_lead{args.lead_time}_month{month_idx}_"
            f"init{args.first_year}-{args.last_year}.nc"
        )

        # if the file does not exist, continue
        if not os.path.exists(os.path.join(base_dir, fname)):
            raise FileNotFoundError(f"File {fname} not found in {base_dir}")
        
        # Apply the country mask to the data
        model_mon_this = apply_country_mask(
            ds=xr.open_dataset(os.path.join(base_dir, fname)),
            country=args.country,
            lon_name=lon_name,
            lat_name=lat_name,
        )

        # Calculate the spatial mean
        model_mon_this = calc_spatial_mean(
            ds=model_mon_this,
            country=country_name,
            variable=variable_saved,
            variable_name=args.variable,
            convert_kelv_to_cel=True,
        )