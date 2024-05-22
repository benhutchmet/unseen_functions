"""
Functions used for calculating the wind power output from the wind speed data.

In the first instance, we just want to do this for onshore and offshore wind,
for the UK (and for ERA5 reanalysis data).

Then we want to extend this to use the decadal prediction systems (daily data),
converting wind speeds to 100m level and bias correcting, before passing through the daily power curves.

Methods are much like those used by Hannah Bloomfield and Laura Dawkins (NIC report)
"""
# Imports
import os
import glob
import sys

# Third-party libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Write a function to load in the 
