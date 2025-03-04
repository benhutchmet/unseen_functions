#!/usr/bin/env python

"""
functions_regr.py
~~~~~~~~~~~~~~~~~

This module contains the functions used in the regression analysis.
The stages of this are as follows:

1. Find the appropriate CLEARHEADS/S2S4E wind and demand data
    to quantify the demand net wind variable for a given country
    and time period (e.g., UK, ONDJFM, 1950-2020).

2. Quantify a deterministic multi-linear regression model to map
    2m temperature and 10m wind speed over a given country to
    the demand net wind variable.

3. Quantify a probabilistic multi-linear regression model to map
    2m temperature and 10m wind speed over a given country to
    the demand net wind variable by including a residual term
    to represent the uncertainty in the demand net wind variable.

4. ...

Usage:
~~~~~~

    $ python functions_regr.py --country_code UK --season ONDJFM --start 1950 --end 2020

Arguments:
~~~~~~~~~

    --country_code: str
        The country code for which the analysis is conducted.
    --season: str
        The season for which the analysis is conducted.
    --start: int
        The start year for the analysis.
    --end: int
        The end year for the analysis.

Returns:
~~~~~~~~

    Scikit-learn models for the deterministic and probabilistic
    regression analyses.

"""

# Import local libraries
import os
import sys
import glob
import time
import random
import argparse

# Import third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Specific imports
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, norm
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from typing import Any, List, Tuple


# Define a function for creating the demand net wind dataframe
def create_dnw_df(
    country_code: str,
    season: str,
    start_year: int,
    end_year: int,
    ch_dir: str = "/home/users/benhutch/CLEARHEADS_EU_Power_Data",
    s2s4e_dir: str = "/home/users/benhutch/ERA5_energy_update/",
    s2s4e_demand_fname: str = "ERA5_weather_dependent_demand_1979_2018.csv",
    power_system_stats_path="/home/users/benhutch/unseen_functions/power_system_stats/power_system_stats_copy_raw.csv",
) -> pd.DataFrame:
    """
    Create the demand net wind dataframe for a given country and time period.

    Parameters
    ----------

    country_code: str
        The country for which the analysis is conducted. E.g., "UK", "DE", "FR".
    season: str
        The season for which the analysis is conducted.
    start_year: int
        The start year for the analysis.
    end_year: int
        The end year for the analysis.
    ch_dir: str
        The directory for the CLEARHEADS data.
    s2s4e_dir: str
        The directory for the S2S4E data.
    s2s4e_demand_fname: str
        The filename for the S2S4E demand data.
    power_system_stats_path: str
        The path to the power system statistics file.

    Returns
    -------

    df_monthly: pd.DataFrame
        The demand net wind dataframe. With mean values for the provided season.

    """

    # extract the current date right now
    current_date = time.strftime("%Y-%m-%d")

    # set up a fname for the df
    save_fname = f"{country_code}_CLEARHEADS_S2S4E_{start_year}-01-01_{end_year}-12-31_{current_date}.csv"

    # Set up the directory for the output
    saved_df_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # load the power system stats
    power_system_stats = pd.read_csv(power_system_stats_path)

    # trim the whitespace from the country code column
    power_system_stats["NUTS ID"] = power_system_stats["NUTS ID"].str.strip()

    # assert that the country code is in the power system stats "NUTS ID" column
    assert (
        country_code in power_system_stats["NUTS ID"].values
    ), f"Country code {country_code} not in power system stats."

    # strip any whitespace before or after the str from the column names
    power_system_stats["Country name"] = power_system_stats["Country name"].str.strip()

    # Find the full name of the country from the country code
    # which is the "Country name" column in the power_system_stats df
    country_name = power_system_stats[power_system_stats["NUTS ID"] == country_code][
        "Country name"
    ].values[0]

    # if the country name contains a space
    if " " in country_name:
        # replace the space with an underscore
        country_name = country_name.replace(" ", "_")

    # print the country name
    print(f"Country name: {country_name}")

    # if the saved_df_dir does not exist
    if not os.path.exists(os.path.join(saved_df_dir, save_fname)):
        print(f"File {save_fname} does not exist.")
        print(f"Creating file {save_fname}.")
        print(f"Loading data for {country_code}.")
        print(f"Between {start_year} and {end_year}.")

        # All use CLEARHEADS for wind/solar components
        # and country agg variables (e.g. 10m wind speed, 2m temperature)
        # Use S2S4E for demand data
        # First hardcode the list of files to extract
        # FIXME: Simplify for speed
        files_to_extract = [
            # "NUTS_0_sp_historical.nc",
            # "NUTS_0_speed100m_historical.nc",
            "NUTS_0_speed10m_historical.nc",
            # "NUTS_0_ssrd_historical.nc",
            "NUTS_0_wp_ofs_sim_0_historical_loc_weighted.nc",
            "NUTS_0_wp_ons_sim_0_historical_loc_weighted.nc",
        ]

        # Create a list with the names
        names = [
            # "solar_power",
            # "100m_wind_speed",
            "10m_wind_speed",
            # "solar_irradiance",
            "ofs_cfs",
            "ons_cfs",
        ]

        # Set up the trend levels to extract for the temperature data
        temp_fname = "NUTS_0_t2m_detrended_timeseries_historical.nc"
        trend_levels = [0, 2020]

        # Set up an empty dataframe to load the clearheads data
        ch_df = pd.DataFrame()

        # Loop over the files to extract
        for file, name in tqdm(
            zip(files_to_extract, names), total=len(files_to_extract)
        ):
            # Form the complete path
            path = os.path.join(ch_dir, file)

            # assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist."

            # Load the data with xarray
            data = xr.open_dataset(path)

            # Turn the data into a df
            data_df_this = data.to_dataframe().reset_index()

            # Pivot the dataframe with hard coded params
            data_df_this_pivot = data_df_this.pivot(
                index="time_in_hours_from_first_jan_1950",
                columns="NUTS",
                values="timeseries_data",
            )

            # Set the columns as the NUTS keys
            data_df_this_pivot.columns = data["NUTS_keys"].values

            # Convert time in hours from first Jan 1950 to datetime
            data_df_this_pivot.index = pd.to_datetime(
                data_df_this_pivot.index, unit="h", origin="1950-01-01"
            )

            # If the country code column exists
            if country_code in data_df_this_pivot.columns:
                # Extract the country data
                data_df_this_country = data_df_this_pivot[country_code]
            else:
                raise ValueError(f"Country code {country_code} not in data.")

            # if this is the first iteration set the index
            if ch_df.empty:
                ch_df = data_df_this_country.to_frame(name=name)
            else:
                ch_df[name] = data_df_this_country

        # # Print the head of this dataframe
        # print(ch_df.head())

        # Form the filepath for the temperature data
        temp_path = os.path.join(ch_dir, temp_fname)

        # assert that the path exists
        assert os.path.exists(temp_path), f"Path {temp_path} does not exist."

        # Load the temperature data with xarray
        data_t2m = xr.open_dataset(temp_path)

        # Loop over the trend levels
        for tl in trend_levels:
            # Extract the trend_levels from the data
            trend_levels_file = data_t2m.trend_levels.values

            # Find the index of the trend level
            idx_tl = np.where(trend_levels_file == tl)[0][0]

            # Extract the data for this trend level
            data_t2m_this = data_t2m.isel(trend=idx_tl)

            # Turn the data into a df and pivot
            df_t2m_pivot = (
                data_t2m_this.to_dataframe()
                .reset_index()
                .pivot(
                    index="time_in_hours_from_first_jan_1950",
                    columns="NUTS",
                    values="detrended_data",
                )
            )

            # Set the columns as the NUTS keys
            df_t2m_pivot.columns = data_t2m_this["NUTS_keys"].values

            # Convert time in hours from first Jan 1950 to datetime
            df_t2m_pivot.index = pd.to_datetime(
                df_t2m_pivot.index, unit="h", origin="1950-01-01"
            )

            # If the country code column exists
            if country_code in df_t2m_pivot.columns:
                # Extract the country data
                df_t2m_this_country = df_t2m_pivot[country_code]
            else:
                raise ValueError(f"Country code {country_code} not in data.")

            # add the column to the ch_df
            ch_df[f"t2m_{tl}_dt"] = df_t2m_this_country

        # # Print the head of this dataframe
        # print(ch_df.head())

        # load the demand data
        df_demand = pd.read_csv(os.path.join(s2s4e_dir, s2s4e_demand_fname))

        # rename "Unnamed: 0" to "time"
        df_demand = df_demand.rename(columns={"Unnamed: 0": "time"})

        # set the time column as a datetime
        df_demand["time"] = pd.to_datetime(df_demand["time"])

        # set the time column as the index
        df_demand = df_demand.set_index("time")

        # reset all of the column names by splitting by the "_"
        df_demand.columns = df_demand.columns.str.split("_").str[-11]

        # print all of the column names
        # rename kingdom to "United_Kingdom"
        df_demand = df_demand.rename(columns={"Kingdom": "United_Kingdom"})

        # subset the demand data to the country
        df_demand_country = df_demand[country_name]

        # rename the column to "wd_demand"
        df_demand_country = df_demand_country.rename(
            f"{df_demand_country.name}_wd_demand"
        )

        # collapse the df_ch into daily means
        ch_df_daily = ch_df.resample("D").mean()

        # subset the ch_df to the start and end years
        ch_df_daily = ch_df_daily.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

        # subset the df_demand_country to the start and end years
        df_demand_country = df_demand_country.loc[
            f"{start_year}-01-01":f"{end_year}-12-31"
        ]

        # merge the ch_df_daily and df_demand_country
        df = pd.merge(ch_df_daily, df_demand_country, left_index=True, right_index=True)

        # print the head of the df
        print(df.head())

        # print the tail of the df
        print(df.tail())

        # if the full path does not exist
        if not os.path.exists(os.path.join(saved_df_dir, save_fname)):
            # save the df to the saved_df_dir
            df.to_csv(os.path.join(saved_df_dir, save_fname), index=True)
        else:
            # print that the file already exists
            print(f"File {save_fname} already exists.")
    else:
        print(f"File {save_fname} already exists.")
        print(f"Loading file {save_fname}.")

        # load the df from the saved_df_dir
        df = pd.read_csv(os.path.join(saved_df_dir, save_fname))

        # print the head of the df
        print(df.head())

        # print the values of time_in_hours_from_first_jan_1950
        print(df["time_in_hours_from_first_jan_1950"].values)

        # convert the index to a datetime
        df["time_in_hours_from_first_jan_1950"] = pd.to_datetime(
            df["time_in_hours_from_first_jan_1950"]
        )

        # set the time column as the index
        df = df.set_index("time_in_hours_from_first_jan_1950")

    print(f"Dataframe shape: {df.shape}")

    print(f"Dataframe columns: {df.columns}")

    # if the season is ONDJFM, then subset the df to the months
    if season == "ONDJFM":
        # subset the df to the months
        df = df[df.index.month.isin([10, 11, 12, 1, 2, 3])]
    elif season == "AMJJAS":
        # subset the df to the months
        df = df[df.index.month.isin([4, 5, 6, 7, 8, 9])]
    elif season == "DJF":
        # subset the df to the months
        df = df[df.index.month.isin([12, 1, 2])]
    elif season == "MAM":
        # subset the df to the months
        df = df[df.index.month.isin([3, 4, 5])]
    elif season == "JJA":
        # subset the df to the months
        df = df[df.index.month.isin([6, 7, 8])]
    elif season == "SON":
        # subset the df to the months
        df = df[df.index.month.isin([9, 10, 11])]
    else:
        raise ValueError(f"Season {season} not recognised.")

    # # print power system stats .head
    # print(power_system_stats.head())

    # sys.exit()

    # Extract the values of installed onshore and offshore wind
    # Columns: "Installed wind (onshore, MW)" and "Installed wind (onshore, MW)"
    # in power system stats
    installed_onshore = power_system_stats[
        power_system_stats["NUTS ID"].values == country_code
    ]["Installed wind (onshore, MW)"].values[0]

    installed_offshore = power_system_stats[
        power_system_stats["NUTS ID"].values == country_code
    ]["Installed wind (offshore, MW)"].values[0]

    # print the installed onshore and offshore wind
    print(f"Installed onshore wind: {installed_onshore} MW for {country_code}")
    print(f"Installed offshore wind: {installed_offshore} MW for {country_code}")

    # Calculate the wind power generation for the country
    df[f"wind_gen_{country_code}"] = (installed_onshore / 1000) * df["ons_cfs"] + (
        installed_offshore / 1000
    ) * df[
        "ofs_cfs"
    ]  # in GW

    # Calculate the demand net wind variable
    df["wd_demand_net_wind"] = (
        df[f"{country_name}_wd_demand"] - df[f"wind_gen_{country_code}"]
    )

    # resample into monthly means
    df_monthly = df.resample("ME").mean()

    # drop any of the NaN rows
    df_monthly = df_monthly.dropna()

    # if the season is in ["ONDJFM", "DJFM"]
    if season in ["ONDJFM", "DJFM", "NDJFM"]:
        # shift back back 3 months
        df_monthly = df_monthly.shift(-3, freq="ME").resample("YE").mean()

        # drop the first and last year
        df_monthly = df_monthly.iloc[1:-1]
    elif season in ["DJF", "ONDJF", "NDJF"]:
        # shift back 2 months
        df_monthly = df_monthly.shift(-2, freq="ME").resample("YE").mean()

        # drop the first and last year
        df_monthly = df_monthly.iloc[1:-1]
    elif season in ["DJ", "ONDJ", "NDJ"]:
        # shift back 1 month
        df_monthly = df_monthly.shift(-1, freq="ME").resample("YE").mean()

        # drop the first and last year
        df_monthly = df_monthly.iloc[1:-1]
    else:
        # take the annual mean
        df_monthly = df_monthly.resample("YE").mean()

    # set the index just to the year
    df_monthly.index = df_monthly.index.year

    return df_monthly


# Simple plotting function for two variables
def plot_scatter_with_fit(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str
) -> None:
    """
    This function plots a scatter plot between two columns of a DataFrame and includes a line of best fit.
    It also calculates and displays the r2 value for the fit.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    # plot a scatter between x_col and y_col
    plt.scatter(df[x_col], df[y_col])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # include a line of best fit
    m, b = np.polyfit(df[x_col], df[y_col], 1)

    plt.plot(df[x_col], m * df[x_col] + b, color="k")

    # calculate the r2 value
    r2 = np.corrcoef(df[x_col], df[y_col])[0, 1] ** 2

    # text in the top left with r2
    plt.text(
        0.05,
        0.95,
        f"r2 = {r2:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # set the title
    plt.title(title)

    plt.show()

    return None

# Form an MLR model and plot as a 3D scatter plot
def plot_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    """
    This function plots a 3D scatter plot between three columns of a DataFrame and includes a plane of best fit.
    It also calculates and displays the R-squared value for the fit.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        z_col (str): The name of the column to use for the z-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        zlabel (str): The label for the z-axis.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(df[x_col], df[y_col], df[z_col])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Fit a multiple linear regression model
    X = df[[x_col, y_col]]
    y = df[z_col]
    model = LinearRegression().fit(X, y)

    # Calculate the R-squared value
    r2 = model.score(X, y)

    # calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

    # Print the R-squared value
    print(f"R-squared: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot the regression plane
    x_range = np.linspace(X[x_col].min(), X[x_col].max(), num=10)
    y_range = np.linspace(X[y_col].min(), X[y_col].max(), num=10)
    x_range, y_range = np.meshgrid(x_range, y_range)
    z_range = model.predict(np.array([x_range.flatten(), y_range.flatten()]).T).reshape(
        x_range.shape
    )
    ax.plot_surface(x_range, y_range, z_range, alpha=0.5)

    # include the r2 and the rmse in the top left
    ax.text2D(
        0.05,
        0.95,
        f"r2 = {r2:.2f}\nRMSE = {rmse:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.show()

    return None

# Function to test the strength of the deterministic fit
def plot_deterministic_fit(
    df: pd.DataFrame,
    X1_col: str,
    X2_col: str,
    Y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    This function plots the deterministic fit of a multiple linear regression model.
    MLR is fitted to two independent variables (X1 and X2) and one dependent 
    variable (Y). The function plots the actual values of Y against the predicted
    values of Y.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        X1_col (str): The name of the first independent variable.
        X2_col (str): The name of the second independent variable.
        Y_col (str): The name of the dependent variable.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None

    """

    # Set up the X
    X = df[[X1_col, X2_col]]
    y = df[Y_col]

    # Fit the model
    model = LinearRegression().fit(X, y)

    # Calculate the R2 and RMSE values
    r2 = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

    # Print the R2 and RMSE values
    print(f"R-squared: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Get the model to predict the values of Y
    y_pred = model.predict(X)

    # set up a figure
    fig = plt.figure(figsize=(10, 5))

    # plot the actual wd demand net wind values
    plt.scatter(df.index, df[Y_col], label="actual", color="k")

    # plot the predicted wd demand net wind values
    plt.plot(df.index, y_pred, label="predicted", color="r")

    # set the title
    plt.title(title)

    # calculate the correlation between the actual and predicted values
    r, p = pearsonr(df[Y_col], y_pred)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(df[Y_col], y_pred))

    # text in the top left with r2
    plt.text(
        0.05,
        0.95,
        f"r = {r:.2f} (p = {p:.2f})"
        f"\nRMSE = {rmse:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # set the x label
    plt.xlabel(xlabel)

    # set the y label
    plt.ylabel(ylabel)

    # show the legend
    plt.legend()

    plt.show()

    return None

# define a function to plot the histogram of the residuals between two
# variables
# e.g. 10m wind speed anbd uk mean wind power generation
# first order fit (i.e. linear regression, y = mx + c)
def plot_residual_hist_first(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    block_length: int = 10,
    nboot: int = 1000,
) -> None:
    """
    This function plots a histogram of the residuals between two columns of a DataFrame.
    The residuals are calculated as the difference between the actual and predicted values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The name of the first column.
        y_col (str): The name of the second column.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        block_length (int): The block length for the bootstrapping.
        nboot (int): The number of bootstrapping trials.

    Returns:
        None

    """

    # get the number of times
    # length of the index
    ntimes = len(df)

    # get the number of blocks
    nblocks = int(ntimes / block_length)

    # if the nblocks * block is less than the ntimes
    if (nblocks * block_length) < ntimes:
        # add one to the nblocks
        nblocks = nblocks + 1

    # set up the index for time
    index_time = range(ntimes - block_length + 1)

    # set up the empty array for the bootstrapped data
    X_boot_full = np.zeros((nboot, ntimes))
    y_boot_full = np.zeros((nboot, ntimes))

    # Set up an empty array for the residuals
    residuals_boot = np.zeros((nboot, ntimes))

    # Set up an empty array for the spread
    res_spread_boot = np.zeros(nboot)

    # loop over the nboot
    for iboot in tqdm(np.arange(nboot)):
        # Select starting indices for the blocks
        if iboot == 0:
            ind_time_this = range(0, ntimes, block_length)
        else: # random samples
            ind_time_this = np.array([random.choice(index_time) for _ in range(nblocks)])

        # Set up the shape of the bootstrapped data
        X_boot = np.zeros(ntimes)

        # Same for the predictand
        y_boot = np.zeros(ntimes)

        # reset time index
        itime = 0

        # loop over the indices
        for ithis in ind_time_this:
            # Set up the block index
            ind_block = np.arange(ithis, ithis + block_length)

            # if the block index is greater than the number of times
            # then subtract the number of times from the block index
            ind_block[(ind_block>ntimes-1)] = ind_block[(ind_block>ntimes-1)]-ntimes

            # Restrict the block index to the minimum of the block length
            ind_block = ind_block[:min(block_length,ntimes-itime)]

            # loop over the blocks
            for iblock in ind_block:
                # Set up the bootstrapped data
                X_boot[itime] = df[x_col].values[iblock]
                y_boot[itime] = df[y_col].values[iblock]

                # increment the time index
                itime += 1

        # Append the data
        X_boot_full[iboot, :] = X_boot
        y_boot_full[iboot, :] = y_boot

        # print the shape of the bootstrapped data
        # # print(X_boot_full.shape)
        # # print(y_boot_full.shape)
        # print(np.shape(X_boot))
        # print(np.shape(y_boot))

        # Fit the model
        m, b = np.polyfit(X_boot, y_boot, 1)

        # Predict the values of Y
        y_pred = m * X_boot + b

        # Calculate the residuals
        # the difference between the actual and predicted values
        residuals_boot[iboot, :] = y_pred - y_boot

        # Calculate the spread of the residuals
        res_spread_boot[iboot] = np.std(residuals_boot[iboot, :])

    # Plot a histogram of the bootstrapped residuals
    plt.hist(residuals_boot.flatten(), bins=30, color="k", alpha=0.5)

    # Include a solid line for the mean
    plt.axvline(np.mean(residuals_boot), color="r", linestyle="--")

    # Include two dashed red lines for the 5th and 95th percentiles
    plt.axvline(np.percentile(residuals_boot, 5), color="r", linestyle="--")
    plt.axvline(np.percentile(residuals_boot, 95), color="r", linestyle="--")

    # Set the title
    plt.title(title)

    # Set the x label
    plt.xlabel(xlabel)

    # Set the y label
    plt.ylabel(ylabel)

    plt.show()

    # print the mean, 5th and 95th percentiles of the residuals spread
    print(f"Mean residual spread: {np.mean(res_spread_boot):.2f}")
    print(f"5th percentile residual spread: {np.percentile(res_spread_boot, 5):.2f}")
    print(f"95th percentile residual spread: {np.percentile(res_spread_boot, 95):.2f}")

    return None

# define a function to plot the histogram of the residuals
# for a multiple linear regression model
def plot_residual_hist_mlr(
    df: pd.DataFrame,
    X1_col: str,
    X2_col: str,
    Y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    block_length: int = 10,
    nboot: int = 1000,
) -> None:
    """

    This function plots a histogram of the residuals for a multiple linear regression model.
    The residuals are calculated as the difference between the actual and predicted values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        X1_col (str): The name of the first independent variable.
        X2_col (str): The name of the second independent variable.
        Y_col (str): The name of the dependent variable.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        block_length (int): The block length for the bootstrapping.
        nboot (int): The number of bootstrapping trials.

    Returns:
        None

    """

    # Get the number of times
    ntimes = len(df)

        # get the number of blocks
    nblocks = int(ntimes / block_length)

    # if the nblocks * block is less than the ntimes
    if (nblocks * block_length) < ntimes:
        # add one to the nblocks
        nblocks = nblocks + 1

    # set up the index for time
    index_time = range(ntimes - block_length + 1)

    # set up the empty array for the bootstrapped data
    X1_boot_full = np.zeros((nboot, ntimes))
    X2_boot_full = np.zeros((nboot, ntimes))
    Y_boot_full = np.zeros((nboot, ntimes))

    # Set up an empty array for the residuals
    residuals_boot = np.zeros((nboot, ntimes))

    # Set up an empty array for the spread
    res_spread_boot = np.zeros(nboot)

    # set up an empty array for the r2 and rmse values
    r2_boot = np.zeros(nboot)
    rmse_boot = np.zeros(nboot)

    # loop over the nboot
    for iboot in tqdm(np.arange(nboot)):
        # Select starting indices for the blocks
        if iboot == 0:
            ind_time_this = range(0, ntimes, block_length)
        else: # random samples
            ind_time_this = np.array([random.choice(index_time) for _ in range(nblocks)])

        # Set up the shape of the bootstrapped data
        X1_boot = np.zeros(ntimes)
        X2_boot = np.zeros(ntimes)

        # Same for the predictand
        Y_boot = np.zeros(ntimes)

        # reset time index
        itime = 0

        # loop over the indices
        for ithis in ind_time_this:
            # Set up the block index
            ind_block = np.arange(ithis, ithis + block_length)

            # if the block index is greater than the number of times
            # then subtract the number of times from the block index
            ind_block[(ind_block>ntimes-1)] = ind_block[(ind_block>ntimes-1)]-ntimes

            # Restrict the block index to the minimum of the block length
            ind_block = ind_block[:min(block_length,ntimes-itime)]

            # loop over the blocks
            for iblock in ind_block:
                # Set up the bootstrapped data
                X1_boot[itime] = df[X1_col].values[iblock]
                X2_boot[itime] = df[X2_col].values[iblock]
                Y_boot[itime] = df[Y_col].values[iblock]

                # increment the time index
                itime += 1

        # Append the data
        X1_boot_full[iboot, :] = X1_boot
        X2_boot_full[iboot, :] = X2_boot
        Y_boot_full[iboot, :] = Y_boot

        # print the shape of the bootstrapped data
        # # print(X_boot_full.shape)
        # # print(y_boot_full.shape)
        # print(np.shape(X_boot))
        # print(np.shape(y_boot))

        # # # print the shape of X1_boot and X2_boot
        # print(np.shape(X1_boot))
        # print(np.shape(X2_boot))

        # Set up the predictors
        X_boot = np.column_stack((X1_boot, X2_boot))

        # Fit the model
        model = LinearRegression().fit(X_boot, Y_boot)

        # predict the values of Y
        Y_pred = model.predict(X_boot)

        # calculate and append the r2 and rmse values
        r2_boot[iboot] = model.score(X_boot, Y_boot)
        rmse_boot[iboot] = np.sqrt(mean_squared_error(Y_boot, Y_pred))

        # Calculate the residuals
        # the difference between the actual and predicted values
        residuals_boot[iboot, :] = Y_pred - Y_boot

        # Calculate the spread of the residuals
        res_spread_boot[iboot] = np.std(residuals_boot[iboot, :])

    # Plot a histogram of the bootstrapped residuals
    plt.hist(residuals_boot.flatten(), bins=30, color="k", alpha=0.5)

    # Include a solid line for the mean
    plt.axvline(np.mean(residuals_boot), color="r", linestyle="--")

    # Include two dashed red lines for the 5th and 95th percentiles
    plt.axvline(np.percentile(residuals_boot, 5), color="r", linestyle="--")
    plt.axvline(np.percentile(residuals_boot, 95), color="r", linestyle="--")

    # include a title
    plt.title(title)

    # include an x label
    plt.xlabel(xlabel)

    # include a y label
    plt.ylabel(ylabel)

    plt.show()

    # print the mean, 5th and 95th percentiles of the residuals spread
    print(f"Mean residual spread: {np.mean(res_spread_boot):.2f}")
    print(f"5th percentile residual spread: {np.percentile(res_spread_boot, 5):.2f}")
    print(f"95th percentile residual spread: {np.percentile(res_spread_boot, 95):.2f}")

    # print the mean, 5th and 95th percentiles of the r2 values
    print(f"Mean r2: {np.mean(r2_boot):.2f}")
    print(f"5th percentile r2: {np.percentile(r2_boot, 5):.2f}")
    print(f"95th percentile r2: {np.percentile(r2_boot, 95):.2f}")

    # print the mean, 5th and 95th percentiles of the rmse values
    print(f"Mean RMSE: {np.mean(rmse_boot):.2f}")
    print(f"5th percentile RMSE: {np.percentile(rmse_boot, 5):.2f}")
    print(f"95th percentile RMSE: {np.percentile(rmse_boot, 95):.2f}")

    return None


# define a function to plot the stochastic fit
def plot_stochastic_fit(
    df: pd.DataFrame,
    X1_col: str,
    X2_col: str,
    Y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    num_trials: int = 1000,
    sample_uncertainty: bool = False,
    block_length: int = 10,
    nboot: int = 1000,
) -> None:
    """
    This function plots the stochastic fit of a multiple linear regression model.
    MLR is fitted to two independent variables (X1 and X2) and one dependent
    variable (Y). The function plots the residuals of the model.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        X1_col (str): The name of the first independent variable.
        X2_col (str): The name of the second independent variable.
        Y_col (str): The name of the dependent variable.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        num_trials (int): The number of trials to run for the stochastic fit.
        sample_uncertainty (bool): Whether to also include sample uncertainty.
        block_length (int): The block length for the bootstrapping.
        nboot (int): The number of block

    Returns:
        None

    """

    # if sample uncertainty is false
    if not sample_uncertainty:
        print("Sampling uncertainty in the MLR fit via stochastic trials.")
        # Set up the predictors (X)
        X = df[[X1_col, X2_col]]

        # Set up the dependent variable/predictand (Y)
        y = df[Y_col]

        # Fit the model
        model = LinearRegression().fit(X, y)

        # Calculate the R2 and RMSE values
        r2 = model.score(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

        # Print the R2 and RMSE values
        print(f"R-squared: {r2:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Get the model to predict the values of Y
        y_pred = model.predict(X)

        # Calculate the residuals
        # the difference between the actual and predicted values
        residuals = y_pred - y

        # # plot a histogram of the residuals
        # plt.hist(residuals, bins=30, color="k", alpha=0.5)

        # plt.show()

        # calculate the stdev of the residuals
        res_stdev = np.std(residuals)

        # # create num_trials of random time series where the magnitude
        # # of the standard deviation matches the residuals
        # # assuming that the residuals we have are normally distributed
        # # with a mean of 0
        # MLR fit is not very normal!
        stoch = np.random.normal(0, res_stdev, size=(len(df), num_trials))


        # print the shape oif y_pred
        print(f"Shape of y_pred: {np.shape(y_pred)}")
        print(f"Shape of stoch: {np.shape(stoch)}")

        # add the random trials to the deterministic model time series
        # to create a stochastic model
        trials = pd.DataFrame(
            y_pred[:, None] + stoch, index=df.index, columns=range(num_trials)
        )

        # now plot the same deterministic fit as before
        # alongside boxplots of the stochastic fit
        fig, ax = plt.subplots(figsize=(10, 5))

        # plot the actual wd demand net wind values
        ax.scatter(df.index, df[Y_col], label="actual", color="k")

        # plot the predicted wd demand net wind values
        ax.plot(df.index, y_pred, label="predicted", color="r")

        # process the trials data
        model_years_stoch = trials.groupby(df.index).mean()

        # find the 5th and 95th percentiles
        p05, p95 = [model_years_stoch.T.quantile(q) for q in [0.05, 0.95]]

        # plot the 5th and 95th percentiles for the stochastic model
        ax.fill_between(
            model_years_stoch.index,
            p05,
            p95,
            color="r",
            alpha=0.2,
            label="stochastic",
        )
        
        # set the title
        ax.set_title(title)

        # calculate the correlation between the actual and predicted values
        r, p = pearsonr(df[Y_col], y_pred)

        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(df[Y_col], y_pred))

        # text in the top left with r2
        ax.text(
            0.05,
            0.95,
            f"r = {r:.2f} (p = {p:.2f})"
            f"\nRMSE = {rmse:.2f}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.5),
        )
    else:
        print("Sampling uncertainty in the MLR fit via stochastic trials.")
        print("And sample uncertainty in the time series via block bootstrapping.")

        # Get the number of times
        ntimes = len(df)

            # get the number of blocks
        nblocks = int(ntimes / block_length)

        # if the nblocks * block is less than the ntimes
        if (nblocks * block_length) < ntimes:
            # add one to the nblocks
            nblocks = nblocks + 1

        # set up the index for time
        index_time = range(ntimes - block_length + 1)
        
        # set up the empty array for the bootstrapped data
        X1_boot_full = np.zeros((nboot, ntimes))
        X2_boot_full = np.zeros((nboot, ntimes))
        Y_boot_full = np.zeros((nboot, ntimes))

        # Set up an empty array for the residuals
        residuals_boot = np.zeros((nboot, ntimes))

        # Set up an empty array for the spread
        res_spread_boot = np.zeros(nboot)

        # set up an empty array for the r2 and rmse values
        r2_boot = np.zeros(nboot)
        rmse_boot = np.zeros(nboot)

        # loop over the nboot
        for iboot in tqdm(np.arange(nboot)):
            # Select starting indices for the blocks
            if iboot == 0:
                ind_time_this = range(0, ntimes, block_length)
            else: # random samples
                ind_time_this = np.array([random.choice(index_time) for _ in range(nblocks)])

            # Set up the shape of the bootstrapped data
            X1_boot = np.zeros(ntimes)
            X2_boot = np.zeros(ntimes)

            # Same for the predictand
            Y_boot = np.zeros(ntimes)

            # reset time index
            itime = 0

            # loop over the indices
            for ithis in ind_time_this:
                # Set up the block index
                ind_block = np.arange(ithis, ithis + block_length)

                # if the block index is greater than the number of times
                # then subtract the number of times from the block index
                ind_block[(ind_block>ntimes-1)] = ind_block[(ind_block>ntimes-1)]-ntimes

                # Restrict the block index to the minimum of the block length
                ind_block = ind_block[:min(block_length,ntimes-itime)]

                # loop over the blocks
                for iblock in ind_block:
                    # Set up the bootstrapped data
                    X1_boot[itime] = df[X1_col].values[iblock]
                    X2_boot[itime] = df[X2_col].values[iblock]
                    Y_boot[itime] = df[Y_col].values[iblock]

                    # increment the time index
                    itime += 1

            # Append the data
            X1_boot_full[iboot, :] = X1_boot
            X2_boot_full[iboot, :] = X2_boot
            Y_boot_full[iboot, :] = Y_boot

            if iboot == 0:
                ind_time_this = range(0, ntimes, block_length)

                X_boot_first = np.column_stack((X1_boot, X2_boot))

                # print the shape of the bootstrapped data
                print(X_boot_first.shape)

                # Fit the model
                model_first = LinearRegression().fit(X_boot_first, Y_boot)

                # predict the values of Y
                Y_pred_first = model_first.predict(X_boot_first)

                # calculate and append the r2 and rmse values
                r2_boot_first = model_first.score(X_boot_first, Y_boot)

                # Calculate the residuals
                # the difference between the actual and predicted values
                residuals_boot_first = Y_pred_first - Y_boot

                # Calculate the spread of the residuals
                res_spread_boot_first = np.std(residuals_boot_first)
            else: # random samples
                # print the shape of the bootstrapped data
                # # print(X_boot_full.shape)
                # # print(y_boot_full.shape)
                # print(np.shape(X_boot))
                # print(np.shape(y_boot))

                # # # print the shape of X1_boot and X2_boot
                # print(np.shape(X1_boot))
                # print(np.shape(X2_boot))

                # Set up the predictors
                X_boot = np.column_stack((X1_boot, X2_boot))

                # Fit the model
                model = LinearRegression().fit(X_boot, Y_boot)

                # predict the values of Y
                Y_pred = model.predict(X_boot)

                # calculate and append the r2 and rmse values
                r2_boot[iboot] = model.score(X_boot, Y_boot)
                rmse_boot[iboot] = np.sqrt(mean_squared_error(Y_boot, Y_pred))

                # Calculate the residuals
                # the difference between the actual and predicted values
                residuals_boot[iboot, :] = Y_pred - Y_boot

                # Calculate the spread of the residuals
                res_spread_boot[iboot] = np.std(residuals_boot[iboot, :])

        # Quantify the mean and spread of the residuals
        res_stdev_mean = np.mean(res_spread_boot)
        res_stdev_05 = np.percentile(res_spread_boot, 5)
        res_stdev_95 = np.percentile(res_spread_boot, 95)

        # Create three different stochastic fits
        # with different levels of uncertainty
        stoch_mean = np.random.normal(0, res_stdev_mean, size=(len(df), num_trials))
        stoch_05 = np.random.normal(0, res_stdev_05, size=(len(df), num_trials))
        stoch_95 = np.random.normal(0, res_stdev_95, size=(len(df), num_trials))

        # Print the shape of Y_pred_first
        print(f"Shape of Y_pred_first: {np.shape(Y_pred_first)}")

        # Print the shape of stoch_mean
        print(f"Shape of stoch_mean: {np.shape(stoch_mean)}")

        # Add the random trials to the deterministic model time series
        # to create a stochastic model
        trials_mean = pd.DataFrame(
            Y_pred_first[:, None] + stoch_mean, index=df.index, columns=range(num_trials)
        )
        trials_05 = pd.DataFrame(
            Y_pred_first[:, None] + stoch_05, index=df.index, columns=range(num_trials)
        )
        trials_95 = pd.DataFrame(
            Y_pred_first[:, None] + stoch_95, index=df.index, columns=range(num_trials)
        )

        # Limit df to the first 100 rows
        df = df.head(88)

        # limit y_pred_first to the first 100 rows
        Y_pred_first = Y_pred_first[:88]

        # limit trials_mean to the first 100 rows
        trials_mean = trials_mean.head(88)

        # limit trials_05 to the first 100 rows
        trials_05 = trials_05.head(88)

        # limit trials_95 to the first 100 rows
        trials_95 = trials_95.head(88)

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the actual wd demand net wind values
        ax.scatter(df.index, df[Y_col], label="actual", color="k")

        # Plot the predicted wd demand net wind values
        ax.plot(df.index, Y_pred_first, label="predicted", color="r")

        # print the trials mean head
        print(f"Trials mean head: {trials_mean.head()}")

        # print the df index
        print(f"df index: {df.index}")

        # print the shape of trials mean
        print(f"Shape of trials mean: {np.shape(trials_mean)}")

        # print the shape of df index
        print(f"Shape of df index: {np.shape(df.index)}")

        # Process the trials data
        model_years_stoch_mean = trials_mean.groupby(df.index).mean()
        model_years_stoch_05 = trials_05.groupby(df.index).mean()
        model_years_stoch_95 = trials_95.groupby(df.index).mean()

        # print the shape of the model years stoch mean
        print(f"Shape of model years stoch mean: {np.shape(model_years_stoch_mean)}")

        # Find the 5th and 95th percentiles
        p05, p95 = [model_years_stoch_mean.T.quantile(q) for q in [0.05, 0.95]]
        p05_05, p95_05 = [model_years_stoch_05.T.quantile(q) for q in [0.05, 0.95]]
        p05_95, p95_95 = [model_years_stoch_95.T.quantile(q) for q in [0.05, 0.95]]

        # print the shape of the 5th and 95th percentiles
        print(f"Shape of p05: {np.shape(p05)}")
        print(f"Shape of p95: {np.shape(p95)}")

        # # Plot the 5th and 95th percentiles for the stochastic model
        # ax.fill_between(
        #     model_years_stoch_mean.index,
        #     p05,
        #     p95,
        #     color="r",
        #     alpha=0.2,
        #     label="stochastic mean spread",
        # )

        # ax.fill_between(
        #     model_years_stoch_05.index,
        #     p05_05,
        #     p95_05,
        #     color="b",
        #     alpha=0.2,
        #     label="stochastic 5th percentile spread",
        # )

        ax.fill_between(
            model_years_stoch_95.index,
            p05_95,
            p95_95,
            color="r",
            alpha=0.2,
            label="stochastic 95th percentile spread",
        )

        # text in the top left with mean r2 and rmse
        ax.text(
            0.05,
            0.95,
            f"Mean r2 = {np.mean(r2_boot):.2f} (5th = {np.percentile(r2_boot, 5):.2f}, 95th = {np.percentile(r2_boot, 95):.2f})"
            f"\nMean RMSE = {np.mean(rmse_boot):.2f} (5th = {np.percentile(rmse_boot, 5):.2f}, 95th = {np.percentile(rmse_boot, 95):.2f})",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # set the x label
    ax.set_xlabel(xlabel)

    # set the y label
    ax.set_ylabel(ylabel)

    # show the legend
    ax.legend(loc="upper right")

    plt.show()

    return None


# Define a function to load in the observed data
# as processed from the RACC regridded (to DePreSys resolution) ERA5 fields
# for the masked country region
def load_processed_obs_data(
    country_name: str,
    season: str,
    start_year: int,
    end_year: int,
    variable_wind: str = "sfcWind",
    variable_t2m: str = "tas",
    csv_dir: str = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"
) -> pd.DataFrame:
    """
    This function loads in the processed observed data for a given country and season.

    Args:
        country_name (str): The name of the country.
        season (str): The season to load the data for.
        start_year (int): The start year of the data.
        end_year (int): The end year of the data.
        variable_wind (str): The name of the wind variable.
        variable_t2m (str): The name of the temperature variable.
        csv_dir (str): The directory containing the CSV files.

    Returns:
        pd.DataFrame: The processed observed data.
    """

    # Set up the sfcwind filename
    sfcwind_filename = f"ERA5_obs_{variable_wind}_{country_name}_{season}_{start_year}_{end_year}.csv"

    # Set up the temperature filename
    t2m_filename = f"ERA5_obs_{variable_t2m}_{country_name}_{season}_{start_year}_{end_year}.csv"

    # form the full path to the csv files
    sfcwind_path = os.path.join(csv_dir, sfcwind_filename)
    t2m_path = os.path.join(csv_dir, t2m_filename)

    # assert that these files exist
    assert os.path.exists(sfcwind_path), f"File not found: {sfcwind_path}"
    assert os.path.exists(t2m_path), f"File not found: {t2m_path}"

    # Load the dataframes
    sfcwind_df = pd.read_csv(sfcwind_path, index_col=0, parse_dates=True)
    t2m_df = pd.read_csv(t2m_path, index_col=0, parse_dates=True)

    # # print the head of the sfcwind_df
    # print(sfcwind_df.head())

    # # print the head of the t2m_df
    # print(t2m_df.head())

    # rename obs to the variable name in each case
    sfcwind_df.rename(columns={"obs": variable_wind}, inplace=True)
    t2m_df.rename(columns={"obs": variable_t2m}, inplace=True)

    # merge the dataframes
    obs_df = pd.merge(sfcwind_df, t2m_df, left_index=True, right_index=True)

    # # print the head of the obs_df
    print(obs_df.head())

        # if the season is ONDJFM, then subset the df to the months
    if season == "ONDJFM":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([10, 11, 12, 1, 2, 3])]
    elif season == "AMJJAS":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([4, 5, 6, 7, 8, 9])]
    elif season == "DJF":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([12, 1, 2])]
    elif season == "MAM":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([3, 4, 5])]
    elif season == "JJA":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([6, 7, 8])]
    elif season == "SON":
        # subset the df to the months
        obs_df = obs_df[obs_df.index.month.isin([9, 10, 11])]
    else:
        raise ValueError(f"Season {season} not recognised.")
    
    # # resample into monthly means
    # obs_df = obs_df.resample("ME").mean()

    # if the season is in ["ONDJFM", "DJFM"]
    if season in ["ONDJFM", "DJFM", "NDJFM"]:
        # shift back back 3 months
        obs_df_monthly = obs_df.shift(-3, freq="ME").resample("YE").mean()

        # drop the first and last year
        obs_df_monthly = obs_df_monthly.iloc[1:-1]
    elif season in ["DJF", "ONDJF", "NDJF"]:
        # shift back 2 months
        obs_df_monthly = obs_df.shift(-2, freq="ME").resample("YE").mean()

        # drop the first and last year
        obs_df_monthly = obs_df_monthly.iloc[1:-1]
    elif season in ["DJ", "ONDJ", "NDJ"]:
        # shift back 1 month
        obs_df_monthly = obs_df.shift(-1, freq="ME").resample("YE").mean()

        # drop the first and last year
        obs_df_monthly = obs_df_monthly.iloc[1:-1]
    else:
        # take the annual mean
        obs_df_monthly = obs_df.resample("YE").mean()

    # # print the head of the obs_df_monthly
    print(obs_df_monthly.head())

    # if temperature is in K, convert to C
    # if any values are greater than 100
    # assume that the temperature is in K
    if obs_df_monthly[variable_t2m].max() > 100:
        # convert to C
        obs_df_monthly[variable_t2m] = obs_df_monthly[variable_t2m] - 273.15

    # fix the index to just hve the years
    obs_df_monthly.index = obs_df_monthly.index.year

    # Return the data
    return obs_df_monthly

# Define a function to load in the processed model data
def load_processed_model_data(
    obs_df: pd.DataFrame,
    country_name: str,
    season: str,
    start_year: int,
    end_year: int,
    obs_years: List[int] = [1979, 2018],
    leads: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    variable_wind: str = "sfcWind",
    variable_t2m: str = "tas",
    model_name: str = "HadGEM3-GC31-MM",
    experiment: str = "dcppA-hindcast",
    frequency: str = "Amon",
    csv_dir: str = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"
) -> pd.DataFrame:
    """
    This function loads in the processed model data for a given country and season.

    Args:
        country_name (str): The name of the country.
        season (str): The season to load the data for.
        start_year (int): The start year of the data.
        end_year (int): The end year of the data.
        obs_years (List[int]): The years of the observed data.
        variable_wind (str): The name of the wind variable.
        variable_t2m (str): The name of the temperature variable.
        model_name (str): The name of the model.
        experiment (str): The name of the experiment.
        frequency (str): The frequency of the data.
        csv_dir (str): The directory containing the CSV files.

    Returns:
        pd.DataFrame: The processed model data.
    """

    # Set up the sfcwind filename
    sfcwind_filename = f"{model_name}_{variable_wind}_{country_name}_{season}_{start_year}_{end_year}_{experiment}_{frequency}.csv"

    # Set up the temperature filename
    t2m_filename = f"{model_name}_{variable_t2m}_{country_name}_{season}_{start_year}_{end_year}_{experiment}_{frequency}.csv"

    # form the full path to the csv files
    sfcwind_path = os.path.join(csv_dir, sfcwind_filename)
    t2m_path = os.path.join(csv_dir, t2m_filename)

    # assert that these files exist
    assert os.path.exists(sfcwind_path), f"File not found: {sfcwind_path}"
    assert os.path.exists(t2m_path), f"File not found: {t2m_path}"

    # Load the dataframes
    sfcwind_df = pd.read_csv(sfcwind_path)
    t2m_df = pd.read_csv(t2m_path)

    # # print the head of the sfcwind_df
    # print(sfcwind_df.head())

    # # print the head of the t2m_df
    # print(t2m_df.head())

    # rename obs to the variable name in each case
    sfcwind_df.rename(columns={"obs": variable_wind}, inplace=True)
    t2m_df.rename(columns={"obs": variable_t2m}, inplace=True)

    # replace the 'data' column
    # with the variable name
    sfcwind_df.rename(columns={"data": variable_wind}, inplace=True)
    t2m_df.rename(columns={"data": variable_t2m}, inplace=True)

    # merge the dataframes
    model_df = pd.merge(sfcwind_df, t2m_df, on=["init_year", "member", "lead"])

    # if the tas column has values greater than 100
    # then convert to C
    if model_df[variable_t2m].max() > 100:
        # convert to C
        model_df[variable_t2m] = model_df[variable_t2m] - 273.15

    # assert that the season is ONDJFM
    assert season == "ONDJFM", f"Season {season} not recognised - must be ONDJFM."

    # Set up a new empty dataframe
    df_new = pd.DataFrame()

    # loop over the unique initialisation years
    for iyear in model_df["init_year"].unique():
        for m in model_df["member"].unique():
            for l in leads:
                # subset the data
                df_this = model_df[(model_df["init_year"] == iyear) & (model_df["member"] == m)]

                # subset to lead values 
                model_data = df_this[df_this['lead'].isin([(12*l) ,(12*l) + 1, (12*l) + 2, (12*l) + 3, (12*l) + 4, (12*l) + 5])]

                mean_tas = model_data[variable_t2m].mean()
                mean_sfcWind = model_data[variable_wind].mean()

                # create a dataframe this
                model_data_this = pd.DataFrame(
                    {
                        'init_year': [iyear],
                        'member': [m],
                        'lead': [l],
                        'tas': [mean_tas],
                        'sfcWind': [mean_sfcWind]
                    }
                )

                df_new = pd.concat([df_new, model_data_this])

    # print the head of the df_new
    # print(df_new.head())
                
    # Constrain to the obs years
    # init years must be greater than or equal to the first obs year
    # and less than or equal to the last obs year
    df_new = df_new[(df_new["init_year"] >= obs_years[0]) & (df_new["init_year"] <= obs_years[1])]

    # # print the head of the df_new
    print(df_new.head())

    # Quantify the bias
    # model - obs
    sfcWind_bias = df_new["sfcWind"].mean() - obs_df["sfcWind_mon"].mean()

    # Quyantify the tas bias
    tas_bias = df_new["tas"].mean() - obs_df["tas_mon"].mean()

    # Print the biases
    print(f"Mean sfcWind bias: {sfcWind_bias:.2f}")
    print(f"Mean tas bias: {tas_bias:.2f}")

    # include a new column for bc wind and tas
    df_new["sfcWind_bc"] = df_new["sfcWind"] - sfcWind_bias
    df_new["tas_bc"] = df_new["tas"] - tas_bias

    # assert that the mean of the bc column is the same as the obs mean
    assert np.isclose(df_new["sfcWind_bc"].mean(), obs_df["sfcWind_mon"].mean()), "Mean sfcWind bias not removed."

    # assert that the mean of the bc column is the same as the obs mean
    assert np.isclose(df_new["tas_bc"].mean(), obs_df["tas_mon"].mean()), "Mean tas bias not removed."

    return df_new



# Define the main function
def main():
    """
    Main function for the regression analysis.
    """

    # Start a timer
    time_this_start = time.time()

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Regression analysis.")

    # Country
    parser.add_argument(
        "--country_code", type=str, help="Country code for the analysis.", default="UK"
    )
    # Season
    parser.add_argument(
        "--season", type=str, help="Season for the analysis.", default="ONDJFM"
    )
    # Start year
    parser.add_argument(
        "--start", type=int, help="Start year for the analysis.", default=1979
    )
    # End year
    parser.add_argument(
        "--end", type=int, help="End year for the analysis.", default=2018
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print(f"Country code: {args.country_code}")
    print(f"Season: {args.season}")
    print(f"Start year: {args.start}")
    print(f"End year: {args.end}")

    # Create the demand net wind dataframe
    dnw_df = create_dnw_df(
        country_code=args.country_code,
        season=args.season,
        start_year=args.start,
        end_year=args.end,
    )

    # print the head of the dnw_df
    print(dnw_df.head())

    # End the timer
    time_this_end = time.time()

    # Print the time taken
    print(f"Time taken: {time_this_end - time_this_start}")

    return None


# if the script is run
if __name__ == "__main__":
    main()
