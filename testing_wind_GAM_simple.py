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

# %%
import time

# Start the time
# start a timer
start_time = time.time()

# Imports
import os
import sys
import glob
import argparse

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

# print("success")
# sys.exit(0)


# Define a main function
def main():

    # TODO: will also have to be regridded to the same grid as the demand data
    # like for the demand from temperature data
    # Set up hard coded variables
    wind_obs_path = (
        "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1960_2020.nc"
    )
    country = "United Kingdom"
    country_name = "United_Kingdom"
    start_date = "1960-01-01"
    end_date = "1965-12-31"
    # Define the number of simulations required
    nsims = 1000

    # set up a fname for the output file
    fname_10_100 = os.path.join(
        os.getcwd(),
        "csv_dir",
        f"wind_speeds_10m_100m_{country_name}_{start_date}_{end_date}.csv",
    )

    # set up a name for the output file
    output_file = os.path.join(
        os.getcwd(),
        "npy_dir",
        f"wind_speeds_100m_preds_{country_name}_{start_date}_{end_date}_{nsims}_train80_test20.npy",
    )

    # if the directory does not exist, create it
    if not os.path.exists(os.path.join(os.getcwd(), "npy_dir")):
        os.makedirs(os.path.join(os.getcwd(), "npy_dir"))

    # if the output file exists, then print that it does
    if os.path.exists(output_file):
        print(f"File {output_file} already exists.")
        print("Loading the file.")

        # load the file
        preds = np.load(output_file)

        # print the shape of the predictions
        print(f"Predictions shape:", np.shape(preds))

        # if the file exists, then print that it does
        if os.path.exists(fname_10_100):
            print(f"File {fname_10_100} already exists.")
            print("Loading the file.")

            # load the file
            wind_obs_10m_100m_bc_uk = pd.read_csv(fname_10_100, index_col=0)

            train = wind_obs_10m_100m_bc_uk.iloc[: int(0.8 * len(wind_obs_10m_100m_bc_uk))]

            # print the training data shape
            print(f"Training data shape:", train.shape)

            # Set up the testing data
            test = wind_obs_10m_100m_bc_uk.iloc[int(0.8 * len(wind_obs_10m_100m_bc_uk)) :]

    else:
        print(f"File {output_file} does not exist for predictions.")
        print("Processing from .nc.")

        # Set up the R home environment variable
        os.environ["R_HOME"] = (
            "/apps/jasmin/jaspy/miniforge_envs/jasr4.3/mf3-23.11.0-0/envs/jasr4.3-mf3-23.11.0-0-r20240320/lib/R"
        )

        # import rpy2
        # ! pip install rpy2

        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()
        r_mgcv = importr("mgcv")
        base = importr("base")
        stats = importr("stats")
        graphics = importr("graphics")
        ggplot2 = importr("ggplot2")

        # if the file exists, then print that it does
        if os.path.exists(fname_10_100):
            print(f"File {fname_10_100} already exists.")
            print("Loading the file.")

            # load the file
            wind_obs_10m_100m_bc_uk = pd.read_csv(fname_10_100, index_col=0)
        else:
            print(f"File {fname_10_100} does not exist.")
            print("Processinf from .nc")

            # if the csv_dir does not exist, create it
            if not os.path.exists(os.path.join(os.getcwd(), "csv_dir")):
                os.makedirs(os.path.join(os.getcwd(), "csv_dir"))

            # assert that this file exist
            assert os.path.exists(wind_obs_path), f"File {wind_obs_path} does not exist."

            # load the file with xarray
            wind_obs = xr.open_dataset(
                wind_obs_path,
                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
            )

            # print the wind obs
            print(wind_obs)

            # Select the data
            wind_obs_10m = wind_obs["si10"]
            wind_obs_100m_bc = wind_obs["si100_bc"]

            # # restrict to the first year
            # # For initial testing
            wind_obs_10m = wind_obs_10m.sel(time=slice(start_date, end_date))
            wind_obs_100m_bc = wind_obs_100m_bc.sel(time=slice(start_date, end_date))

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

            # join the two dataframes by the index
            wind_obs_10m_100m_bc_uk = pd.concat(
                [wind_obs_10m_uk, wind_obs_100m_bc_uk], axis=1
            )

            # add a new column for day of the year
            wind_obs_10m_100m_bc_uk["day_of_year"] = wind_obs_10m_100m_bc_uk.index.dayofyear

            # save the joined dataframe as a csv
            wind_obs_10m_100m_bc_uk.to_csv(fname_10_100, index=True)

        # print the joined dataframe
        print(f"Joined dataframe:", wind_obs_10m_100m_bc_uk)

        # Split the data into training and testing
        # first 80% of rows for training
        # last 20% of rows for testing
        train = wind_obs_10m_100m_bc_uk.iloc[: int(0.8 * len(wind_obs_10m_100m_bc_uk))]

        # print the training data shape
        print(f"Training data shape:", train.shape)

        # Set up the testing data
        test = wind_obs_10m_100m_bc_uk.iloc[int(0.8 * len(wind_obs_10m_100m_bc_uk)) :]

        # print the testing data shape
        print(f"Testing data shape:", test.shape)

        # convert to R dataframe
        r_df_train = ro.conversion.py2rpy(train)

        # convert to R dataframe
        r_df_test = ro.conversion.py2rpy(test)

        # # investigate the R dataframe
        # print(f"R dataframe:", r_df)

        # Define the GAM functions for location and scale
        modparams = []

        # United_Kingdom_si10 United_Kingdom_si100_bc
        # We want to estimate the 100m wind speed from the 10m wind speed and day of the year
        modparams.append(
            "United_Kingdom_si100_bc ~ ti(United_Kingdom_si10, k=15, bs='tp') + ti(day_of_year, k=15, bs='tp')"
        )

        # append the scale function
        modparams.append("~ 1")

        # fit the GAM
        gamFit = r_mgcv.gam(
            [ro.Formula(modparams[0]), ro.Formula(modparams[1])],
            data=r_df_train,
            family="gaulss",
            method="REML",
        )

        # print the GAM fit
        print(f"GAM fit:", gamFit)

        # print the summary of the GAM fit
        summary = base.summary(gamFit)

        # print the summary
        print(f"Summary:", summary)

        # # use ggplot to plot the GAM smooths
        # ggplot2.ggplot(gamFit)

        # ptrint the coefficients of the base model
        coeffs = stats.coef(gamFit)

        # model predictions
        model_preds = r_mgcv.predict_gam(gamFit, r_df_test, type="terms")

        # print the model predictions
        print(f"Model predictions:", model_preds)

        sys.exit(0)

        # print the coefficients
        print(f"Coefficients shape:", np.shape(coeffs))

        # Generate samples of 100m wind speeds from the GAM
        # extract the parameters and the covariance matrix of the parameters
        Vc = np.asmatrix(gamFit.rx2("Vc"))

        # print the covariance matrix of the smooths
        print(f"Covariance matrix of the smooths shape:", np.shape(Vc))

        # Define the number of knots
        nknots = np.array([len(coeffs) - 1, 1]) # the last one is the scale, all others are associated with the mean

        print(f"Number of knots:", nknots)

        # define coefs and Vc as R vector/matrix
        coefs_R = ro.vectors.FloatVector(coeffs)
        Vc_R = ro.r.matrix(Vc, nrow=len(coeffs), ncol=len(coeffs))

        # sample parameters from MVN posterior
        betas = r_mgcv.rmvn(nsims, coefs_R, Vc_R)

        # print the shape of the betas
        print(f"Betas shape:", np.shape(betas))

        # print the values of the betas
        print(f"Betas values:", betas)
        

        # extract the linear predictor matrix
        # vector of linear predictor values (minus any offest)
        # at the supplied covariate values, when applied to the model coefficient vector
        X = stats.predict(
            gamFit, newdata=r_df_test, type="lpmatrix"
        )

        # calculate the GAM mean and sd
        Mean = np.dot(X[:, 0 : int(nknots[0])], np.transpose(betas[:, 0 : int(nknots[0])]))

        # print the mean shape
        print(f"Mean shape:", np.shape(Mean))

        # set up the linear prediction standard deviation
        LPsd = np.dot(X[:, int(nknots[0]) : int(nknots[0] + int(nknots[1]))], np.transpose(betas[:, int(nknots[0]) : int(nknots[0]) + int(nknots[1])]))

        # print the shape of the LPsd
        print(f"LPsd shape:", np.shape(LPsd))

        # set up the standard deviation
        Sd = np.exp(LPsd) + 0.01

        # simulate from the predictive distribution
        preds = np.empty(test.shape[0] * nsims).reshape(test.shape[0], nsims)

        # loop over the rows
        for i in range(0, test.shape[0]):
            for j in range(0, nsims):
                preds[i, j] = stats.rnorm(1, mean=Mean[i, j], sd=Sd[i, j])

        # print the shape of the predictions
        print(f"Predictions shape:", np.shape(preds))

        # print the predictions
        print(f"Predictions:", preds)

        # save the predictions as a numpy file
        np.save(output_file, preds)

        # print that the array has been saved
        print(f"Predictions saved as {output_file}")


    # # Select the first time step for si10
    # si10 = wind_obs["si10"].isel(time=0)

    # # print the values
    # print(si10.values)

    # # do the same but for si100_bc
    # si100_bc = wind_obs["si100_bc"].isel(time=0)

    # # print the values
    # print(si100_bc.values)

    # # plot the 10m wind speeds against the 100m wind speeds for the year
    # plt.figure()

    # # subset the data by season
    # # winter = DJF
    # # autum = SON
    # # spring = MAM
    # # summer = JJA
    # winter_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([12, 1, 2])]
    # winter_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([12, 1, 2])]

    # autumn_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([9, 10, 11])]
    # autumn_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([9, 10, 11])]

    # spring_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([3, 4, 5])]
    # spring_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([3, 4, 5])]

    # summer_10m = wind_obs_10m_uk[wind_obs_10m_uk.index.month.isin([6, 7, 8])]
    # summer_100m_bc = wind_obs_100m_bc_uk[wind_obs_100m_bc_uk.index.month.isin([6, 7, 8])]

    # # plot the 10m wind speeds against the 100m wind speeds as a scatter plot
    # # with different colours for the different seasons
    # # autumn orange
    # # spring green
    # # summer blue
    # # winter purple
    # plt.scatter(winter_10m, winter_100m_bc, color="purple", label="Winter")
    # plt.scatter(autumn_10m, autumn_100m_bc, color="orange", label="Autumn")
    # plt.scatter(spring_10m, spring_100m_bc, color="green", label="Spring")
    # plt.scatter(summer_10m, summer_100m_bc, color="blue", label="Summer")

    # # set the title
    # plt.title("10m vs 100m wind speeds for the UK")

    # # set the x-axis label
    # plt.xlabel("10m wind speed (m/s)")

    # # set the y-axis label
    # plt.ylabel("100m wind speed bc (m/s)")

    # # include the legend in the top left
    # plt.legend(loc="upper left")

    # # set up the dir
    # # the current dir + testing_plots
    # plot_dir = os.path.join(os.getcwd(), "testing_plots")

    # # if this does not exist, create it
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)

    # # set up the current time
    # current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # # save the plot
    # plt.savefig(os.path.join(plot_dir, "scatter_plot_10m_100m_wind_speeds_" + current_time + ".png"))

    # plot the predictions with the mean and confidence intervals
    plt.figure()

    # Set up a dataframe
    # with test.index as the index
    # then the mean of the predictions in the first column
    # then the 2.5th percentile of the predictions in the second column
    # then the 97.5th percentile of the predictions in the third column

    # set up the dataframe
    df = pd.DataFrame(
        {
            "mean": np.mean(preds, axis=1),
            "2.5th": np.percentile(preds, 2.5, axis=1),
            "97.5th": np.percentile(preds, 97.5, axis=1),
        },
        index=test.index,
    )

    # print the dataframe
    print(f"Dataframe:", df)

    # plot the mean as a 10 day rolling mean
    plt.plot(df.index, df["mean"].rolling(10).mean(), label="Mean")

    # plot the 95% confidence intervals
    plt.fill_between(
        df.index,
        df["2.5th"].rolling(10).mean(),
        df["97.5th"].rolling(10).mean(),
        color="grey",
        alpha=0.5,
        label="95% CI",
    )

    # Convert the DatetimeIndex to a list of datetime objects
    # ticks every 30 days
    plt.xticks(
        ticks=df.index[::50],
        labels=[str(i) for i in df.index[::50]],
        rotation=45,
    )

    # set the y-axis label
    plt.ylabel("100m wind speed bc predictions (m/s)")

    # set the title
    plt.title("Predictions of 100m wind speeds for the UK")

    # end the timer
    print(f"Time taken: {time.time() - start_time} seconds.")

    # print that we are done
    # print("Done.")
    # sys.exit(0)

    return


if __name__ == "__main__":
    main()

# %%
