import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from netCDF4 import Dataset
import shapely.geometry
import cartopy.io.shapereader as shpreader
import csv
import scipy.interpolate


def find_data_path(
    year: float,
    month: float,
    field_str: str,
    daily_flag: bool = False,
):
    """
    This makes sure the right path is selected for the data (may be redundant in different data storage
    architecture. to that used on the Reading University Computing Cluster. Please adapt to your needs.

    Args:
        year (float): This is the year of data you are interested in
        month (float): The month you are interested in from 1-12
        field_str (str): The name of the field we are interested in: options: 't2m','speed100m',
        'speed10m','ssrd' , 'sp' , 'wp'
        daily_flag (bool): If the data is daily or not. Default is False.

    Returns:
        file_loc (str): The path that the data is stored in
        file_name (str): The name of the file that is needed to create data for field_str
    """
    qyearchar = str(year)
    qmonthchar = str(month).zfill(2)  # Set month name to always have 2

    if (field_str in ["t2m", "speed10m", "speed100m"]) and (year >= 1979):

        file_loc = "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/"

        file_name = "ERA5_1hr_" + qyearchar + "_" + qmonthchar + "_DET.nc"

    elif (field_str in ["t2m", "speed10m", "speed100m"]) and (year < 1979):

        file_loc = "/gws/nopw/j04/canari/users/benhutch/ERA5/"
        file_name = "ERA5_1hr_" + qyearchar + "_" + qmonthchar + "_DET.nc"

        if daily_flag:
            file_loc = "/gws/nopw/j04/canari/users/benhutch/ERA5/"
            file_name = "ERA5_daily_" + qyearchar + "_" + qmonthchar + ".nc"

    elif (field_str in ["ssrd"]) and (year >= 1979):

        file_loc = "/storage/silver/S2S4E/energymet/ERA5/RSDS/native_grid_hourly/"

        file_name = "ERA5_1hr_RSDS_" + qyearchar + "_" + qmonthchar + "_DET.nc"

    elif (field_str in ["ssrd"]) and (year < 1979):

        file_loc = "/storage/silver/clearheads/Data/ERA5_data/native_grid/RSDS/"

        file_name = "ERA5_1hr_RSDS_" + qyearchar + "_" + qmonthchar + "_DET.nc"

    else:
        print("file location details needed in find_data_path function")
    return (file_loc, file_name)


def load_country_mask(COUNTRY, data_dir, filename, nc_key, pop_weights):
    """
    This function loads the country masks for the ERA5 data grid we have been using

    Args:
        COUNTRY (str): This must be a name of a country (or set of) e.g.
            'United Kingdom','France','Czech Republic'

       data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The string you need to load the .nc data
            e.g. 't2m','rsds'

        pop_weights (int): 0 if no population weightings, 1 if required pop weightings.

    Returns:
       MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if
           the data is within a country border and zeros if data is outside a
           country border. This will include population weightings if pop_weights = 1


    """

    # first loop through the countries and extract the appropraite shapefile
    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME_LONG"] == COUNTRY:
            print("Found country")
            country_shapely.append(country.geometry)

    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str, mode="r")
    lons = dataset.variables["longitude"][:]
    lats = dataset.variables["latitude"][:]
    if nc_key == "speed10m":
        data1 = dataset.variables["u10"][:]
        data2 = dataset.variables["v10"][:]
        data = np.sqrt(data1**2 + data2**2)
    elif nc_key == "speed100m":
        data1 = dataset.variables["u100"][:]
        data2 = dataset.variables["v100"][:]
        data = np.sqrt(data1**2 + data2**2)
    else:
        data = dataset.variables[nc_key][:]  # data in shape [time,lat,lon]
    dataset.close()

    # get data in appropriate units for models
    if nc_key == "t2m":
        data = data - 273.15  # convert to Kelvin from Celsius
    if nc_key == "ssrd":
        data = data / 3600.0  # convert Jh-1m-2 to Wm-2

    LONS, LATS = np.meshgrid(lons, lats)  # make grids of the lat and lon data
    x, y = LONS.flatten(), LATS.flatten()  # flatten these to make it easier to
    # loop over.
    points = np.vstack((x, y)).T
    MASK_MATRIX = np.zeros((len(x), 1))
    # loop through all the lat/lon combinations to get the masked points
    for i in range(0, len(x)):
        my_point = shapely.geometry.Point(x[i], y[i])
        if country_shapely[0].contains(my_point) == True:
            MASK_MATRIX[i, 0] = 1.0  # creates 1s and 0s where the country is

    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX, (len(lats), len(lons)))

    # if it is the UK lets mask out Northern Ireland so we've got GB instead

    if COUNTRY == "United Kingdom":
        for i in range(0, len(lats)):
            for j in range(0, len(lons)):
                if (lats[i] < 55.3) and (lats[i] > 54.0):
                    if lons[j] < -5.0:
                        MASK_MATRIX_RESHAPE[i, j] = 0.0

    # include the population data if required.
    if pop_weights == 1:
        LONS, LATS = np.meshgrid(lons, lats)  # make grids of the lat and lon data
        dataset = Dataset(
            "/home/users/zd907959/Data/S2S4E/population_data/gpw_v4_e_atotpopbt_cntm_30_min.nc",
            mode="r",
        )
        # load in the data dataset.variables.keys() shows all of the short names. available from: https://doi.org/10.1080/23754931.2015.1014272
        pop_lons = dataset.variables["longitude"][
            260:460
        ]  # isolate over Europe (-49.5E:50.25E)
        pop_lats = dataset.variables["latitude"][
            8:160
        ]  # isolate 4.75N:80.75N (similar to the ERA5 gridded data.
        pop_totals = dataset.variables[
            "Population Count, v4.10 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes"
        ][
            3, 8:160, 260:460
        ]  # [15 x 290lats x 720 lons] input 3 is the 2015 population counts.
        dataset.close()
        # now lets interpolate this onto the same grid as ERA5.
        POPULATION_LONS, POPULATION_LATS = np.meshgrid(pop_lons, pop_lats)

        points_array = np.array([POPULATION_LONS.flatten(), POPULATION_LATS.flatten()])
        points_to_interp_array = np.array([LONS.flatten(), LATS.flatten()])

        Population_totals_interp = scipy.interpolate.griddata(
            points_array.transpose(),
            pop_totals.flatten(),
            points_to_interp_array.transpose(),
        )
        MASK_MATRIX_RESHAPE = MASK_MATRIX_RESHAPE * Population_totals_interp

    return MASK_MATRIX_RESHAPE


def load_country_weather_data(
    MASK_MATRIX_RESHAPE, data_dir, filename, nc_key, ons_ofs, MODEL_NAME, month
):
    """
    This function takes the gridded weather data, loads it and applies a
    pre-loaded country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints
    set to zeros.

    You will need the shpreader.natural_earth data downloaded
    to find the shapefiles.

    Args:

      MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if
           the data is within a country border and zeros if data is outside a
           country border.

        data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The string you need to load the .nc data
            e.g. 't2m','rsds'

        ons_ofs (str): The string to confirm if wind power data is onshore or offshore (might
                       not be applicable to all cases but still required. e,g, 'ons','ofs'


        MODEL_NAME (str): The name of the climate model or reanalysis being used.

        month (int): The month in question that needs loading from 1-12

    Returns:

        country_masked_data (array): Country-masked weather data, dimensions
            [time,lat,lon] where there are 0's in locations where the data is
            not within the country border.


    """

    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str, mode="r")
    lons = dataset.variables["longitude"][:]
    lats = dataset.variables["latitude"][:]
    if nc_key == "speed10m":
        data1 = dataset.variables["u10"][:]
        data2 = dataset.variables["v10"][:]
        data = np.sqrt(data1**2 + data2**2)
    elif nc_key == "speed100m":
        data1 = dataset.variables["u100"][:]
        data2 = dataset.variables["v100"][:]
        data_raw = np.sqrt(data1**2 + data2**2)
        # note we bias correct the ERA5 data to the global wind atlas as we know ERA5 winds are a bit low.
        bias_correction_file = "/home/users/benhutch/UREAD_energy_models_demo_scripts/ERA5_speed100m_mean_factor_v16_hourly.npy"
        correction_factors = np.load(bias_correction_file)
        data_cor = np.zeros(np.shape(data_raw))
        for i in range(0, np.shape(data_raw)[0]):
            # dont worry about speeds going less than zero, the power curve will ignore these.
            data_cor[i, :, :] = data_raw[i, :, :] + correction_factors
            # scale down to hub-height using a power law
        if ons_ofs == "ons":
            data = data_cor * (71.0 / 100.0) ** (
                1.0 / 7.0
            )  # average heights from UK windpower.net 2021 onshore wind farms
        if ons_ofs == "ofs":
            data = data_cor * (92.0 / 100.0) ** (
                1.0 / 7.0
            )  # average heights from UK windpower.net 2021 offshore wind farms

    else:
        data = dataset.variables[nc_key][:]  # data in shape [time,lat,lon]

    dataset.close()

    # get data in appropriate units for models
    if nc_key == "t2m":
        data = data - 273.15  # convert to Kelvin from Celsius
    if nc_key == "ssrd":
        data = data / 3600.0  # convert Jh-1m-2 to Wm-2

    # important with new code structure for aggregation
    MASK_MATRIX_RESHAPE[MASK_MATRIX_RESHAPE == 0.0] = np.nan

    country_masked_data = np.zeros(np.shape(data))
    for i in range(0, len(country_masked_data)):
        country_masked_data[i, :, :] = data[i, :, :] * MASK_MATRIX_RESHAPE

    return country_masked_data


def load_country_weather_data_with_BC(
    MASK_MATRIX_RESHAPE, data_dir, filename, nc_key, ons_ofs, CC_flag, MODEL_NAME, month
):
    """
    This function takes the gridded weather data, loads it and applies a percentile based delta correction
    to each grid box (these have been pre-calculated).
    It then applies a  pre-loaded country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints
    set to zeros.

    You will need the shpreader.natural_earth data downloaded
    to find the shapefiles.

    Args:

      MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if
           the data is within a country border and zeros if data is outside a
           country border.

        data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The string you need to load the .nc data
            e.g. 't2m','rsds'

        ons_ofs (str): The string to confirm if wind power data is onshore or offshore (might
                       not be applicable to all cases but still required. e,g, 'ons','ofs'

       CC_flag (str): This says whether the bias correction should be applied or not. 1=yes 0=no.

        MODEL_NAME (str): The name of the climate model or reanalysis being used.

        month (int): The month in question that needs loading from 1-12

    Returns:

        country_masked_data (array): Country-masked weather data, dimensions
            [time,lat,lon] where there are 0's in locations where the data is
            not within the country border.
    """

    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str, mode="r")
    lons = dataset.variables["longitude"][107:304]
    lats = dataset.variables["latitude"][50:213]
    if nc_key == "speed10m":
        data1 = dataset.variables["u10"][:, 50:213, 107:304]
        data2 = dataset.variables["v10"][:, 50:213, 107:304]
        data = np.sqrt(data1**2 + data2**2)
    elif nc_key == "speed100m":
        data1 = dataset.variables["u100"][:, 50:213, 107:304]
        data2 = dataset.variables["v100"][:, 50:213, 107:304]
        data_raw = np.sqrt(data1**2 + data2**2)
        # note we bias correct the ERA5 data to the global wind atlas as we know it has a low bias.
        bias_correction_file = "/home/users/zd907959/code_folders/S2S4E/wind_power_model/ERA5_model/ERA5_speed100m_mean_factor_v16_hourly.npy"
        correction_factors = np.load(bias_correction_file)
        # extract the relevant section we have BC data for.
        correction_factors = correction_factors[50:213, 107:304]
        data_cor = np.zeros(np.shape(data_raw))
        for i in range(0, np.shape(data_raw)[0]):
            # dont worry about speeds going less than zero, the power curve will ignore these.
            data_cor[i, :, :] = data_raw[i, :, :] + correction_factors
            # scale down to hub-height using a power law
        if ons_ofs == "ons":
            data = data_cor * (71.0 / 100.0) ** (
                1.0 / 7.0
            )  # average onshore 2021 wind turbine heights from UK windpower.net
        if ons_ofs == "ofs":
            data = data_cor * (92.0 / 100.0) ** (
                1.0 / 7.0
            )  # average offshore 2021 wind turbine  heights from UK windpower.net

    else:
        data = dataset.variables[nc_key][
            :, 50:213, 107:304
        ]  # data in shape [time,lat,lon]

    dataset.close()

    # get data in appropriate units for models
    if nc_key == "t2m":
        data = data - 273.15  # convert to Kelvin from Celsius
    if nc_key == "ssrd":
        data = data / 3600.0  # convert Jh-1m-2 to Wm-2

    if CC_flag == 1:

        # tas and t2m are same fields, just different keys in ERA5 and climate model
        if nc_key == "t2m":
            if MODEL_NAME == "MOHC_MM-1hr2":
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_tas_MOHC_MM-1hr_perc_diffs_ens2.npz"
                )
            else:
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_tas_"
                    + MODEL_NAME
                    + "_perc_diffs.npz"
                )
            load_era5 = np.load(
                "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/ERA5_tas_perc_dist.npz"
            )

        # same corrections for speed10m and speed100m as only 10m wind speed avail in climate models.
        elif nc_key in ["speed10m", "speed100m"]:
            if MODEL_NAME == "MOHC_MM-1hr2":
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_uvas_MOHC_MM-1hr_perc_diffs_ens2.npz"
                )
            else:
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_uvas_"
                    + MODEL_NAME
                    + "_perc_diffs.npz"
                )
            load_era5 = np.load(
                "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/ERA5_uvas_perc_dist.npz"
            )

        # ssrd and rsds are same fields, just different keys
        elif nc_key in ["ssrd"]:
            if MODEL_NAME == "MOHC_MM-1hr2":
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_rsds_MOHC_MM-1hr_perc_diffs_ens2.npz"
                )
            else:
                load_np = np.load(
                    "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/fut_min_hist_mean_rsds_"
                    + MODEL_NAME
                    + "_perc_diffs.npz"
                )
            load_era5 = np.load(
                "/home/users/zd907959/code_folders/CLEARHEADS/reading_clim_data/data/all_ERA5grid/ERA5_rsds_perc_dist.npz"
            )
        else:
            print("need to create appropriate correction factors!")

        delta_corrs_all = load_np["arr_0"]
        perc_e = load_era5["arr_0"]

        if month in [12, 1, 2]:
            delta_corrs = delta_corrs_all[0, :, :, :]
            perc_of_era5 = perc_e[0, :, :, :]
        elif month in [3, 4, 5]:
            delta_corrs = delta_corrs_all[1, :, :, :]
            perc_of_era5 = perc_e[1, :, :, :]
        elif month in [6, 7, 8]:
            delta_corrs = delta_corrs_all[2, :, :, :]
            perc_of_era5 = perc_e[2, :, :, :]
        else:
            delta_corrs = delta_corrs_all[3, :, :, :]
            perc_of_era5 = perc_e[3, :, :, :]

        data_corr = np.zeros(np.shape(data))
        for lat_i in range(0, len(lats)):
            for lon_i in range(0, len(lons)):

                temp = data[:, lat_i, lon_i]

                # in some gridboxes in winter it is not possible to do the delta correction as they spend too much time in darkness, so if there is not a full percentile distribution (e.g. some repeated zeros) then just take the data rather than trying to do a correction.
                if (
                    np.min(
                        (
                            perc_of_era5[1:101, lat_i, lon_i]
                            - perc_of_era5[0:100, lat_i, lon_i]
                        )
                    )
                    <= 0.0
                ):
                    # take percentiles and delta corrections for that gridbox
                    perc_subset = perc_of_era5[:, lat_i, lon_i]
                    delta_subset = delta_corrs[:, lat_i, lon_i]
                    # take the data out for where the percentile is >0. (with tiny threshold for small numbers)
                    perc_keys = np.where(perc_subset >= 0.01)
                    percs = perc_subset[perc_keys]
                    deltas = delta_subset[perc_keys]
                    # set bottom correction to zero so not adding on anything at night. (this is fine as it's usually a very small number!
                    deltas[0] = 0

                    # apply the correction
                    test = np.digitize(
                        temp, percs, right=False
                    )  # indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin.
                    test[test == len(percs)] = (
                        len(percs) - 1
                    )  # make sure the bins don't go off the end (power is zero by then anyway)
                    data_corr[:, lat_i, lon_i] = temp + deltas[test]

                else:
                    test = np.digitize(
                        temp, perc_of_era5[:, lat_i, lon_i], right=False
                    )  # indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin.
                    test[test == len(perc_of_era5[:, lat_i, lon_i])] = (
                        100  # make sure the bins don't go off the end (power is zero by then anyway)
                    )
                    data_corr[:, lat_i, lon_i] = temp + delta_corrs[test, lat_i, lon_i]

    else:
        print("no bias corr, check settings")

    # important with new code structure
    MASK_MATRIX_RESHAPE[MASK_MATRIX_RESHAPE == 0.0] = np.nan
    country_masked_data = np.zeros(np.shape(data_corr))
    for i in range(0, len(country_masked_data)):
        country_masked_data[i, :, :] = (
            data_corr[i, :, :] * MASK_MATRIX_RESHAPE[50:213, 107:304]
        )

    return country_masked_data


def create_solar_power(
    MASK_MATRIX_RESHAPE,
    data_dir1,
    filename1,
    data_dir2,
    filename2,
    month,
    year,
    ons_ofs,
    CC_flag,
    MODEL_NAME,
):
    """

    This function loads in arrays of country_masked 2m temperature (celsius)
    and surface solar irradiance (Wm-2) and converts this into an array of
    solar power capacity factor using the method from Bloomfield et al.,
    (2020) https://doi.org/10.1002/met.1858

    Args:
      MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if
           the data is within a country border and zeros if data is outside a
           country border.

        data_dir1,data_dir2 (str): The path for where the t2m and ssrd data is stored. 1 = t2m, 2=ssrd
            e.g '/home/users/zd907959/'

        filename1,filename2 (str): The filename of a .netcdf file.  1 = t2m, 2=ssrd
            e.g. 'ERA5_1979_01.nc'
         month (int): The month you are in.

        year (int): The year you are in.

        ons_ofs (str): the flag stating if this is for onshore or offshore e.g. 'ons','ofs'

        CC_flag (int): the flag stating if a delta correction has been included e.g. 0,1

        MODEL_NAME (str): the name of the climate model being used ('ERA5' if no delta correction).


    Returns:
       capacity_factor_of_pannel (array): Country-masked solar PV data, dimensions
            [time,lat,lon] where there are 0's in locations where the data is
            not within the country border.


    """

    if CC_flag == 0:
        country_masked_data_T2m = load_country_weather_data(
            MASK_MATRIX_RESHAPE, data_dir1, filename1, "t2m", ons_ofs, MODEL_NAME, month
        )
        country_masked_data_ssrd = load_country_weather_data(
            MASK_MATRIX_RESHAPE,
            data_dir2,
            filename2,
            "ssrd",
            ons_ofs,
            MODEL_NAME,
            month,
        )
    else:
        country_masked_data_T2m = load_country_weather_data_with_BC(
            MASK_MATRIX_RESHAPE,
            data_dir1,
            filename1,
            "t2m",
            ons_ofs,
            CC_flag,
            MODEL_NAME,
            month,
        )
        country_masked_data_ssrd = load_country_weather_data_with_BC(
            MASK_MATRIX_RESHAPE,
            data_dir2,
            filename2,
            "ssrd",
            ons_ofs,
            CC_flag,
            MODEL_NAME,
            month,
        )

    # fill in 1950 gaps as first 7 hours of radiation are missing.
    if (month == 1) and year == (1950):
        temp_data = np.zeros_like(country_masked_data_T2m)
        print(np.shape(temp_data))
        temp_data[7:, :, :] = country_masked_data_ssrd
        country_masked_data_ssrd = temp_data

    # reference values, see Evans and Florschuetz, (1977)
    T_ref = 25.0
    eff_ref = 0.9  # adapted based on Bett and Thornton (2016)
    beta_ref = 0.0042
    G_ref = 1000.0

    rel_efficiency_of_pannel = eff_ref * (
        1 - beta_ref * (country_masked_data_T2m - T_ref)
    )
    capacity_factor_of_pannel = np.nan_to_num(
        rel_efficiency_of_pannel * (country_masked_data_ssrd / G_ref)
    )

    return capacity_factor_of_pannel


def create_wind_power(
    MASK_MATRIX_RESHAPE,
    data_dir1,
    filename1,
    ons_ofs,
    ERA5_lats,
    ERA5_lons,
    total_MW,
    wp_weights,
    NUTS_lev,
    CC_flag,
    MODEL_NAME,
    month,
):
    """

    This function loads in arrays of 100m wind speed from ERA5
    and converts this into an array of
    wind power capacity factor. if CC_flag = 1 this will also include a delta correction to include the impacts of climate change.

    Args:
      MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if
           the data is within a country border and zeros if data is outside a
           country border. (may be a bit different for weighted fields)

        data_dir1, (str): The path for where the speed100m data is stored.
            e.g '/home/users/zd907959/'

        filename1,(str): The filename of a .netcdf file.  1 = t2m, 2=ssrd
            e.g. 'ERA5_1979_01.nc'

        ons_ofs (str): Either 'ons' or 'ofs' to represent onshore or offshore wind respectively so the correct power curve is loaded.

        ERA5_lats (array): 1D array of the latitudes of the mask

        ERA5_lons (array): 1D array of the longitudes of the mask

    total_MW (array): 2D array dim [lats,lons] of the wind turbine locations (all ones if no weightings applied)

    wp_weights (array): 2D array with amount of wind power installed in each gridbox included.

    NUTS_lev (int): details of the NUTS area e.g. 0,1,2,8,9.

    CC_flag (int): details of whether a delta correction will be applied. 0=no 1=yes

    MODEL_NAME (str): the name of the climate model being used for the delta correction. If no delta corretion this is 'ERA5'

    month (int): The month of data we are using from 1-12.

    Returns:
       cf (array): Country-masked wind power capacity factor data, dimensions
            [time,lat,lon] where there are 0's in locations where there is no wind power generation installed.


    """

    pc_p = []
    pc_w = []
    # firstly load the power curve
    if ons_ofs == "ons":
        with open("/home/users/benhutch/Hannah_model/power_onshore.csv") as f:
            for line in f:
                columns = line.split()
                # print columns[0]
                pc_p.append(float(columns[1][0:8]))  # get power curve output (CF)
                pc_w.append(float(columns[0][0:8]))  # get power curve output (CF)
    elif ons_ofs == "ofs":
        with open("/home/users/benhutch/Hannah_model/power_offshore.csv") as f:
            for line in f:
                columns = line.split()
                # print columns[0]
                pc_p.append(float(columns[1][0:8]))  # get power curve output (CF)
                pc_w.append(float(columns[0][0:8]))  # get power curve output (CF)

    # interpolate appropriately
    power_curve_w = np.array(pc_w)
    power_curve_p = np.array(pc_p)
    pc_winds = np.linspace(0, 50, 501)  # make it finer resolution
    pc_power = np.interp(pc_winds, power_curve_w, power_curve_p)

    if NUTS_lev > 0:
        if CC_flag == 0:
            country_masked_data = load_country_weather_data(
                MASK_MATRIX_RESHAPE,
                data_dir1,
                filename1,
                "speed100m",
                ons_ofs,
                CC_flag,
                MODEL_NAME,
                month,
            )
        else:
            country_masked_data = load_country_weather_data_with_BC(
                MASK_MATRIX_RESHAPE,
                data_dir1,
                filename1,
                "speed100m",
                ons_ofs,
                CC_flag,
                MODEL_NAME,
                month,
            )

    else:  # we want to load all the data and weight at the end!
        MASK_MATRIX_RESHAPE1 = np.zeros_like(MASK_MATRIX_RESHAPE) + 1
        if CC_flag == 0:
            country_masked_data = load_country_weather_data(
                MASK_MATRIX_RESHAPE1,
                data_dir1,
                filename1,
                "speed100m",
                ons_ofs,
                MODEL_NAME,
                month,
            )
        else:
            country_masked_data = load_country_weather_data_with_BC(
                MASK_MATRIX_RESHAPE1,
                data_dir1,
                filename1,
                "speed100m",
                ons_ofs,
                CC_flag,
                MODEL_NAME,
                month,
            )

    maxqhr = np.shape(country_masked_data)[0]  # get number of hours

    # create array to fill with capacity factors.
    cf = np.zeros_like(country_masked_data)
    for qhr in range(0, maxqhr):
        VAR_QHR = country_masked_data[qhr, :, :]

        # now lets apply the masks

        Reshaped_speed = np.reshape(
            VAR_QHR, [np.shape(VAR_QHR)[0] * np.shape(VAR_QHR)[1]]
        )

        test2 = np.digitize(
            Reshaped_speed, pc_winds, right=False
        )  # indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin.
        test2[test2 == len(pc_winds)] = (
            500  # make sure the bins don't go off the end (power is zero by then anyway)
        )
        p_hh_temp = 0.5 * (pc_power[test2 - 1] + pc_power[test2])
        capacity_factor_of_pannel = np.reshape(
            p_hh_temp, [np.shape(VAR_QHR)[0], np.shape(VAR_QHR)[1]]
        )
        capacity_factor_of_pannel[capacity_factor_of_pannel <= 0.006] = 0.0
        cf[qhr, :, :] = capacity_factor_of_pannel * total_MW

    return cf


def get_country_list(NUTS_lev, NUTS9_country_choice):
    """

    This function provides the names of all the countries to model for a given NUTS zone.
    Args:

    NUTS_lev (int): 0,1,2 representing the NUTS level.

    NUTS9_country_choice (str): either 'United Kingdom','Ireland' or 'Norway' to give the data in the MetOffice shipping zones.

    Returns:
    country_list (list): list of strings with the country names that will be modelled.


    """
    if NUTS_lev == 0:

        country_list = [
            "Austria",
            "Albania",
            "Belarus",
            "Belgium",
            "Bosnia and Herzegovina",
            "Bulgaria",
            "Croatia",
            "Czech Republic",
            "Denmark",
            "Estonia",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Hungary",
            "Ireland",
            "Italy",
            "Kosovo",
            "Latvia",
            "Lithuania",
            "Luxembourg",
            "Macedonia",
            "Moldova",
            "Montenegro",
            "Netherlands",
            "Norway",
            "Poland",
            "Portugal",
            "Romania",
            "Serbia",
            "Slovakia",
            "Slovenia",
            "Spain",
            "Sweden",
            "Switzerland",
            "Turkey",
            "Ukraine",
            "United Kingdom",
        ]  #

        # the subset we really need!
        # country_list = ['United Kingdom','Austria','Belgium','Denmark','Finland','France','Germany','Ireland','Netherlands','Norway','Sweden','Latvia','Lithuania','Estonia'] #

    if NUTS_lev == 1:
        country_list = [
            "UKC",
            "UKD",
            "UKE",
            "UKF",
            "UKG",
            "UKH",
            "UKI",
            "UKJ",
            "UKK",
            "UKL",
            "UKM",
            "UKN",
        ]

    if NUTS_lev == 2:
        country_list = ["UKM5", "UKM6", "UKM7", "UKM8", "UKM9"]

    if NUTS_lev == 8:

        country_list = [
            "FR",
            "EE",
            "LI",
            "LV",
            "SI",
            "NO",
            "UK",
            "IE",
            "FI",
            "SE",
            "BE",
            "NL",
            "DE",
            "DK",
        ]

    if NUTS_lev == 9:  # this is the offshore wind zones!
        if NUTS9_country_choice == "United Kingdom":
            country_list = [
                "Forties",
                "Cromarty",
                "Forth",
                "Tyne",
                "Dogger",
                "Fisher",
                "Humber",
                "Thames",
                "Dover",
                "Wight",
                "Portland",
                "Plymouth",
                "Lundy",
                "Irish_Sea",
                "Malin",
                "Hebrides",
                "Fair_Isle",
            ]
        if NUTS9_country_choice == "Ireland":
            country_list = [
                "Lundy",
                "Fastnet",
                "Irish_Sea",
                "Shannon",
                "Rockall",
                "Malin",
            ]
        if NUTS9_country_choice == "Norway":
            country_list = [
                "South_Utsire",
                "Forties",
                "Fisher",
            ]  # ['Viking','North_Utsire','South_Utsire','Forties','Fisher']

    return country_list


def load_appropriate_mask(
    COUNTRY,
    NUTS_lev,
    pop_weights,
    sp_weights,
    wp_weights,
    WP_sim,
    ons_ofs,
    NUTS9_country_choice,
):
    """

    This function loads in the country mask you need for the data processing


    Args:

    COUNTRY (str): The name of the region or country you want masked
    NUTS_lev (int): 0,1,2,8,9 representing the NUTS level. (see main script for details)
    pop_weights (int): either 0 (no) or 1 (yes) to include the population weighting
    sp_weights (int): either 0 (no) or 1 (yes) to include the locations of European solar pannels (better data for UK region)
    wp_weights (int): either 0 (no) or 1 (yes) to include the locations of European wind farms from thewindpower.net
    WP_sim (int) : either 0 (present) or 1 (future planned farms) to decide on the wind turbine setup that is used
    ons_ofs (str) : 'ons' or 'ofs' to represent if onshore or oshore wind (may appear redundant for some fields.
    NUTS9_country_choice (str) : either 'United Kingdom', 'Norway', or 'Ireland'. Relecant for the offshore shipping zones.





    Returns:
    country_mask (array): dim [lat,lon], the country mask we need.


    """
    if NUTS_lev == 0:  # assume this is a full nation rather than a EEZ/NUTS zone

        if pop_weights == 1:
            country_mask = load_country_mask(
                COUNTRY,
                "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/",
                "ERA5_1hr_2018_01_DET.nc",
                "t2m",
                pop_weights,
            )
            country_mask = country_mask.data
            # plt.imshow(country_mask)
            # plt.show()

        elif sp_weights == 1:
            if COUNTRY == "United Kingdom":
                # print('cool')
                dataset = Dataset(
                    "/home/users/zd907959/code_folders/CLEARHEADS/solar_power_model/data/United_Kingdom_solar_farm_dist.nc",
                    "r",
                )
                solar_MW = dataset.variables["totals"][:] / 1000.0
                dataset.close()
                # plt.pcolormesh(country_mask)
                # plt.show()
                country_mask = load_country_mask(
                    COUNTRY,
                    "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/",
                    "ERA5_1hr_2018_01_DET.nc",
                    "t2m",
                    pop_weights,
                )
                country_mask = country_mask * solar_MW
                # plt.pcolormesh(country_mask)
                # plt.show()
            else:
                dataset = Dataset(
                    "/home/users/zd907959/code_folders/CLEARHEADS/solar_power_model/data/Europe_solar_farm_dist.nc",
                    "r",
                )
                solar_MW = dataset.variables["totals"][:]
                # plt.pcolormesh(country_mask)
                # plt.show()
                dataset.close()
                country_mask = load_country_mask(
                    COUNTRY,
                    "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/",
                    "ERA5_1hr_2018_01_DET.nc",
                    "t2m",
                    pop_weights,
                )
                country_mask = country_mask * solar_MW

        elif wp_weights == 1:

            if COUNTRY == "United Kingdom":
                COUNTRY = "United_Kingdom"
            elif COUNTRY == "Czech Republic":
                COUNTRY = "Czech_Republic"
            elif COUNTRY == "Bosnia and Herzegovina":
                COUNTRY = "Bosnia_and_Herzegovina"

            if WP_sim == 0:
                print("2020 wind farms")
                # print the file name
                print(
                    "Using: "
                    + "/home/users/benhutch/Hannah_model/installed_capacities/"
                    + str(COUNTRY)
                    + "windfarm_dist_"
                    + str(ons_ofs)
                    + "_2021.nc"
                )

                dataset_1 = Dataset(
                    "/home/users/benhutch/Hannah_model/installed_capacities/"
                    + str(COUNTRY)
                    + "windfarm_dist_"
                    + str(ons_ofs)
                    + "_2021.nc",
                    mode="r",
                )
            elif WP_sim == 1:
                print("future wind farms")
                dataset_1 = Dataset(
                    "/home/users/benhutch/Hannah_model/installed_capacities/"
                    + str(COUNTRY)
                    + "windfarm_dist_"
                    + str(ons_ofs)
                    + "_future.nc",
                    mode="r",
                )
            total_MW = (
                np.flip(dataset_1.variables["totals"][:], axis=0) / 1000.0
            )  # [lat,lon]
            print(f"installed cap = {str(np.sum(total_MW))}")
            dataset_1.close()

            country_mask = total_MW
            # plt.imshow(country_mask)
            # print(country_mask)
            # plt.colorbar()
            # plt.show()

        else:
            country_mask = load_country_mask(
                COUNTRY,
                "/gws/nopw/j04/canari/users/benhutch/ERA5/",
                "ERA5_1hr_1950_01_DET.nc",
                "t2m",
                pop_weights,
            )

    elif NUTS_lev in [1, 2]:
        # the smallers NUTS zones are never population weighted.
        dataset = Dataset(
            "/home/users/zd907959/code_folders/CLEARHEADS/apply_country_masks/NUTS_regions/NUTS_"
            + str(NUTS_lev)
            + "_masks.nc",
            mode="r",
        )
        lons = dataset.variables["lon"][:]
        lats = dataset.variables["lat"][:]
        data = dataset.variables["NUTS zones"][:]  # data in shape [time,lat,lon]
        data_keys = dataset.variables["NUTS keys"][:]  # data in shape [time,lat,lon]
        dataset.close()
        KEY = np.where(data_keys == COUNTRY)
        country_mask = data[KEY[0][0], :, :]
        LONS, LATS = np.meshgrid(lons, lats)
        # plt.pcolor(LONS,LATS,country_mask)
        # plt.show()
        KEY = np.where(data_keys == COUNTRY)
        country_mask = data[KEY[0][0], :, :]

    elif NUTS_lev == 8:
        dataset = Dataset(
            "/home/users/zd907959/code_folders/CLEARHEADS/apply_country_masks/EEZ_regions/EEZ_selected_masks.nc",
            mode="r",
        )
        lons = dataset.variables["lon"][:]
        lats = dataset.variables["lat"][:]
        data = dataset.variables["EEZ zones"][:]  # data in shape [time,lat,lon]
        data_keys = dataset.variables["EEZ keys"][:]  # data in shape [time,lat,lon]
        dataset.close()
        KEY = np.where(data_keys == COUNTRY)
        country_mask = data[KEY[0][0], :, :]
        # plt.pcolormesh(country_mask)
        # plt.show()

    elif NUTS_lev == 9:
        dataset = Dataset(
            "/home/users/zd907959/code_folders/CLEARHEADS/apply_country_masks/EEZ_regions/EEZ_and_Shipping_zones_31_masks.nc",
            mode="r",
        )
        lons = dataset.variables["lon"][:]
        lats = dataset.variables["lat"][:]
        if NUTS9_country_choice == "United Kingdom":
            data = dataset.variables["SHIP_zones_UK"][:]  # data in shape [time,lat,lon]
        if NUTS9_country_choice == "Ireland":
            data = dataset.variables["SHIP_zones_IE"][:]  # data in shape [time,lat,lon]
        if NUTS9_country_choice == "Norway":
            data = dataset.variables["SHIP_zones_NO"][:]  # data in shape [time,lat,lon]
        data_keys = dataset.variables["SHIP_keys"][:]  # data in shape [time,lat,lon]
        dataset.close()
        KEY = np.where(data_keys == COUNTRY)
        country_mask = data[KEY[0][0], :, :]
        # plt.pcolormesh(country_mask)
        # plt.show()
    else:
        print("NUTS zone not known")

    return country_mask


def load_appropriate_data(
    field_str,
    year,
    month,
    country_mask,
    ons_ofs,
    wp_weights,
    pop_weights,
    sp_weights,
    NUTS_lev,
    CC_flag,
    MODEL_NAME,
    daily_flag=False,
):
    """
    This function loads in the climate data you need to do the processing and converts into a timeseries of area-aggregated data.

    Args:
        field_str (str): Decides what will be loaded. options: 't2m','speed100m','speed10m','ssrd' , 'sp' , 'wp'
        year (int): a year from 1950-2020
        month (int) : a month from 1-12.
        country_mask (array): dim [lat,lon] gives the area mask that the data is weighted by.
        ons_ofs (str) : 'ons' or 'ofs' to represent if onshore or oshore wind (may appear redundant for some fields.
        wp_weights (int): either 0 (no) or 1 (yes) to include the locations of European wind farms from thewindpower.net
        pop_weights (int): either 0 (no) or 1 (yes) to include the population weighting
        sp_weights (int): either 0 (no) or 1 (yes) to include the locations of European solar pannels (better data for UK region)
        NUTS_lev (int): 0,1,2,8,9 representing the NUTS level. (see main script for details)
        CC_flag (int) : either 0 (no) or 1 (yes) to include the delta correction to the ERA5 data.
        MODEL_NAME (str) : options of 'ERA5', 'EC-EARTH3P','EC-EARTH3P-HR','MOHC_HH_3hr', 'MOHC_MM-1hr','MOHC_MM-1hr2',
        daily_flag (bool): if True, the data is daily, if False, the data is hourly

    Returns:
        country_agg (array): dim [time], the country aggregated time series of the variable you requested.
    """
    if field_str in ["t2m", "speed10m", "speed100m", "ssrd"]:
        file_loc, file_name = find_data_path(year, month, field_str, daily_flag)
        if CC_flag == 1:
            data = load_country_weather_data_with_BC(
                country_mask,
                file_loc,
                file_name,
                field_str,
                ons_ofs,
                CC_flag,
                MODEL_NAME,
                month,
            )
        else:
            data = load_country_weather_data(
                country_mask, file_loc, file_name, field_str, ons_ofs, MODEL_NAME, month
            )

    elif field_str in ["sp"]:
        file_loc1, file_name1 = find_data_path(year, month, "t2m", daily_flag)
        file_loc2, file_name2 = find_data_path(year, month, "ssrd", daily_flag)
        data = create_solar_power(
            country_mask,
            file_loc1,
            file_name1,
            file_loc2,
            file_name2,
            month,
            year,
            ons_ofs,
            CC_flag,
            MODEL_NAME,
        )
        data[data <= 0.0] = np.nan

    elif field_str in ["wp"]:
        file_loc1, file_name1 = find_data_path(year, month, "speed100m", daily_flag)

        if wp_weights == 0:
            # make an equal area weighting for if weights are not used
            total_MW = np.zeros_like(country_mask) + 1
            if CC_flag == 1:
                total_MW = total_MW[50:213, 107:304]
            lats = np.zeros(np.shape(country_mask)[0])
            lons = np.zeros(np.shape(country_mask)[1])
            data = create_wind_power(
                country_mask,
                file_loc1,
                file_name1,
                ons_ofs,
                lats,
                lons,
                total_MW,
                wp_weights,
                NUTS_lev,
                CC_flag,
                MODEL_NAME,
                month,
            )
        if wp_weights == 1:
            # the country mask is the weighted wind power distribution at this point.
            if CC_flag == 1:
                total_MW = country_mask[50:213, 107:304]
            else:
                total_MW = country_mask

            # plt.imshow(country_mask)
            # plt.title('is it still ok?')
            # plt.show()
            lats = np.zeros(np.shape(country_mask)[0])
            lons = np.zeros(np.shape(country_mask)[1])
            data = create_wind_power(
                country_mask,
                file_loc1,
                file_name1,
                ons_ofs,
                lats,
                lons,
                total_MW,
                wp_weights,
                NUTS_lev,
                CC_flag,
                MODEL_NAME,
                month,
            )  # repeat country mask to fill gaps?
            # plt.pcolormesh(data[0,:,:])
            # plt.colorbar()
            # plt.show()
            # plt.imshow(country_mask)
            # plt.title('mask')
            # plt.colorbar()
            # plt.show()

        # set zeros to NaNs where the country mask has been applied
        data[data == 0.0] = np.nan

    else:
        print("field string not known")

    # if no weights are used anywhere:
    if np.sum(pop_weights + sp_weights + wp_weights) == 0:
        country_agg = np.nanmean(np.nanmean(data, axis=2), axis=1)
        # if weights are usef somewhere
    elif np.sum(pop_weights + sp_weights + wp_weights) > 0:
        country_agg = np.nansum(np.nansum(data, axis=2), axis=1) / np.nansum(
            country_mask
        )
    else:
        print("we have a problem?")

    # make sure any zeros are nans before returning
    country_agg[country_agg == 0.0] = np.nan
    return country_agg
