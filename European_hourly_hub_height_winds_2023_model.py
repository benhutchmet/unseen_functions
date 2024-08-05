"""
Hannah Bloomfield code
for live reanalysis project
broadly based on CLEARHEADS/CCC work.

Trying to get this to work with decadal predictions stuff.
"""

import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.quickplot as qplt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.dates import ConciseDateConverter
import cftime
import matplotlib.units as munits

munits.registry[cftime.DatetimeGregorian] = ConciseDateConverter()
import datetime
from datetime import timedelta
from iris.coord_categorisation import add_categorised_coord
from iris.coord_categorisation import _pt_date
from iris.coord_categorisation import add_day_of_year
import pandas as pd

# steps:

# make country mask so we know if somewhere is onshore or offshore.
# create GB wind turbine locations
# load in the power curves.
# load in wind speeds and take them to hub-height.
# make wind power
# aggregate to country level.


def country_mask(data_dir, test_str, COND, COUNTRY):

    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    dataset = iris.load(data_dir + test_str, COND)
    LONS, LATS = iris.analysis.cartography.get_xy_grids(dataset[0])
    print(np.shape(LONS))
    x, y = LONS.flatten(), LATS.flatten()
    points = np.vstack((x, y)).T

    MASK_MATRIX_TEMP = np.zeros((len(x), 1))
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"][0:14] == COUNTRY:
            print("Found Country " + COUNTRY)
            country_shapely.append(country.geometry)

    print("making mask")
    for i in range(0, len(x)):
        my_point = shapely.geometry.Point(x[i], y[i])
        if country_shapely[0].contains(my_point) == True:
            MASK_MATRIX_TEMP[i, 0] = 1.0

    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX_TEMP, (np.shape(LONS)))
    print(np.shape(MASK_MATRIX_RESHAPE))

    return MASK_MATRIX_RESHAPE, LONS, LATS


####
# load in power curve
#####
def load_power_curves(path_onshore_curve, path_offshore_curve):
    # inputs:
    # powe curves .csv file
    # Outputs:
    # interpolated power curves.

    pc_w_ons = []
    pc_p_ons = []

    with open(path_onshore_curve) as f:
        for line in f:
            columns = line.split()
            # print columns[0]
            pc_p_ons.append(float(columns[1][0:8]))  # get power curve output (CF)
            pc_w_ons.append(float(columns[0][0:8]))  # get power curve output (CF)

    power_curve_w_ons = np.array(pc_w_ons)
    power_curve_p_ons = np.array(pc_p_ons)

    pc_winds = np.linspace(0, 50, 501)  # make it finer resolution
    pc_power_onshore = np.interp(pc_winds, power_curve_w_ons, power_curve_p_ons)

    pc_w_ofs = []
    pc_p_ofs = []

    with open(path_offshore_curve) as f:
        for line in f:
            columns = line.split()
            # print columns[0]
            pc_p_ofs.append(float(columns[1][0:8]))  # get power curve output (CF)
            pc_w_ofs.append(float(columns[0][0:8]))  # get power curve output (CF)

    power_curve_w_ofs = np.array(pc_w_ofs)
    power_curve_p_ofs = np.array(pc_p_ofs)

    pc_winds = np.linspace(0, 50, 501)  # make it finer resolution
    pc_power_offshore = np.interp(pc_winds, power_curve_w_ofs, power_curve_p_ofs)

    return pc_winds, pc_power_onshore, pc_power_offshore


####
# load in wind turbine locations
#####


def load_wind_farm_location(path_to_farm_locations, path_to_sample_datafile, cube_cond):

    # load in information about where the wind farms are.
    wind_farms = iris.load_cube(path_to_farm_locations)
    # rename coordinates and variables so we can get_xy_grids
    wind_farms.coord("lat").rename("latitude")
    wind_farms.coord("lon").rename("longitude")

    # get lats and lons
    LONS_wp, LATS_wp = iris.analysis.cartography.get_xy_grids(wind_farms)
    # add onshore and offshore together.
    wind_farms_total = wind_farms
    # This allows us to calcuate area weights and so that units all match up for regridding.
    for coordinate in ["latitude", "longitude"]:
        wind_farms_total.coord(coordinate).units = "degrees"
        wind_farms_total.coord(coordinate).guess_bounds()

    data = iris.load_cube(path_to_sample_datafile, cube_cond)

    LONS, LATS = iris.analysis.cartography.get_xy_grids(data)

    # for regridding and area weights
    data.coord("latitude").guess_bounds()
    data.coord("longitude").guess_bounds()
    # get area weights of target grid
    data_weights = iris.analysis.cartography.area_weights(data[0, :, :])
    # to do a sum rather than an average of wind farms you need to multiply wind farms_total by its area weights and divide by weights of old grid.
    weights = iris.analysis.cartography.area_weights(wind_farms_total)
    wind_farms_total_over_area = wind_farms_total / weights
    regridded_windfarms_total = wind_farms_total_over_area.regrid(
        data, iris.analysis.AreaWeighted()
    )
    # undo the divide by dA
    regridded_windfarms_total_corrected = regridded_windfarms_total * data_weights
    return regridded_windfarms_total_corrected.data


def load_wind_speed_and_take_to_hubheight(
    path_to_wind_speed: str,
    C1: str,
    C2: str,
    landmask: np.ndarray,
    height_of_wind_speed: float,
    corrected_var_name: str = "si100_bc"
):
    """
    Load wind speed data from a specified path and adjust it to a specified hub height.

    Parameters:
    path_to_wind_speed (str): The path to the file containing the wind speed data.
    C1 (str): The first correction factor for adjusting the wind speed to the hub height.
    C2 (str): The second correction factor for adjusting the wind speed to the hub height.
    landmask (np.ndarray): A land mask array to apply to the wind speed data.
    height_of_wind_speed (float): The height at which the original wind speed data was measured.
    corrected_var_name (str, optional): The name of the variable for the corrected wind speed data. Defaults to "si100_bc".

    Returns:
    corrected_wind_speed (np.ndarray): The wind speed data adjusted to the hub height.
    """

    # load in the wind speed data
    print("loading wind speed data")
    data = iris.load(path_to_wind_speed)

    # if the data has two variables, then remove the "si100_bc"
    if len(data) == 2 and corrected_var_name in [cube.name() for cube in data]:
        print("removing corrected wind speed data")
        data = [cube for cube in data if cube.name() != corrected_var_name]

    # if the corrected variable name is not in the data, concatenate the cubes
    if corrected_var_name not in [cube.name() for cube in data]:
        print("calculating corrected wind speed data from U and V components")
        COND1 = iris.Constraint(C1)
        COND2 = iris.Constraint(C2)

        # if the variable "si100" is not in the data, load in the U and V components and concatenate them
        if "si100" not in [cube.name() for cube in data]:
            data_u = iris.load(path_to_wind_speed, COND1)
            data_v = iris.load(path_to_wind_speed, COND2)
            equalised_cubes_u = iris.util.equalise_attributes(data_u)
            equalised_cubes_v = iris.util.equalise_attributes(data_v)
            conc_cubes_u = data_u.concatenate_cube()
            conc_cubes_v = data_v.concatenate_cube()

            # turn into 10m wind speed and get up to hub-height.
            speed = (conc_cubes_u * conc_cubes_u + conc_cubes_v * conc_cubes_v) ** 0.5
            print(speed)
            print("concatenating cubes")

            print(np.shape(speed)[0])
        elif "si100" in [cube.name() for cube in data]:
            data_si100 = iris.load(path_to_wind_speed, "si100")
            data_si100 = data_si100.concatenate_cube()
            speed = data_si100
        else:
            raise ValueError("The wind speed data does not contain the required variables.")

        correction_hubheight = landmask * (71.0 / height_of_wind_speed) ** (
            1.0 / 7.0
        ) + abs(landmask - 1) * (92.0 / height_of_wind_speed) ** (1.0 / 7.0)

        speed_hubheight = speed * correction_hubheight

        # load in the Global wind atlas 100m correction and apply.
        # have checked adding is right, as wind speeds are too low over orography...so need to add on the potitve bits in the array loaded here.
        # GWA_correction = np.load(
        #     "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/GWA_correction_factor_on_ERA5_grid.npy"
        # )
        # slightly older version of this with correct shape
        GWA_correction = np.load(
            "/home/users/benhutch/UREAD_energy_models_demo_scripts/ERA5_speed100m_mean_factor_v16_hourly.npy"
        )

        GWA_broadcast = iris.util.broadcast_to_shape(
            GWA_correction, np.shape(speed_hubheight), (1, 2)
        )

        corrected_speed_hubheight = (
            speed_hubheight + GWA_correction
        )  # add in the bias correction stage.

        speed_hubheight.rename("Hub Height Wind Speed")
        speed_hubheight.units = "m s-1"
    else:
        print("loading corrected wind speed data")
        data_si100_bc = iris.load(path_to_wind_speed, corrected_var_name)

        # concatenate the cubes
        data_si100_bc = data_si100_bc.concatenate_cube()

        # # get the speed
        # speed = data_si100_bc

        # # print the shape of data_si100_bc
        print(data_si100_bc)

        print(np.shape(data_si100_bc)[0])

        # quantify the hub-height correction factor
        correction_hubheight = landmask * (71.0 / height_of_wind_speed) ** (
            1.0 / 7.0
        ) + abs(landmask - 1) * (92.0 / height_of_wind_speed) ** (1.0 / 7.0)

        # extract the data from the corrected wind speed cube
        speed_hubheight = data_si100_bc.data

        # multiply the speed hubheight by the correction hubheight
        corrected_speed_hubheight = speed_hubheight * correction_hubheight

        # create a new cube with the corrected wind speed data
        # but with the same metadata as the original cube
        corrected_speed_hubheight = data_si100_bc.copy(data=corrected_speed_hubheight)

    return corrected_speed_hubheight


def convert_to_wind_power(
    pc_winds,
    pc_power_onshore,
    pc_power_offshore,
    speed_hubheight,
    regridded_windfarms_total_corrected,
    landmask,
):
    
    test = np.digitize(
        speed_hubheight.data, pc_winds, right=False
    )  # indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin.
    test[test == len(pc_winds)] = 500  # make sure the bins don't go off the
    # end (power is zero by then anyway)
    # [time, lat, lon]

    p_hh_temp1 = landmask * 0.5 * (pc_power_onshore[test - 1] + pc_power_onshore[test])

    p_hh_temp2 = (
        abs(landmask - 1)
        * 0.5
        * (pc_power_offshore[test - 1] + pc_power_offshore[test])
    )

    p_hh_temp = p_hh_temp1 + p_hh_temp2

    # get timeseries accumilated over country
    phh_in_GW = (
        p_hh_temp * regridded_windfarms_total_corrected
    ) / 1000.0  # get it in GW
    phh_cube = speed_hubheight.copy(data=phh_in_GW)
    # rename the cube in preparation for conversion to CF later.
    phh_cube.rename("Wind Power Capacity factor")
    phh_cube.units = "%"

    # do the same but for onshore and offshore seperately
    # for validation
    phh_in_GW_onshore = (
        p_hh_temp1 * regridded_windfarms_total_corrected
    ) / 1000.0  # get it in GW
    phh_cube_onshore = speed_hubheight.copy(data=phh_in_GW_onshore)
    # rename the cube in preparation for conversion to CF later.
    phh_cube_onshore.rename("Wind Power Capacity factor")
    phh_cube_onshore.units = "%"

    phh_in_GW_offshore = (
        p_hh_temp2 * regridded_windfarms_total_corrected
    ) / 1000.0  # get it in GW
    phh_cube_offshore = speed_hubheight.copy(data=phh_in_GW_offshore)
    # rename the cube in preparation for conversion to CF later.
    phh_cube_offshore.rename("Wind Power Capacity factor")

    WP_country_level_combined = phh_cube.collapsed(["latitude", "longitude"], iris.analysis.SUM)
    # turn into CF by dividing by the total installed CF.

    WP_country_level_onshore = phh_cube_onshore.collapsed(["latitude", "longitude"], iris.analysis.SUM)

    WP_country_level_offshore = phh_cube_offshore.collapsed(["latitude", "longitude"], iris.analysis.SUM)

    return WP_country_level_combined, WP_country_level_onshore, WP_country_level_offshore


def main():
    print(
        "loading ERA5 one year at a time to create wind power and other useful variables"
    )

    # directories and information common to each year and country:
    # load in the natural earth data
    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    # directories and information common to each month of the dataset:
    countries = [
        "United Kingdom"
    ]  # ["Austria","Belgium","Bulgaria","Croatia","Czechia","Denmark","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Montenegro","Netherlands","Norway","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","Switzerland"]
    # ['United Kingdom']

    timeperiod = "current"  # ['current or 'planned'] or 'approved if UK

    # load in the power curves.
    pc_winds, pc_power_onshore, pc_power_offshore = load_power_curves(
        "/home/users/hbloomfield01/UKMO_pilot/ERA5_wind_power_model/power_onshore.csv",
        "/home/users/hbloomfield01/UKMO_pilot/ERA5_wind_power_model/power_offshore.csv",
    )

    for COUNTRY in countries:

        if COUNTRY == "Czechia":
            COUNTRY_nospace = "Czech_Republic"
        else:
            COUNTRY_nospace = COUNTRY.replace(" ", "_")

        # make country mask
        data_dir = "/gws/pw/j07/ceraf/hbloomfield01/Data/ERA5/big_uv100m_for_CCC/"
        test_str = "/ERA5_EU_1hr_uv100m_1940_01.nc"
        MASK_MATRIX_RESHAPE, LONS, LATS = country_mask(
            data_dir, test_str, "100 metre U wind component", COUNTRY
        )

        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        # plt.pcolormesh(LONS,LATS,MASK_MATRIX_RESHAPE,cmap='YlGn',transform=ccrs.PlateCarree())
        # plt.title('Land Mask',fontsize=16)
        # ax.set_aspect('auto',adjustable=None)
        # ax.coastlines(resolution='50m')
        # ax.add_feature(cf.BORDERS)

        # load in the wind farm locations

        regridded_windfarms_total_corrected = load_wind_farm_location(
            "/home/users/hbloomfield01/UKMO_pilot/ERA5_wind_power_model/"
            + str(COUNTRY_nospace)
            + "windfarm_dist_"
            + str(timeperiod)
            + ".nc",
            data_dir + test_str,
            "100 metre U wind component",
        )

        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        # plt.pcolormesh(LONS,LATS,regridded_windfarms_total_corrected,cmap='YlGn',transform=ccrs.PlateCarree())
        # plt.title('Land Mask',fontsize=16)
        # ax.set_aspect('auto',adjustable=None)
        # ax.coastlines(resolution='50m')
        # ax.add_feature(cf.BORDERS)

        for YEAR in range(1940, 2023):  # (1940,2023):
            print(YEAR)
            data_str = (
                "/gws/pw/j07/ceraf/hbloomfield01/Data/ERA5/big_uv100m_for_CCC/ERA5_EU_1hr_uv100m_"
                + str(YEAR)
                + "*.nc"
            )  # 2019_03.nc

            print("loading cubes")
            speed_hubheight = load_wind_speed_and_take_to_hubheight(
                data_str,
                "100 metre U wind component",
                "100 metre V wind component",
                MASK_MATRIX_RESHAPE,
                100.0,
            )
            # print(speed_hubheight)

            WP_data = convert_to_wind_power(
                pc_winds,
                pc_power_onshore,
                pc_power_offshore,
                speed_hubheight,
                regridded_windfarms_total_corrected,
                MASK_MATRIX_RESHAPE,
            )

            weights_broadcast = iris.util.broadcast_to_shape(
                regridded_windfarms_total_corrected, np.shape(speed_hubheight), (1, 2)
            )

            speed_hubheight_timeseries = speed_hubheight.collapsed(
                ["latitude", "longitude"], iris.analysis.MEAN, weights=weights_broadcast
            )

            # convert to GW
            WP_cf_timeseries = (WP_data / 1000.0) / (
                np.sum(regridded_windfarms_total_corrected) / 1000000
            )
            iris.save(
                WP_cf_timeseries,
                "/gws/pw/j07/ceraf/hbloomfield01/Data/ERA5/ERA5_demand_and_national_aggs/ERA5_"
                + str(COUNTRY_nospace)
                + "_"
                + str(timeperiod)
                + "_"
                + str(YEAR)
                + "_WP_timeseries.nc",
            )
            iris.save(
                speed_hubheight_timeseries,
                "/gws/pw/j07/ceraf/hbloomfield01/Data/ERA5/ERA5_demand_and_national_aggs/ERA5_"
                + str(COUNTRY_nospace)
                + "_"
                + str(timeperiod)
                + "_"
                + str(YEAR)
                + "_hub_height_speed_timeseries.nc",
            )


if __name__ == "__main__":
    main()
