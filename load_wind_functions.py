"""
Functions for loading in the wind data from CLEARHEADS and S2S4E.

Thanks to Hannah for downloading all of the data!
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
import xarray as xr
import iris
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import regionmask
from tqdm import tqdm
import cftime

# Specific imports
from ncdata.iris_xarray import cubes_to_xarray, cubes_from_xarray

# Import dictionaries
import unseen_dictionaries as dicts


# Define the function for preprocessing the data
def preprocess(
    ds: xr.Dataset,
    u100_name="u100",
    v100_name="v100",
    si100_name="si100",
    u10_name="u10",
    v10_name="v10",
    si10_name="si10",
    t2m_name="t2m",
    msl_name="msl",
) -> xr.Dataset:
    """
    Preprocess the data.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be preprocessed.

    u100_name : str
        The name of the zonal wind component.

    v100_name : str
        The name of the meridional wind component.

    si100_name : str
        The name of the wind speed at 100m to be output.

    u10_name : str
        The name of the zonal wind component at 10m.

    v10_name : str
        The name of the meridional wind component at 10m.

    si10_name : str
        The name of the wind speed at 10m to be output.

    t2m_name : str
        The name of the 2m temperature.

    msl_name : str
        The name of the mean sea level pressure.

    Returns
    -------

    ds : xarray.Dataset

    """

    # Calculate the wind speed at 100m
    ds[si100_name] = np.sqrt(ds[u100_name] ** 2 + ds[v100_name] ** 2)

    # Calculate the wind speed at 10m
    ds[si10_name] = np.sqrt(ds[u10_name] ** 2 + ds[v10_name] ** 2)

    # Drop the other variables
    ds = ds.drop_vars([u100_name, v100_name, u10_name, v10_name, t2m_name, msl_name])

    return ds


# Define a function to load the 100m wind speed data from the CLEARHEADS and S2S4E directories
# S2S4E - ERA5_1hr_2020_12_DET.nc
# CLEARHEADS - ERA5_1hr_1978_12_DET.nc
def load_obs_data(
    last_year: int,
    last_month: int = 12,
    first_year: int = 1950,
    first_month: int = 1,
    S2S4E_dir: str = "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/",
    CLEARHEADS_dir: str = "/storage/silver/clearheads/Data/ERA5_data/native_grid/T2m_U100m_V100m_MSLP/",
    engine: str = "netcdf4",
    parallel: bool = True,
    bias_correct_wind: bool = True,
    bias_correct_file: str = "/home/users/pn832950/UREAD_energy_models_demo_scripts/ERA5_speed100m_mean_factor_v16_hourly.npy",
    preprocess: callable = preprocess,
):
    """
    Load the 100m wind speed data from the CLEARHEADS and S2S4E directories.

    Parameters
    ----------

    last_year : int
        The last year of the data to be loaded.

    last_month : int
        The last month of the data to be loaded.

    first_year : int
        The first year of the data to be loaded.

    first_month : int
        The first month of the data to be loaded.

    S2S4E_dir : str
        The directory containing the S2S4E data.

    CLEARHEADS_dir : str
        The directory containing the CLEARHEADS data.

    engine : str
        The engine to use for loading the data.

    parallel : bool
        Whether to use parallel loading.

    bias_correct : bool
        Whether to use Hannah's bias correction for onshore and offshore wind speeds.

    bias_correct_file : str
        The file containing the bias correction data.

    Returns
    -------
    ds : xarray.Dataset
        The 100m and 10m wind speed data.
    """
    # Create an empty list to store the data
    ERA5_files = []

    # create a list of the files to load based on the years and months provided
    for year in range(first_year, last_year + 1):
        if year == last_year:
            for month in range(1, last_month + 1):
                if month < 10:
                    month = f"0{month}"
                else:
                    month = f"{month}"
                # Choose the directory based on the year
                directory = CLEARHEADS_dir if year < 1979 else S2S4E_dir
                # glob the files in the chosen directory
                for file in glob.glob(directory + f"ERA5_1hr*{year}_{month}*DET.nc"):
                    ERA5_files.append(file)
        else:
            for month in range(1, 13):
                if month < 10:
                    month = f"0{month}"
                else:
                    month = f"{month}"
                # Choose the directory based on the year
                directory = CLEARHEADS_dir if year < 1979 else S2S4E_dir
                # glob the files in the chosen directory
                for file in glob.glob(directory + f"ERA5_1hr*{year}_{month}*DET.nc"):
                    ERA5_files.append(file)

    # Print the length of the list
    print("Number of files: ", len(ERA5_files))

    # Load the data
    ds = xr.open_mfdataset(
        ERA5_files,
        combine="by_coords",
        preprocess=lambda ds: preprocess(ds),
        engine=engine,
        parallel=parallel,
        coords="minimal",
        data_vars="minimal",
        compat="override",
    ).squeeze()

    # chunk the data
    ds = ds.chunk({"time": "auto", "latitude": "auto", "longitude": "auto"})

    # Take a daily mean
    ds = ds.resample(time="D").mean()

    # if bias correction is required
    if bias_correct_wind:
        # Load the bias correction data
        bc = np.load(bias_correct_file)

        # Convert the DataArrays to numpy arrays
        si100_name_np = ds["si100"].values
        bc_totals_np = bc

        # create a new numpy array to store the result
        si100_bc_np = np.zeros(np.shape(si100_name_np))

        # Perform the addition
        # TODO: Is there an issue with bias correcting daily data here?
        for i in tqdm(
            range(np.shape(si100_name_np)[0]), desc="Applying bias correction"
        ):
            si100_bc_np[i, :, :] = si100_name_np[i, :, :] + bc_totals_np

        # Convert the result back to an xarray DataArray
        si100_bc = xr.DataArray(
            data=si100_bc_np,
            dims=ds["si100"].dims,
            coords=ds["si100"].coords,
        )

        # Add the new DataArray to the dataset
        ds = ds.assign(si100_bc=si100_bc)

        # Set up the variables
        # Power law exponent from UK windpower.net 2021 onshore wind farm heights
        ds["si100_ons"] = ds["si100_bc"] * (71.0 / 100.0) ** (1.0 / 7.0)

        # Same but for offshore
        # Average height of offshore wind farms
        ds["si100_ofs"] = ds["si100_bc"] * (92.0 / 100.0) ** (1.0 / 7.0)

        # Drop si100 in favour of si100_ons and si100_ofs
        ds = ds.drop_vars(["si100"])

    return ds


# define a function to preprocess rsds
def preprocess_rsds(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Preprocess the rsds data.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be preprocessed.

    Returns
    -------

    ds : xarray.Dataset
        The preprocessed dataset.

    """

    return ds


# define another preprocessing function for temperature
def preprocess_temp(
    ds: xr.Dataset,
    msl_name="msl",
    u100_name="u100",
    v100_name="v100",
    u10_name="u10",
    v10_name="v10",
) -> xr.Dataset:
    """
    Preprocess the temperature data.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be preprocessed.

    msl_name : str
        The name of the mean sea level pressure.

    u100_name : str
        The name of the zonal wind component at 100m.

    v100_name : str
        The name of the meridional wind component at 100m.

    u10_name : str
        The name of the zonal wind component at 10m.

    v10_name : str
        The name of the meridional wind component at 10m.

    Returns
    -------

    ds : xarray.Dataset
        The preprocessed dataset.

    """

    # Drop the mean sea level pressure
    ds = ds.drop_vars([msl_name, u100_name, v100_name, u10_name, v10_name])

    return ds


# Define a function which applies a mask to the data
def apply_country_mask(
    ds: xr.Dataset,
    country: str,
    pop_weights: int = 0,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
) -> xr.Dataset:
    """
    Apply a mask to the data for a specific country.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be masked.

    country : str
        The country to be masked.

    pop_weights : int
        The population weights to be applied.

    Returns
    -------

    ds : xarray.Dataset
        The masked dataset.

    """

    # Identify an appropriate shapefile for the country
    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    # Find the country
    country_shp = None
    for country_shp in shpreader.Reader(countries_shp).records():
        if country_shp.attributes["NAME_LONG"] == country:
            print("Found the country!")

            # Load using geopandas
            country_shp_gp = gpd.read_file(countries_shp)

            # Filter the dataframe to only include the row for the UK
            country_shp_gp = country_shp_gp[country_shp_gp["NAME_LONG"] == country]

    # Ensure that the 'numbers' column exists in the geodataframe
    if "numbers" not in country_shp_gp.columns:
        country_shp_gp["numbers"] = np.array([1])

    # Create the mask using the regionmask and geopandas
    country_mask_poly = regionmask.from_geopandas(
        country_shp_gp,
        names="NAME_LONG",
        abbrevs="ABBREV",
        numbers="numbers",
    )

    # # Print the mask
    # print(f"Country mask: {country_mask_poly}")

    # # print the data
    # print(ds)

    # # print ds.isel(time=0)
    # print(ds.isel(time=0))

    print("Pre-country mask")

    # if time is not a dimension in ds
    if "time" not in ds.dims:
        ds_init = ds.isel(
            init=0,
            lead=0,
            member=0,
        )
    else:
        # seelct the first time
        ds_init = ds.isel(time=0)

    # print ds_init
    print(ds_init)

    # print ds_init[lon_name]
    print(ds_init[lon_name].values)

    # print ds_init[lat_name]
    print(ds_init[lat_name].values)

    # Select the first timestep of the data
    country_mask = country_mask_poly.mask(
        ds_init[lon_name].values, ds_init[lat_name].values
    )

    # print the country mask
    print(f"Country mask: {country_mask}")

    # print the country mask coords
    print(f"Country mask coords: {country_mask.coords}")

    # if longitude and latitude are not in the coords
    if lon_name not in country_mask.coords:
        lon_name = "lon"

    if lat_name not in country_mask.coords:
        lat_name = "lat"

    if country == "United Kingdom":
        print("Masking out Northern Ireland.")
        # If the country is the UK then mask out Northern Ireland
        country_mask = country_mask.where(
            ~(
                (country_mask[lat_name] < 55.3)
                & (country_mask[lat_name] > 54.0)
                & (country_mask[lon_name] < -5.0)
            ),
            other=np.nan,
        )

    # # print the country mask
    # print(f"Country mask: {country_mask}")

    # Extract the lat and lon values
    mask_lats = country_mask[lat_name].values
    mask_lons = country_mask[lon_name].values

    ID_REGION = 1  # only 1 region in this instance

    # Select mask for specific region
    sel_country_mask = country_mask.where(country_mask == ID_REGION).values

    # # # print the selected country mask
    # print(f"Selected country mask: {sel_country_mask}")

    # print ds
    print(ds)

    # Select the data within the mask
    out_ds = ds.compute().where(sel_country_mask == ID_REGION)

    # print the data
    print("output dataset:", out_ds)

    # # print that we are exiting the script
    # print("Exiting the script.")
    # sys.exit()

    return out_ds


# define a function to return a country mask with 1's where the country is
# and 0's where it isn't
def load_country_mask(
    ds: xr.Dataset,
    country: str,
    pop_weights: int = 0,
) -> xr.Dataset:
    """
    Load a mask for a specific country.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be masked.

    country : str
        The country to be masked.

    pop_weights : int
        The population weights to be applied.

    Returns
    -------

    ds : xarray.Dataset
        The masked dataset.

    """

    # Identify an appropriate shapefile for the country
    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    # Find the country
    country_shp = None
    for country_shp in shpreader.Reader(countries_shp).records():
        if country_shp.attributes["NAME_LONG"] == country:
            print("Found the country!")

            # Load using geopandas
            country_shp_gp = gpd.read_file(countries_shp)

            # Filter the dataframe to only include the row for the UK
            country_shp_gp = country_shp_gp[country_shp_gp["NAME_LONG"] == country]

    # Ensure that the 'numbers' column exists in the geodataframe
    if "numbers" not in country_shp_gp.columns:
        country_shp_gp["numbers"] = np.array([1])

    # Create the mask using the regionmask and geopandas
    country_mask_poly = regionmask.from_geopandas(
        country_shp_gp,
        names="NAME_LONG",
        abbrevs="ABBREV",
        numbers="numbers",
    )

    # Create a mask for the dataset
    country_mask_ds = country_mask_poly.mask(
        ds.isel(time=0), lon_name="longitude", lat_name="latitude"
    )

    if country == "United Kingdom":
        print("Masking out Northern Ireland.")
        # If the country is the UK then mask out Northern Ireland
        country_mask_ds = country_mask_ds.where(
            ~(
                (country_mask_ds.latitude < 55.3)
                & (country_mask_ds.latitude > 54.0)
                & (country_mask_ds.longitude < -5.0)
            ),
            other=np.nan,
        )

    # extract the values of this mask
    country_mask_vals = country_mask_ds.values

    # Create a mask that is True where country_mask_vals is NaN
    nan_mask = np.isnan(country_mask_vals)

    # Set the NaN values in country_mask_vals to 0
    country_mask_vals[nan_mask] = 0

    # Set the non-zero values to 1
    country_mask_vals[country_mask_vals != 0] = 1

    # Print the sum total of tyhe new values (to check that they are all 1's and 0's)
    print(f"Sum of new country mask values: {str(np.sum(country_mask_vals))}")

    return country_mask_vals


# define a function which saves the data
def save_wind_data(
    ds: xr.Dataset,
    output_dir: str,
    fname: str,
) -> None:
    """
    Save the wind data to a netCDF file.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be saved.

    output_dir : str
        The directory to save the file in.

    fname : str
        The name of the file to be saved.

    Returns
    -------

    None
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the data
    ds.to_netcdf(os.path.join(output_dir, fname))

    return None


# Define a function to create the wind power data
# FIXME: Find the correct onshore and offshore power curves here
# FIXME: Find the correct installed capacity data here
def create_wind_power_data(
    ds: xr.Dataset,
    country: str = "United_Kingdom",
    ons_ofs: str = "ons",
    var_name: str = "si100_bc",
    min_cf: float = 0.0006,
    onshore_curve_file: str = "/home/users/pn832950/100m_wind/power_curve/powercurve.csv",
    offshore_curve_file: str = "/home/users/pn832950/100m_wind/power_curve/powercurve.csv",
    installed_capacities_dir: str = "/storage/silver/S2S4E/zd907959/MERRA2_wind_model/python_version/",
    lat_name: str = "lat",
    lon_name: str = "lon",
    corr_var_name: str = "si100_bc",
    obs_flag: bool = True,
) -> xr.Dataset:
    """
    Loads in datasets containing the 100m wind speed data from ERA5 (in the first
    version) and converts this into an array of wind power generation.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset containing the 100m wind speed data.

    country : str
        The name of the country.
        E.g. "United_Kingdom".

    ons_ofs : str
        The type of wind farm to be considered (either onshore or offshore).

    var_name : str
        The name of the bias corrected 100m wind speed data.
        Default is "si100_bc".

    min_cf : float
        The minimum capacity factor.
        Default is 0.0006.

    onshore_curve_file : str
        The file containing the onshore power curve data.
        Default is "/home/users/pn832950/100m_wind/power_curve/powercurve.csv".
        Random power curve from S2S4E.

    offshore_curve_file : str
        The file containing the offshore power curve data.
        Default is "/home/users/pn832950/100m_wind/power_curve/powercurve.csv".
        Random power curve from S2S4E.

    installed_capacities_idr : str
        The file containing the installed capacities data.
        Default is "/storage/silver/S2S4E/zd907959/MERRA2_wind_model/python_version/".
        Random installed capacities data from S2S4E.

    lat_name : str
        The name of the latitude variable.
        Default is "lat".

    lon_name : str
        The name of the longitude variable.
        Default is "lon".

    corr_var_name : str
        The correct variable name

    obs_flag : bool
        Whether or not this function is processing the observations.

    Returns
    -------

    cfs: np.ndarray
        The array of wind power generation.

    """

    # TODO: Get the correct installed capacities
    # Think this is onshore?
    # Form the filepath
    installed_capacities_file = os.path.join(
        installed_capacities_dir, f"{country}windfarm_dist.nc"
    )

    # depending on the ofs ons flag
    if ons_ofs == "ons":
        print("Loading in the installed capacities for onshore wind farms.")

        # Installed capacties
        installed_capacities_file = os.path.join(
            installed_capacities_dir, f"{country}windfarm_dist_ons_2021.nc"
        )
    elif ons_ofs == "ofs":
        print("Loading in the installed capacities for offshore wind farms.")

        # Installed capacties
        installed_capacities_file = os.path.join(
            installed_capacities_dir, f"{country}windfarm_dist_ofs_2021.nc"
        )
    else:
        print("Invalid wind farm type. Please choose either onshore or offshore.")
        sys.exit()

    # print the installed capacities file
    print(f"loading installed capacities from: {installed_capacities_file}")

    # glob the file
    installed_capacities_files = glob.glob(installed_capacities_file)

    if len(installed_capacities_files) != 1:
        print(f"Installed capacities file not found: {installed_capacities_files}")
        print(f"For country: {country}")

        # Extract the time
        time = ds["time"].values

        # Create an array full of NaNs with length time
        cfs = np.full(len(time), np.nan)

        # Print that we are returning the NaNs
        print(
            f"Returning array of NaNs as installed capacities file not found for country: {country}"
        )

        return cfs

    # assert that the file exists
    assert len(installed_capacities_files) == 1, "Installed capacities file not found."

    # Load the installed capacities data
    installed_capacities = xr.open_dataset(installed_capacities_files[0])

    # # extract the total mw
    totals = installed_capacities["totals"].values

    # extract the total MW
    # 1000kW = 1MW therefore divide by 1000
    # 1000MW = 1GW therefore divide by 1000 to convert from MW to GW
    totals_MW_pre = np.flip(installed_capacities["totals"].values, axis=0) / 1000.0

    # print the total instal;led capacities
    print(f"Installed capacity: {str(np.sum(totals_MW_pre))} for {country}")

    # # print totals
    # print("Totals:", totals)

    # # print the shape of totals
    # print("Totals shape:", totals.shape)

    # return totals_MW

    ic_lat = installed_capacities["lat"].values
    ic_lon = installed_capacities["lon"].values

    ds_lat = ds[lat_name].values
    ds_lon = ds[lon_name].values

    if obs_flag == False:
        # extract the variable from ds
        ds = ds[var_name]

        # select the first lead
        ds_test = ds.isel(lead=0)

        # select the first member
        ds_test = ds_test.isel(member=0)

        # squeze the data
        ds_test = ds_test.squeeze()

        # print ds_test
        print("ds_test:", ds_test)

        # change the variable name from '__xarray_dataarray_variable__'
        ds_test.name = corr_var_name
    else:
        ds_test = ds

    # if the lats and lons are not the same, interpolate the installed capacities
    if not np.array_equal(ic_lat, ds_lat) or not np.array_equal(ic_lon, ds_lon):
        print("Lats and lons are not the same.")
        print(
            "Interpolating installed capacities to the same grid as the wind speed data."
        )

        # convert installed capacities from xarray object to iris object
        # if ds is an xarray Dataarray
        if isinstance(ds_test, xr.DataArray):
            ds_cube = ds_test.to_iris()

        # # print ds_cube
        # print("ds_cube:", ds_cube)

        # convert installed capacities from xarray object to iris object
        ic_cube = cubes_from_xarray(installed_capacities)

        # extract the bc_si100_name cube
        bc_si100_cube = ds_cube

        # extract the totals cube
        ic_cube = ic_cube.extract("totals")[0]

        # promote the latitude and longitude to dimension coordinates
        # bc_si100_cube = bc_si100_cube.rename(
        #     {lat_name: "latitude", lon_name: "longitude"}
        # )

        # # catch for obs data
        if obs_flag == True:
            # promote the latitude and longitude to dimension coordinates
            # promote the latitude and longitude to dimension coordinates
            lat_coord = iris.coords.DimCoord(
                bc_si100_cube.coord(lat_name).points,
                standard_name="latitude",
                units="degrees",
            )
            lon_coord = iris.coords.DimCoord(
                bc_si100_cube.coord(lon_name).points,
                standard_name="longitude",
                units="degrees",
            )

            bc_si100_cube.remove_coord(lat_name)
            bc_si100_cube.remove_coord(lon_name)

            bc_si100_cube.add_dim_coord(lat_coord, 1)
            bc_si100_cube.add_dim_coord(lon_coord, 2)
        else:
            # if the names of bc_si100_cube coords are still lat and lon
            if bc_si100_cube.coords != ic_cube.coords:
                # rename lat and lon to latitude and longitude
                bc_si100_cube.coord("lat").rename("latitude")
                bc_si100_cube.coord("lon").rename("longitude")

        # if the coords
        if ic_cube.coords != bc_si100_cube.coords:
            # rename lat and lon to latitude and longitude
            ic_cube.coord("lat").rename("latitude")
            ic_cube.coord("lon").rename("longitude")

        # Ensure the units of the coordinates match
        ic_cube.coord("latitude").units = bc_si100_cube.coord("latitude").units
        ic_cube.coord("longitude").units = bc_si100_cube.coord("longitude").units

        # Ensure the attributes of the coordinates match
        ic_cube.coord("latitude").attributes = bc_si100_cube.coord(
            "latitude"
        ).attributes
        ic_cube.coord("longitude").attributes = bc_si100_cube.coord(
            "longitude"
        ).attributes

        # print the ic_cube
        print("ic_cube:", ic_cube)

        # print the bc_si100_cube
        print("bc_si100_cube:", bc_si100_cube)

        # regrid the installed capacities to the same grid as the wind speed data
        ic_cube_regrid = ic_cube.regrid(bc_si100_cube, iris.analysis.Linear())

        # regrid the wind speed data to the same grid as the installed capacities
        bc_si100_cube_regrid = bc_si100_cube.regrid(ic_cube, iris.analysis.Linear())

    # Extract the values
    # Flip to get the correct order of lat lon
    totals_MW_regrid = np.flip(ic_cube_regrid.data, axis=0) / 1000.0

    # print the installed capacity
    print(f"Installed capacity: {str(np.sum(totals_MW_regrid))} for {country}")

    # print the shape of the total MW
    print("Total MW pre shape:", totals_MW_pre.shape)
    print("Total MW regrid shape:", totals_MW_regrid.shape)

    # return totals_MW_pre, totals_MW_regrid, bc_si100_cube_regrid, bc_si100_cube

    if obs_flag == False:
        raise NotImplementedError("Model data not yet implemented.")
    else:
        print("Processing the observations into wind power data.")

        # Extract the values from the regridded cube
        bc_si100_vals = bc_si100_cube_regrid.data

        # print the shape of the bc_si100_vals
        print("bc_si100_vals shape:", bc_si100_vals.shape)

        # if ons_ofs is ons
        if ons_ofs == "ons":
            print("Processing the onshore wind power data.")

            # scale down to hub height using a power law
            # Avg height of onshore wind farms 2021 from UK windpower.net
            bc_si100_vals = bc_si100_vals * (71.0 / 100.0) ** (1.0 / 7.0)

            print("Loading in the onshore power curve")

            # Load in the onshore power curve
            power_curve = pd.read_csv(onshore_curve_file, header=None, sep="  ")
        elif ons_ofs == "ofs":
            print("Processing the offshore wind power data.")

            # scale down to hub height using a power law
            # Avg height of offshore wind farms 2021 from UK windpower.net
            bc_si100_vals = bc_si100_vals * (92.0 / 100.0) ** (1.0 / 7.0)

            print("Loading in the offshore power curve.")

            # Load in the offshore power curve
            power_curve = pd.read_csv(offshore_curve_file, header=None, sep="  ")
        else:
            raise ValueError(
                "Invalid wind farm type. Please choose either onshore (ons) or offshore (ofs)."
            )

    # extract the value of totals_MW_pre
    totals_MW = totals_MW_pre

    # where values of the mask are 0. set to Nan
    totals_MW[totals_MW == 0] = np.nan

    # set up a zeros array with the same shape as bc_si100_vals
    wp_dist_masked_data = np.zeros(np.shape(bc_si100_vals))

    # print the shape of the wp_dist_masked_data
    print("wp_dist_masked_data shape:", wp_dist_masked_data.shape)

    # print the shape of the totals_MW
    print("totals_MW shape:", totals_MW.shape)

    # print the shape of bc_si100_vals
    print("bc_si100_vals shape:", bc_si100_vals.shape)

    # create an array of 1's with the same shape as totals_MW
    totals_MW_ones = np.ones_like(totals_MW)

    # # loop over the country masked data
    for i in range(0,len(wp_dist_masked_data)):
        # apply the mask to the wind speed array
        wp_dist_masked_data[i, :, :] = bc_si100_vals[i, :, :] * totals_MW_ones
    
    # print the power curve
    print("Power curve:", power_curve)

    # print the shape of the power curve
    print("Power curve shape:", power_curve.shape)

    # Add column names to the power curve
    power_curve.columns = ["Wind speed (m/s)", "Capacity factors"]

    # Generate an array for wind speeds
    pc_winds = np.linspace(0, 50, 501)

    # print the power curve
    print("Power curve:", power_curve)

    # print the pcwinds
    print("pc_winds:", pc_winds)

    # Using np.interp, find the power output for each wind speed
    pc_power = np.interp(
        pc_winds, power_curve["Wind speed (m/s)"], power_curve["Capacity factors"]
    )

    # Add these to a new dataframe
    pc_df = pd.DataFrame({"Wind speed (m/s)": pc_winds, "Capacity factors": pc_power})

    # print the pc_df
    print("Power curve dataframe:", pc_df.head())

    # get the number of hours from the wp_dist_masked_data
    nhours = np.shape(wp_dist_masked_data)[0]

    # create an array to fill with capacity factors
    cf = np.zeros_like(wp_dist_masked_data)

    # loop over the number of hours
    for i in range(0, nhours):
        wp_dist_masked_data_this = wp_dist_masked_data[i, :, :]

        reshaped_masked_data_this = np.reshape(
                wp_dist_masked_data_this,
                [np.shape(wp_dist_masked_data_this)[0] * np.shape(wp_dist_masked_data_this)[1]],
            )

        # Categorise each wind speed value into a power output
        cf_this = np.digitize(
            reshaped_masked_data_this, pc_df["Wind speed (m/s)"], right=False
        )

        # Make sure the bins don't go out of range
        cf_this[cf_this == len(pc_df)] = len(pc_df) - 1

        # convert pc_df["Capacity_factors] to a numpy array of values
        pc_cf_vals = pc_df["Capacity factors"].values

        # Calculate the average power output for each bin
        cf_bins = 0.5 * (pc_cf_vals[cf_this] + pc_cf_vals[cf_this - 1])

        # Reshape the power output array
        # back to the original shape
        cf_this = np.reshape(
            cf_bins,
            [
                np.shape(wp_dist_masked_data_this)[0],
                np.shape(wp_dist_masked_data_this)[1],
            ],
        )

        # set any values below the minimum capacity factor to the minimum capacity factor
        cf_this[cf_this < min_cf] = 0.0

        # mutlipy by the weighting (again?)
        cf[i, :, :] = cf_this * totals_MW_pre
        # cf[i, :, :] = cf_this

    return cf, wp_dist_masked_data


    print("-------------------------")
    print("Exiting script")
    sys.exit()

    # Extract the wind speed data from the dataset
    # wind_speed = ds[bc_si100_name].values

    # Extract the values of the wind speed
    wind_speed_vals = ds.values

    # print the shape of the wind speed values
    print("Wind speed values shape:", wind_speed_vals.shape)

    # Create an empty array to store the power data
    cfs = np.zeros(np.shape(wind_speed_vals))

    # Extract total MW as the array values of the installed capacities regrid
    total_MW = ic_cube_regrid.data

    if obs_flag == False:
        print("Processing the model data into wind power data.")

        # assert that the first dimension of the wind speed values is 1
        # assert np.shape(wind_speed_vals)[0] == 1, "More than 1 init year."

        nyears = np.shape(wind_speed_vals)[0]

        # Extract the number of members
        nmems = np.shape(wind_speed_vals)[1]

        # create an empty dataarray to store the power data
        cfs = np.zeros(
            [
                nyears,
                nmems,
                np.shape(wind_speed_vals)[2],
                np.shape(wind_speed_vals)[3],
                np.shape(wind_speed_vals)[4],
            ]
        )

        # wind speed empty array
        wind_speed_pre = np.zeros(
            [
                nyears,
                nmems,
                np.shape(wind_speed_vals)[2],
                np.shape(wind_speed_vals)[3],
                np.shape(wind_speed_vals)[4],
            ]
        )

        # wind speed post
        wind_speed_post = np.zeros(
            [
                nyears,
                nmems,
                np.shape(wind_speed_vals)[2],
                np.shape(wind_speed_vals)[3],
                np.shape(wind_speed_vals)[4],
            ]
        )

        # loop over the init years
        for y in tqdm(range(0, nyears), desc="Creating wind power data for model"):
            # Select the wind speed data for the current init year
            wind_speed_vals_y = wind_speed_vals[y, :, :, :, :]
            for m in range(0, nmems):
                # Select the wind speed data for the current member
                wind_speed_vals_mem_this = wind_speed_vals_y[m, :, :, :]
                for i in range(0, np.shape(wind_speed_vals)[2]):
                    # Extract values for the current timestep
                    wind_speed_vals_i = wind_speed_vals_mem_this[i, :, :]

                    wind_speed_pre[y, m, i, :, :] = wind_speed_vals_i

                    # depending whether onshore or offshore, scale to height (from 10m!)
                    if ons_ofs == "ons":
                        # Avg height of onshore wind farms 2021 from UK windpower.net
                        wind_speed_vals_i = wind_speed_vals_i * (71.0 / 10.0) ** (
                            1.0 / 7.0
                        )
                    elif ons_ofs == "ofs":
                        # Avg height of offshore wind farms 2021 from UK windpower.net
                        wind_speed_vals_i = wind_speed_vals_i * (92.0 / 10.0) ** (
                            1.0 / 7.0
                        )

                    # Set any NaN values to zero
                    wind_speed_vals_i[np.isnan(wind_speed_vals_i)] = 0.0

                    wind_speed_post[y, m, i, :, :] = wind_speed_vals_i

                    # reshape into a 1D array
                    reshaped_wind_speed_vals = np.reshape(
                        wind_speed_vals_i,
                        [
                            np.shape(wind_speed_vals_i)[0]
                            * np.shape(wind_speed_vals_i)[1]
                        ],
                    )

                    # Categorise each wind speed value into a power output
                    cfs_i = np.digitize(
                        reshaped_wind_speed_vals, pc_df["Wind speed (m/s)"], right=False
                    )

                    # Make sure the bins don't go out of range
                    cfs_i[cfs_i == len(pc_df)] = len(pc_df) - 1

                    # convert pc_df["Power (W)"] to a numpy array of values
                    pc_power_vals = pc_df["Power (W)"].values

                    # Calculate the average power output for each bin
                    p_bins = 0.5 * (pc_power_vals[cfs_i] + pc_power_vals[cfs_i - 1])

                    # Reshape the power output array
                    cfs_i = np.reshape(
                        p_bins,
                        [
                            np.shape(wind_speed_vals_i)[0],
                            np.shape(wind_speed_vals_i)[1],
                        ],
                    )

                    # Set any values below the minimum capacity factor to the minimum capacity factor
                    cfs_i[cfs_i < min_cf] = 0.0

                    # raise an error if any of the cfs_i values are greater than 1.0
                    if np.any(cfs_i > 1.0):
                        raise ValueError("Capacity factor greater than 1.0.")

                    # Multiply by the installed capacity in MW
                    cfs[y, m, i, :, :] = cfs_i

        # where cfs are 0.0, set to NaN
        cfs[cfs == 0.0] = np.nan

        # Take the spatial mean
        cfs = np.nanmean(cfs, axis=(3, 4))

        # take the spatial mean of wind speed
        wind_speed_pre = np.nanmean(wind_speed_pre, axis=(3, 4))
        wind_speed_post = np.nanmean(wind_speed_post, axis=(3, 4))

    else:
        print("Processing the observations into wind power data.")

        # empty array for wind speeds
        wind_speed_pre = np.zeros(
            [
                np.shape(wind_speed_vals)[0],
                np.shape(wind_speed_vals)[1],
                np.shape(wind_speed_vals)[2],
            ]
        )
        wind_speed_post = np.zeros(
            [
                np.shape(wind_speed_vals)[0],
                np.shape(wind_speed_vals)[1],
                np.shape(wind_speed_vals)[2],
            ]
        )

        # Loop over the time axis
        for i in tqdm(
            range(0, np.shape(wind_speed_vals)[0]), desc="Creating wind power data"
        ):
            # Extract the wind speed data for the current timestep
            wind_speed_vals_i = wind_speed_vals[i, :, :]

            wind_speed_pre[i, :, :] = wind_speed_vals_i

            # depending whether onshore or offshore, scale to height
            if ons_ofs == "ons":
                # Avg height of onshore wind farms 2021 from UK windpower.net
                wind_speed_vals_i = wind_speed_vals_i * (71.0 / 100.0) ** (1.0 / 7.0)
            elif ons_ofs == "ofs":
                # Avg height of offshore wind farms 2021 from UK windpower.net
                wind_speed_vals_i = wind_speed_vals_i * (92.0 / 100.0) ** (1.0 / 7.0)

            # Set any NaN values to zero
            wind_speed_vals_i[np.isnan(wind_speed_vals_i)] = 0.0

            wind_speed_post[i, :, :] = wind_speed_vals_i

            # reshape into a 1D array
            reshaped_wind_speed_vals = np.reshape(
                wind_speed_vals_i,
                [np.shape(wind_speed_vals_i)[0] * np.shape(wind_speed_vals_i)[1]],
            )

            # Categorise each wind speed value into a power output
            cfs_i = np.digitize(
                reshaped_wind_speed_vals, pc_df["Wind speed (m/s)"], right=False
            )

            # Make sure the bins don't go out of range
            cfs_i[cfs_i == len(pc_df)] = len(pc_df) - 1

            # convert pc_df["Power (W)"] to a numpy array of values
            pc_power_vals = pc_df["Power (W)"].values

            # Calculate the average power output for each bin
            p_bins = 0.5 * (pc_power_vals[cfs_i] + pc_power_vals[cfs_i - 1])

            # Reshape the power output array
            cfs_i = np.reshape(
                p_bins, [np.shape(wind_speed_vals_i)[0], np.shape(wind_speed_vals_i)[1]]
            )

            # Set any values below the minimum capacity factor to the minimum capacity factor
            cfs_i[cfs_i < min_cf] = 0.0

            # raise an error if any of the cfs_i values are greater than 1.0
            if np.any(cfs_i > 1.0):
                raise ValueError("Capacity factor greater than 1.0.")

            # # Multiply by the installed capacity in MW
            # cfs[i, :, :] = cfs_i * total_MW
            cfs[i, :, :] = cfs_i

        # Where cfs are 0.0, set to NaN
        cfs[cfs == 0.0] = np.nan

        # then multiply by the total MW?

        # Take the spatial mean
        cfs = np.nanmean(cfs, axis=(1, 2))

        # take the spatial mean of wind speed
        wind_speed_pre = np.nanmean(wind_speed_pre, axis=(1, 2))
        wind_speed_post = np.nanmean(wind_speed_post, axis=(1, 2))

    return cfs, wind_speed_pre, wind_speed_post


# TODO: Finish this + onshore offshore flag
# define a function to form the dataframe for the wind power data
def form_wind_power_dataframe(
    cfs: np.ndarray,
    ds: xr.Dataset,
    country_name: str,
    obs_flag: bool = True,
    model_fpath: str = None,
    init_years: list[int] = None,
    leads: list[int] = None,
    ons_ofs: str = "ons",
) -> pd.DataFrame:
    """
    Form the dataframe for the wind power data.

    Parameters
    ----------

    cfs : np.ndarray
        The array of wind power capacity factors.

    ds : xr.Dataset
        The dataset containing the wind speed data.

    country_name : str
        The name of the country.

    obs_flag : bool
        Whether or not this function is processing the observations.

    model_fpath : str
        The file path for the model

    init_years : list[int]
        The initialisation years.
        E.g. 1960, 2018

    leads : list[int]
        The leads of the data.

    ons_ofs : str
        The type of wind farm to be considered (either onshore or offshore).

    Returns
    -------

    cfs_df : pd.DataFrame
        The dataframe containing the wind power data.

    """

    if obs_flag == True:
        # Extract the time values
        time = ds["time"].values

        # Format the time values as datetime objects
        time = pd.to_datetime(time)

        # Create a dataframe with the time values and an index
        cfs_df = pd.DataFrame(cfs, index=time)

        # Set the column name
        cfs_df.columns = [f"{country_name}_wind_power"]
    else:

        # shape of (59, 10, 30)
        # 59 init years, 10 members, 30 leads
        # Set up an empty dataframe to store the wind power data
        cfs_df = pd.DataFrame()

        # set up the nmems
        nmems = np.shape(cfs)[1]

        # Loop over the init years
        # Enumerate over the init_years
        for i, year in tqdm(
            enumerate(np.arange(init_years[0], init_years[1] + 1)),
            desc="Looping over init years",
        ):
            # select the first year
            cfs_year = cfs[i, :, :]
            for j, lead in enumerate(leads):
                # select the first lead
                cfs_lead = cfs_year[:, j]
                for member in range(1, nmems + 1):
                    cfs_df_member = pd.DataFrame(
                        {
                            "init": [year],  # year value
                            "lead": [lead],
                            "member": [member],
                            f"{country_name}_wind_power_cfs_{ons_ofs}": cfs_lead[
                                member - 1
                            ],
                        }
                    )

                    # Append the data to the dataframe
                    cfs_df = pd.concat([cfs_df, cfs_df_member], ignore_index=True)

    return cfs_df


# Write a function to save the wind power data to a csv file
def save_wind_power_data(
    cfs_df: pd.DataFrame,
    output_dir: str,
    country_name: str,
    first_year: int,
    first_month: int,
    last_year: int,
    last_month: int,
    ons_ofs: str = "ons",
) -> None:
    """
    Save the wind power data to a csv file.

    Parameters
    ----------

    cfs_df : pd.DataFrame
        The dataframe containing the wind power data.

    output_dir : str
        The directory to save the file in.

    country_name : str
        The name of the country.

    first_year : int
        The first year of the data.

    first_month : int
        The first month of the data.

    last_year : int
        The last year of the data.

    last_month : int
        The last month of the data.

    Returns

    None

    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up the path
    output_path = os.path.join(
        output_dir,
        f"{country_name}_wind_power_data_{ons_ofs}_{first_year}_{first_month}-{last_year}_{last_month}.csv",
    )

    if os.path.exists(output_path):
        print(f"File {output_path} already exists.")
        sys.exit()

    # Save the data
    cfs_df.to_csv(
        os.path.join(
            output_dir,
            f"{country_name}_wind_power_data_{ons_ofs}_{first_year}_{first_month}-{last_year}_{last_month}.csv",
        )
    )

    print(f"Wind power data saved to {output_dir}.")

    return None


# Submit this as batch job - array 1950..2020 for reanalysis data
# define the main function
def main():
    # Set up the argument parser

    # set up the start time
    start_time = time.time()

    # Set up the parameters
    # Just load in a single month of data in this test case
    first_year = 2014
    first_month = 1
    last_year = 2014  # do countries have wind power in 2014
    last_month = 1
    ons_ofs = "ons"

    # load the wind data
    ds = load_obs_data(
        last_year=last_year,
        last_month=last_month,
        first_year=first_year,
        first_month=first_month,
    )

    # Set up an empty dataframe to store the wind power data
    cfs_df = pd.DataFrame()

    # Loop over the countries
    for country in tqdm(dicts.country_list_nuts0[-2:], desc="Looping over countries"):
        print(f"Country: {country}")

        # if country is in ["Macedonia"] skip
        if country in ["Macedonia"]:
            print(f"Skipping {country}")
            continue

        # Apply the mask
        ds_country = apply_country_mask(
            ds=ds,
            country=country,
            pop_weights=0,
        )

        # if country contains a space
        if " " in country:
            country_name = country.replace(" ", "_")
        else:
            country_name = country

        # Create the wind power data
        cfs = create_wind_power_data(
            ds=ds_country,
            country=country_name,
            ons_ofs=ons_ofs,
        )

        # # Form the dataframe
        cfs_df_country = form_wind_power_dataframe(
            cfs=cfs,
            ds=ds,
            country_name=country_name,
        )

        # # print the head of the cfs_df
        # print(f"Head of the dataframe: {cfs_df.head()}")

        # Append the data to the dataframe
        cfs_df = pd.concat([cfs_df, cfs_df_country], axis=1)

    # Print the head of the full capacity factors
    print(f"Head of the full dataframe: {cfs_df.head()}")

    # Set up a directory to save in
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_ons_wind_daily_{first_year}_{first_month}_{last_year}_{last_month}_all_countries.csv"

    # set up the path
    path = os.path.join(output_dir, fname)

    # # if the path doesn't exist, create it
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # if the path doesn;t already exist, save the data
    # if not os.path.exists(path):
    #     cfs_df.to_csv(path)

    # # Save the wind power data frame
    # save_wind_power_data(
    #     cfs_df=cfs_df,
    #     output_dir="/storage/silver/clearheads/Ben/csv_files/wind_power/",
    #     country_name=country_name,
    #     first_year=1950,
    #     first_month=1,
    #     last_year=last_year,
    #     last_month=last_month,
    #     ons_ofs=ons_ofs,
    # )

    # # extract the first time step
    # cfs_i = cfs[0, :, :]

    # # Convert the array to a pandas dataframe
    # cfs_df = pd.DataFrame(cfs_i)

    # # Save the dataframe to a csv file
    # cfs_df.to_csv(f"/home/users/pn832950/100m_wind/csv_files/UK_wind_power_data_{last_year}_{last_month}.csv")

    # # print the data
    # print("-------------------")
    # print(ds)
    # print("-------------------")

    # Set up the end time
    end_time = time.time()

    # Ptint the time taken
    print(f"Time taken: {end_time - start_time}")

    # # print that we are exiting the function
    print("Exiting the function.")
    sys.exit()

    # # # Set the output directory
    # output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # # # Set the filename
    # fname = f"ERA5_100m_10m_wind_speed_daily_1950_01-{last_year}_{last_month}.nc"

    # # # Save the data
    # save_wind_data(
    #     ds=ds,
    #     output_dir=output_dir,
    #     fname=fname,
    # )


if __name__ == "__main__":
    main()
