import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from netCDF4 import Dataset
import shapely.geometry
import cartopy.io.shapereader as shpreader
import csv
from mpl_toolkits.basemap import interp as Interp
import functions_for_creating_NUTS_data as energy_model_functions

#
#
# Flags to create data - make sure to set them all (even redundant ones) accurately before
# running the script.
#
#


field_str = 'speed100m' # options: 't2m','speed100m','speed10m','ssrd' , 'sp' , 'wp'
NUTS_lev = 0 # 0,1,2, 9 (9 is the offshore EEZ zones+ shipping), 8 is offshore EEZs
CC_flag = 1 # 0=just ERA5, 1=does delta corrections
MODEL_NAME = 'ERA5' # ERA5, EC-EARTH3P,EC-EARTH3P-HR,MOHC_HH_3hr, MOHC_MM-1hr,MOHC_MM-1hr2,

wp_weights = 0 # 0 for no location weighting, 1 for including wind turbine locations
WP_sim = 0 # 0=current, 1 =future (both taken from thewindpower.net)
ons_ofs = 'ons' # options = 'ons','ofs'
NUTS9_country_choice = 'United Kingdom' #' options = 'United Kingdom','Ireland','Norway'
print(ons_ofs)
print(WP_sim)

pop_weights = 0 # 0=no population weights, 1= 
sp_weights = 0 # 0 if equal area weighted, 1 if location weighted. must be 0 for NUTS_lev 1 and 2.

save_dir = '/storage/silver/clearheads/Data/deliv_round_2/ERA5_raw_timeseries/'

print(NUTS_lev)
print(MODEL_NAME)


##################################
#
#
# touch nothing below this point.
#
##################################

country_list = energy_model_functions.get_country_list(NUTS_lev,NUTS9_country_choice)
print(country_list)

for COUNTRY in country_list: 
    print COUNTRY

    country_mask = energy_model_functions.load_appropriate_mask(COUNTRY,NUTS_lev,pop_weights,sp_weights,wp_weights,WP_sim,ons_ofs,NUTS9_country_choice)

    aggregate_data = []
    ERA5_data = np.zeros([71,8784]) # 71:84
   
    for year in range(1950,2021): # 1950-2021
        print(COUNTRY)
        aggregate_data = []
        print(year)
        for month in range(1,13):
            print(month)
            
            country_agg = energy_model_functions.load_appropriate_data(field_str,year,month,country_mask,ons_ofs,wp_weights,pop_weights,sp_weights,NUTS_lev,CC_flag,MODEL_NAME)
            aggregate_data.append(country_agg)
            #print(len(country_agg))
            #print(country_agg)
            #plt.plot(country_agg)
            #plt.show()
   
        # once you've done a year aggregate it up
        temp_data = np.array([item for sublist in aggregate_data for item in sublist])
        print(temp_data)
        if field_str in ['ssrd']:
            temp_data[temp_data ==0.] = np.nan

        #plt.plot(temp_data)
        #plt.show()
        print(len(temp_data))
        ERA5_data[year-1950,0:len(temp_data)] = temp_data

    # 71 = number of years in ERA5, 8784 = hours in a leapyear.
    ERA5_timeseries = np.reshape(ERA5_data, 71*8784) 

    # make sure any zeros are Nans at this point or we will lose points we wanted to keep
    cleaned_timeseries = ERA5_timeseries[ERA5_timeseries != 0.]  # delete the points that have zeros, this will be a small problem for 1950 where the first 7 hours of te year are missing. This creates problems for solar PV!

    cleaned_timeseries[np.isnan(cleaned_timeseries)] = 0.

    if COUNTRY == 'United Kingdom':
        COUNTRY = 'United_Kingdom'
    elif COUNTRY == 'Czech Republic':
        COUNTRY = 'Czech_Republic'
    elif COUNTRY == 'Bosnia and Herzegovina':
        COUNTRY = 'Bosnia_and_Herzegovina'

    else:
        COUNTRY = COUNTRY


    # save era5 normal data
    if CC_flag == 0:
        if pop_weights == 1:
            with open(save_dir + COUNTRY + '_' + field_str + '_pop_weighted_ERA5_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')

                f.close()
        elif sp_weights == 1:
            with open(save_dir  + COUNTRY + '_' + field_str + '_loc_weighted_ERA5_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')
                f.close()

        elif field_str == 'wp':
            if (NUTS_lev <= 8) and (wp_weights ==0):
                with open(save_dir + COUNTRY + '_' + field_str + '_' + ons_ofs + '_ERA5_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()
            elif NUTS_lev == 9:
                with open(save_dir + NUTS9_country_choice + '_' + COUNTRY + '_' + field_str + '_' + ons_ofs + '_ERA5_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()

            elif wp_weights ==1:
                with open(save_dir + COUNTRY + '_' + field_str + '_' + ons_ofs + '_' + str(WP_sim) + '_loc_weights_ERA5_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()
            else:
                print('problem with windpower save')

        else:
            with open(save_dir + COUNTRY + '_' + field_str + '_ERA5_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')
                f.close()


    # save delta corrected data
    elif CC_flag == 1:
        if pop_weights == 1:
            with open(save_dir + COUNTRY + '_' + field_str + '_pop_weighted_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')

                f.close()
        elif sp_weights == 1:
            with open(save_dir + COUNTRY + '_' + field_str + '_loc_weighted_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')
                f.close()

        elif field_str == 'wp':
            if (NUTS_lev <= 8) and (wp_weights ==0):
                with open(save_dir + COUNTRY + '_' + field_str + '_' + ons_ofs + '_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()
            elif NUTS_lev == 9:
                with open('/storage/silver/clearheads/Data/deliv_round_2/ERA5_raw_timeseries/' + NUTS9_country_choice + '_' + COUNTRY + '_' + field_str + '_' + ons_ofs + '_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()

            elif wp_weights ==1:
                with open(save_dir + COUNTRY + '_' + field_str + '_' + ons_ofs + '_' + str(WP_sim) + '_loc_weights_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                    for row in cleaned_timeseries:
                        f.write(str(row)+ '\n')
                    f.close()
            else:
                print('problem with windpower save')

        else:
            with open(save_dir + COUNTRY + '_' + field_str + '_' + MODEL_NAME + 'delta_cor_1950_2020.dat','w') as f:
                for row in cleaned_timeseries:
                    f.write(str(row)+ '\n')
                f.close()

    else:
        print('problem with climate change save')








