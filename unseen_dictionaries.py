"""
Dictionary of things for converting between different types of data.
"""

# Location of hannah reg coefss
demand_reg_coefs_path = "/home/users/benhutch/ERA5_energy_update/ERA5_Regression_coeffs_demand_model.csv"

country_codes = ['AT' 'AL' 'BY' 'BE' 'BA' 'BG' 'HR' 'CZ' 'DK' 'EE' 'FI' 'FR' 'DE' 'GR'
 'HU' 'IE' 'IT' 'XK' 'LV' 'LT' 'LU' 'MK' 'MD' 'ME' 'NL' 'NO' 'PL' 'PT'
 'RO' 'RS' 'SK' 'SI' 'ES' 'SE' 'CH' 'TR' 'UA' 'UK']

column_dictionary = {
        "Austria" : 1,
        "Belgium" : 2,
        "Bulgaria" : 3,
        "Croatia" : 4,
        "Czech_Republic" : 5,
        "Denmark" : 6,
        "Finland" : 7,
        "France" : 8,
        "Germany" : 9,
        "Greece" : 10,
        "Hungary" : 11,
        "Ireland" : 12,
        "Italy" : 13,
        "Latvia" : 14,
        "Lithuania" : 15,
        "Luxembourg" : 16,
        "Montenegro" : 17,
        "Netherlands" : 18,
        "Norway" : 19,
        "Poland" : 20,
        "Portugal" : 21,
        "Romania" : 22,
        "Slovakia" : 23,
        "Slovenia" : 24,
        "Spain" : 25,
        "Sweden" : 26,
        "Switzerland" : 27,
        "United_Kingdom" : 28,
    }

countries_nuts_id = {
    "Austria": "AT",
    "Albania": "AL",
    "Belarus": "BY",
    "Belgium": "BE",
    "Bosnia_and_Herzegovina": "BA",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Czech_Republic": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "EL",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Kosovo": "XK",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Macedonia": "MK",
    "Moldova": "MD",
    "Montenegro": "ME",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Serbia": "RS",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United_Kingdom": "UK",
}

# Europe grid to subset to
eu_grid = {
    "lon1": -40,  # degrees east
    "lon2": 30,
    "lat1": 30,  # degrees north
    "lat2": 80,
}