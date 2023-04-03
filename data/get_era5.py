import cdsapi

"""
Retrieve data on pressure levels (just U850 and V850 for now), then retreive data on single levels, then combine them.
"""

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': '850',
        'year': '2004',
        'month': '01',
        'day': '01',
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    'pressure_levels.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'mean_sea_level_pressure', 'total_column_water_vapour',
        ],
        'year': '2004',
        'month': '01',
        'day': '01',
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    'single_levels.nc')