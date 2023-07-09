import os
import sys
import shutil
from tqdm import tqdm
from multiprocessing import Pool
import datetime
from itertools import repeat
import pandas 
import cdsapi
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY
from webknossos.dataset.properties import (
    DatasetViewConfiguration,
    LayerViewConfiguration,
)
from webknossos import webknossos_context
from pathlib import Path
import numpy as np
import xarray as xr
from calendar import monthrange

webknossos_token = "hAswxSKPyjrxSKyrYIqGFw" # TODO: don't save this in code

# Retrieves a sample for the specified date
def download_sample(date : datetime.datetime, single_level_variables : list[str]) -> None:
    single_level_path = f"data/temp/single_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': single_level_variables,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'time': date.strftime("%H:%M"),
        },
        single_level_path
    )

# Downloads samples from the ERA5 dataset for a timestamp
# Retrieves data on specific pressure levels, then retrieves data on single levels, then combines them
# Returns the combined data
def retrieve_samples_date_list(dates : list[datetime.datetime], single_level_variables : list[str]) -> None:
    samples = []

    Path("data/temp").mkdir(parents=True, exist_ok=True)

    with Pool(100) as pool: # too much parallelizm gives our jobs low priority by the API
        pool.starmap(download_sample, zip(dates, repeat(single_level_variables)))
    
    for date in dates:
        single_level_path = f"data/temp/single_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        single_level_data = xr.open_dataset(single_level_path)
        samples += [single_level_data]

    return xr.concat(samples, dim="time")

# Downloads samples from the ERA5 dataset for all possible combinations of years, months, days, and times (time as "%H:%M")
def retrieve_samples_multiplex(years : list[int], months : list[int], days : list[int], times : list[str],
                    single_level_variables : list[str]) -> None:

    Path("data/temp").mkdir(parents=True, exist_ok=True)
    single_level_path = f"data/temp/single_level_tmp.nc"

    c = cdsapi.Client(quiet=True)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': single_level_variables,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
        },
        single_level_path
    )

    data = xr.open_dataset(single_level_path)

    os.remove(single_level_path)

    return  data

def create_webknossos_dataset(samples : xr.Dataset, output_variables : list[str], output_path : str) -> None:

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds = wk.Dataset(name=f"AR_TC_{len(samples.time)}_samples_02", # TODO: define the name somewhere else
                    dataset_path=output_path, voxel_size=(26e12, 26e12, 26e12)) 
    ds.default_view_configuration = DatasetViewConfiguration(zoom=1, rotation=(0, 1, 1))

    for variable_name in output_variables:
        # switch on variable name
        ch = ds.add_layer(
            variable_name,
            COLOR_CATEGORY,
            np.float32, # TODO: could use np.uint8 (with scaling) to save space
            # dtype_per_layer=samples.get(variable_name).dtype,
        )
        match variable_name:
            case 'total column water vapour (TCWV)': # total column water vapour
                ch.add_mag(1, compress=True).write(samples.get('tcwv').values)
                ch.default_view_configuration = LayerViewConfiguration(color=(17, 212, 17), intensity_range=(0, 16000))
            case 'mean pressure at sea level (MSL)': # mean sea level pressure
                ch.add_mag(1, compress=True).write(samples.get('msl').values)
                ch.default_view_configuration = LayerViewConfiguration(color=(248, 228, 92), intensity_range=(1e5, 1.1e5), min=9.5e4, is_inverted=False, is_disabled=True)
            case 'integrated vapour transport (IVT)':
                ivt = np.sqrt(samples.get('p72.162')**2 + samples.get('p71.162')**2)
                ch.add_mag(1, compress=True).write(ivt)
                ch.default_view_configuration = LayerViewConfiguration(color=(153, 193, 241), intensity_range=(0, 1000), is_disabled=True)
            case _:
                raise NotImplementedError(f"Variable type {variable_name} specified but not implemented")

    with webknossos_context(token=webknossos_token):
        ds.upload()


# creates a chunk of ERA5 data in webKnossos format 
# the chunk includes all samples between start_date and end_date (inclusively)
# the samples are spaced by delta_between_samples
# samples are only possible at full hours
def create_chunk(dates : list[pandas.Period], chunk_name : str) -> None:
    print(f"Creating chunk {chunk_name}")

    single_level_variables = [
        'mean_sea_level_pressure', # TODO: don't define this here
        'total_column_water_vapour',
        'vertical_integral_of_northward_water_vapour_flux',
        'vertical_integral_of_eastward_water_vapour_flux',
    ]
    output_variables = [
        'mean pressure at sea level (MSL)',
        'total column water vapour (TCWV)',
        'integrated vapour transport (IVT)'
    ]
    samples = retrieve_samples_date_list(dates, single_level_variables)
    create_webknossos_dataset(
        samples,
        output_variables,
        output_path=f"data/chunks/{chunk_name}.wkw"
    )
    samples.to_netcdf(f"data/chunks/{chunk_name}.nc")

def create_chunk_for_time_interval(start_date : datetime.datetime, end_date : datetime, hours_between_samples : int) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if hours_between_samples <= 0:
        raise ValueError("hours_between_samples must be greater than 0")

    dates = pandas.date_range(start_date, end_date, freq=datetime.timedelta(hours=hours_between_samples))
    chunk_name=f"chunk_interval_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
        f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_delta_{hours_between_samples}h"

    create_chunk(dates, chunk_name)

def create_chunk_with_random_samples(start_date : datetime.datetime, end_date : datetime, number_of_samples : int,
                                     excluded_dates_path : str = None) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be greater than 0")

    ## TODO: better scheme do design names
    chunk_name=f"chunk_random_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
        f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_samples_{number_of_samples}_02"

    if excluded_dates_path is not None:
        # as data range
        excluded_dates = pandas.Series(np.load(excluded_dates_path))
    else:
        excluded_dates = []

    all_dates = pandas.date_range(start_date, end_date, freq="H").to_series()

    print(f"number of all dates: {len(all_dates)}")
    remaining_dates = all_dates[~all_dates.isin(excluded_dates)]
    print(f"number of remaining dates: {len(remaining_dates)}")

    sampled_dates = remaining_dates.sample(number_of_samples).dt.to_period("H").tolist()

    # save np to disk
    np.save('./data/timestamps/'+chunk_name+'_dates.npy', sampled_dates)

    # save dates to txt file
    with open('./data/timestamps/'+chunk_name+'_dates.txt', 'w') as f:
        for item in sampled_dates:
            f.write("%s\n" % item.strftime('%Y-%m-%dT%H:%M:%S.000000000'))

    create_chunk(sampled_dates, chunk_name)

# computes the the daily mean values of the z500 variable for a day of the year across an interval of years
# For example, if from_year=1980, to_year=2020, month = 2 and day = 3, then the mean values for the 3rd of February across all years is calculated
def compute_z500_mean_values_for_day(from_year : int, to_year : int, month : int, day : int) -> xr.Dataset:
    years = list(range(from_year, to_year + 1))
    times = [f"{hour:02d}:00" for hour in range(0, 24)]

    samples = retrieve_samples_multiplex(years, [month], [day], times, ['geopotential'])

    average = samples.mean(dim='time')

    return average

# computes the the daily mean values of the z500 variable for each day of the year across an interval of years
# For example, if from_year=1980, to_year=2020, then the mean values for each day of the year across all years in the interval is calculated
def compute_z500_mean_values(from_year : int, to_year : int) -> xr.Dataset:
    """
    average_per_day = xr.Dataset(
        data_vars = {
            'z500': (['day_of_year', 'latitude', 'longitude'], np.zeros((365, 721, 1440))),
        },
        coords = {
            'day_of_year': range(1, 366),
            'latitude': range(-90, 90.5, 0.5),
            'longitude': range(0, 1440)
        }
    )
    """

    # iterate over all days of the year with (month, day)
    average_per_day = []
    pbar = tqdm(total=365, file=sys.stdout)
    leap_year = 2020
    for month in range(1, 13): # Month is always 1..12
        for day in range(1, monthrange(leap_year, month)[1] + 1):
            day_of_year = datetime.datetime(year=leap_year, month=month, day=day).timetuple().tm_yday
            average_for_this_day = compute_z500_mean_values_for_day(from_year, to_year, month, day)
            average_for_this_day = average_for_this_day.assign_coords({'day_of_year': day_of_year})
            average_per_day += [average_for_this_day]
            pbar.update(1)
            print()
            # print(end=' ', flush=True)
            # pbar.refresh()
            # sys.stdout.flush()
    pbar.close()
    average_per_day = xr.concat(average_per_day, dim='day_of_year')

    return average_per_day

# create_chunk_for_time_interval(
#     start_date = datetime.datetime(year=2004, month=1, day=1, hour=0),
#     end_date = datetime.datetime(year=2004, month=1, day=1, hour=0), # inclusively
#     hours_between_samples = 12,
#     exclude_timestamps = '/data/timestamps/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_dates.npy'
# )

create_chunk_with_random_samples(datetime.datetime(year=1980, month=1, day=1, hour=0), datetime.datetime(year=2023, month=1, day=1, hour=0), 10,
                                 excluded_dates_path = './data/timestamps/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_dates.npy')
# create_chunk_for_time_interval(start_date = datetime.datetime(year=1980, month=1, day=1, hour=0), end_date = datetime.datetime(year=1980, month=2, day=1, hour=0), hours_between_samples = 24)

# compute_z500_mean_values_for_day(from_year=2015, to_year=2021, month=2, day=29).to_netcdf("data/z500_mean_values.nc")

# compute_z500_mean_values(from_year=2020, to_year=2020).to_netcdf("data/z500_mean_values.nc")
