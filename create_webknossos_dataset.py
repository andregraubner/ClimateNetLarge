import matplotlib.pyplot as plt
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
from numpy.fft import rfft, irfft
from calendar import monthrange
import math

webknossos_token = ""

# Retrieves a sample for the specified date

def download_sample_single_level(date: datetime.datetime, single_level_variables: list[str]) -> None:
    file_path = f"data/temp/sample_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
    if os.path.exists(file_path):
        return
    c = cdsapi.Client(quiet=False)
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
        file_path
    )

# Retrieves a sample for the specified date


def download_sample_pressure_level(date: datetime.datetime, variable_and_level: tuple[str, int]) -> None:
    variable, level = variable_and_level
    file_path = f"data/temp/sample_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
    c = cdsapi.Client(quiet=False)
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'pressure_level': level,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'time': date.strftime("%H:%M"),
        },
        file_path
    )

# Downloads samples from the ERA5 dataset for a timestamp
# Retrieves data on specific pressure level or retrieves data on single levels
# Exactly one of those two options must be specified (a combination could be implemented when needed)


def retrieve_samples_date_list(dates: list[datetime.datetime], single_level_variables: list[str],
                               pressure_variable_and_level: tuple[str, int]) -> None:
    samples = []
    if (len(pressure_variable_and_level) > 0 and len(single_level_variables) > 0) or \
       (len(pressure_variable_and_level) == 0 and len(single_level_variables) == 0):
        raise ValueError(
            "Exactly one of pressure_level_variables_and_levels and single_level_variables must be specified")

    Path("data/temp").mkdir(parents=True, exist_ok=True)

    with Pool(100) as pool:  # too much parallelism gives our jobs low priority by the API
        if len(pressure_variable_and_level) > 0:
            pool.starmap(download_sample_pressure_level, zip(
                dates, repeat(pressure_variable_and_level)))
        else:
            pool.starmap(download_sample_single_level, zip(
                dates, repeat(single_level_variables)))

    for date in tqdm(dates):
        file_path = f"data/temp/sample_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        sample = xr.open_dataset(file_path, chunks='auto')
        # os.remove(file_path)
        samples += [sample]

    return xr.concat(samples, dim="time")

# Downloads samples from the ERA5 dataset for all possible combinations of years, months, days, and times (time as "%H:%M")


def retrieve_samples_multiplex(years: list[int], months: list[int], days: list[int], times: list[str],
                               pressure_level_variables: list[str], pressure_levels: list[int]) -> None:

    Path("data/temp").mkdir(parents=True, exist_ok=True)
    temp_file_path = f"data/temp/samples_tmp.nc"

    c = cdsapi.Client(quiet=True)
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': pressure_level_variables,
            'pressure_level': pressure_levels,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
        },
        temp_file_path
    )

    data = xr.open_dataset(temp_file_path)

    os.remove(temp_file_path)

    return data


def create_webknossos_dataset(samples: xr.Dataset, output_variables: list[str], output_path: str) -> None:
    print(samples)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds = wk.Dataset(name=f"blocking_events_{len(samples.time)}_01",  # TODO: define the name somewhere else
                    dataset_path=output_path, voxel_size=(26e12, 26e12, 26e12))
    ds.default_view_configuration = DatasetViewConfiguration(
        zoom=1, rotation=(0, 1, 1))

    for variable_name in output_variables:
        # switch on variable name
        ch = ds.add_layer(
            variable_name,
            COLOR_CATEGORY,
            # TODO: could use np.uint8 (with scaling) to save space
            np.float32,
            # dtype_per_layer=samples.get(variable_name).dtype,
        )
        match variable_name:
            # total column water vapour
            case 'total column water vapour (TCWV)':
                ch.add_mag(1, compress=True).write(samples.get('tcwv').values)
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(17, 212, 17), intensity_range=(0, 16000))
            case 'mean pressure at sea level (MSL)':  # mean sea level pressure
                ch.add_mag(1, compress=True).write(samples.get('msl').values)
                ch.default_view_configuration = LayerViewConfiguration(color=(
                    248, 228, 92), intensity_range=(1e5, 1.1e5), min=9.5e4, is_inverted=False, is_disabled=True)
            case 'integrated vapour transport (IVT)':
                ivt = np.sqrt(samples.get('p72.162')**2 +
                              samples.get('p71.162')**2)
                ch.add_mag(1, compress=True).write(ivt)
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(153, 193, 241), is_disabled=True)
            case 'z500 (height at 500 hPa)':
                ch.add_mag(1, compress=True).write(samples.get('z').values)
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(17, 212, 17), is_disabled=True)
            case 'deviation_from_z500_mean':
                ch.add_mag(1, compress=True).write(
                    samples.get(variable_name).values)
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(153, 193, 241), intensity_range=(0,750))
            case 'deviation_from_z500_denoised_mean' | 'z500_mean' | 'z500_denoised_mean' | 'geopotential':
                ch.add_mag(1, compress=True).write(
                    samples.get(variable_name).values)
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(153, 193, 241), is_disabled=True)
            case _:
                raise NotImplementedError(
                    f"Variable type {variable_name} specified but not implemented")

    with webknossos_context(token=webknossos_token):
        ds.upload()

# computes the anomaly of the z500 variable for each sample in the dataset


def compute_z500_anomaly(samples: xr.Dataset) -> xr.Dataset:
    print("computing z500 anamoly..")
    # TODO: path should be in some config
    z500_mean = xr.load_dataset("./data/z500_mean_values.nc")

    print(f"Low-pass filtering of z500 mean")
    z500_mean_fft = rfft(z500_mean["z"], axis=0)
    z500_mean_fft[6:] = 0
    z500_mean["z"].values = irfft(z500_mean_fft, axis=0)

    geopotential = xr.load_dataset('./data/geopotential.nc').get('z')

    print("assigning z500 mean")
    samples = samples.assign(deviation_from_z500_mean=samples.get('z').copy())
    # samples = samples.assign(z500_mean=samples.get('z').copy())
    # samples = samples.assign(geopotential=samples.get('z').copy())

    for date in tqdm(samples.get('time')):
        day_of_year = pandas.Timestamp(date.values).timetuple().tm_yday
        mean_for_this_day = z500_mean.get(
            'z').loc[dict(day_of_year=day_of_year)]
        # samples.get('z500_mean').loc[dict(time=date)] = mean_for_this_day
        deviation = samples.get('z').loc[dict(time=date)] - mean_for_this_day
        samples.get('deviation_from_z500_mean').loc[dict(
            time=date)] = deviation
        # samples.get('geopotential').loc[dict(time=date)] = geopotential

    """
    weights = np.sin(np.deg2rad(45)) / np.sin(np.deg2rad(samples.latitude))
    weights = np.nan_to_num(weights, nan=1, posinf=1, neginf=1)

    weights = np.abs(weights[None,:,None])
    weights[weights > 5] = 5 # Play around with this?

    samples["deviation_from_z500_mean"] *= weights
    """

    # for every sample:
    # fetch the denoised_mean for this day_of_the_year
    # compute deviation denoised_mean
    # add deviation as new variable to samples
    # the expected key is 'deviation_from_z500_denoised_mean'
    # Also add 'z500_denoised_mean' as new variables to samples for debugging

    return samples


# creates a chunk of ERA5 data in webKnossos format
# the chunk includes all samples between start_date and end_date (inclusively)
# the samples are spaced by delta_between_samples
# samples are only possible at full hours
def create_chunk(dates: list[pandas.Period], single_level_variables: list[str],
                 pressure_level_variables_and_level: tuple[str, int],
                 output_variables: list[str], chunk_name: str,
                 create_webknossos : bool = True, compute_z500_anomaly : bool = True,
                 save_dir : str = "/scratch/lukaska/data/chunks") -> None:
    print(f"Creating chunk {chunk_name}")

    if create_webknossos:
        # save dates to np file
        np.save('./data/timestamps/'+chunk_name+'_dates.npy', dates)

        # save dates to txt file
        with open('./data/timestamps/'+chunk_name+'_dates.txt', 'w') as f:
            for item in dates:
                f.write("%s\n" % item.strftime('%Y-%m-%dT%H:%M:%S.000000000'))

    samples = retrieve_samples_date_list(
        dates, single_level_variables, pressure_level_variables_and_level)

    if compute_z500_anomaly:
        samples = compute_z500_anomaly(samples)

    if create_webknossos:
        print("creating webknossos dataset")
        create_webknossos_dataset(
            samples,
            output_variables,
            output_path=f"data/chunks/{chunk_name}.wkw",
        )
    print(f"saving {save_dir}/{chunk_name}.nc")
    samples.to_netcdf(f"{save_dir}/{chunk_name}.nc")
    print(f"completed saving {save_dir}/{chunk_name}.nc")


def create_chunk_for_time_interval_BE(start_date: datetime.datetime, end_date: datetime, hours_between_samples: int) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if hours_between_samples <= 0:
        raise ValueError("hours_between_samples must be greater than 0")

    dates = pandas.date_range(
        start_date, end_date, freq=datetime.timedelta(hours=hours_between_samples))
    chunk_name = f"chunk_interval_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
        f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_delta_{hours_between_samples}h"

    single_level_variables = []

    pressure_variable_and_level = ('geopotential', 500)

    output_variables = [
        'z500 (height at 500 hPa)',
        # 'z500_mean',
        'deviation_from_z500_mean',
        # 'geopotential',
    ]

    create_chunk(dates, single_level_variables,
                 pressure_variable_and_level, output_variables, chunk_name)

def create_chunk_with_random_samples_AR_TC(start_date: datetime.datetime, end_date: datetime, number_of_samples: int,
                                           excluded_dates_path: str = None) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be greater than 0")

    # TODO: better scheme do design names
    chunk_name = f"chunk_random_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
        f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_samples_{number_of_samples}_03"

    if excluded_dates_path is not None:
        # as data range
        excluded_dates = pandas.Series(np.load(excluded_dates_path))
    else:
        excluded_dates = []

    all_dates = pandas.date_range(start_date, end_date, freq="H").to_series()

    print(f"number of all dates: {len(all_dates)}")
    remaining_dates = all_dates[~all_dates.isin(excluded_dates)]
    print(f"number of remaining dates: {len(remaining_dates)}")

    sampled_dates = remaining_dates.sample(
        number_of_samples).dt.to_period("H").tolist()

    single_level_variables = [
        'mean_sea_level_pressure',
        'total_column_water_vapour',
        'vertical_integral_of_northward_water_vapour_flux',
        'vertical_integral_of_eastward_water_vapour_flux',
    ]

    pressure_variable_and_level = []

    output_variables = [
        'mean pressure at sea level (MSL)',
        'total column water vapour (TCWV)',
        'integrated vapour transport (IVT)'
    ]

    create_chunk(sampled_dates, single_level_variables,
                 pressure_variable_and_level, output_variables, chunk_name)

# computes the the daily mean values of the z500 variable for a day of the year across an interval of years
# For example, if from_year=1980, to_year=2020, month = 2 and day = 3, then the mean values for the 3rd of February across all years is calculated


def compute_z500_mean_values_for_day(from_year: int, to_year: int, month: int, day: int) -> xr.Dataset:
    years = list(range(from_year, to_year + 1))
    times = [f"{hour:02d}:00" for hour in range(0, 24)]

    samples = retrieve_samples_multiplex(
        years, [month], [day], times, ['geopotential'], [500])

    average = samples.mean(dim='time')

    return average

# computes the the daily mean values of the z500 variable for each day of the year across an interval of years
# For example, if from_year=1980, to_year=2020, then the mean values for each day of the year across all years in the interval is calculated


def compute_z500_mean_values(from_year: int, to_year: int) -> xr.Dataset:
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
    for month in range(1, 13):  # Month is always 1..12
        for day in range(1, monthrange(leap_year, month)[1] + 1):
            day_of_year = datetime.datetime(
                year=leap_year, month=month, day=day).timetuple().tm_yday
            average_for_this_day = compute_z500_mean_values_for_day(
                from_year, to_year, month, day)
            average_for_this_day = average_for_this_day.assign_coords(
                {'day_of_year': day_of_year})
            average_per_day += [average_for_this_day]
            pbar.update(1)
            print()
            # print(end=' ', flush=True)
            # pbar.refresh()
            # sys.stdout.flush()
    pbar.close()
    average_per_day = xr.concat(average_per_day, dim='day_of_year')

    return average_per_day

def download_ar_tc(dates : list[pandas.Period], output_name: str) -> None:
    single_level_variables = [
        'mean_sea_level_pressure',
        'total_column_water_vapour',
        'vertical_integral_of_northward_water_vapour_flux',
        'vertical_integral_of_eastward_water_vapour_flux',
    ]

    pressure_variable_and_level = []

    output_variables = [
        'mean pressure at sea level (MSL)',
        'total column water vapour (TCWV)',
        'integrated vapour transport (IVT)'
    ]

    create_chunk(dates, single_level_variables,
                 pressure_variable_and_level, output_variables, output_name, create_webknossos=False, compute_z500_anomaly=False)

# TODO: fix first year in which data from modern satelites is available
# Download all of ERA5 for inference purposes.
# In the current version only the data for AR & TC inference is downloaded
def download_all_of_era5(hours_between_samples = 6, first_year = 2022, last_year = 2023) -> None:

    for year in range(first_year, last_year+1):
        print(f"Downloading year {year}")
        dates = pandas.date_range(datetime.datetime(year=first_year, month=1, day=1, hour=0),
                                datetime.datetime(year=last_year+1, month=1, day=1, hour=0),
                                freq=datetime.timedelta(hours=hours_between_samples),
                                inclusive="left").to_series()
        chunk_name = f"{year}_daily_{hours_between_samples}h"

        single_level_variables = [
            'mean_sea_level_pressure',
            'total_column_water_vapour',
            'vertical_integral_of_northward_water_vapour_flux',
            'vertical_integral_of_eastward_water_vapour_flux',
        ]
        pressure_variable_and_level = []
        output_variables = [
            'mean pressure at sea level (MSL)',
            'total column water vapour (TCWV)',
            'integrated vapour transport (IVT)'
        ]

        create_chunk(dates, single_level_variables,
                    pressure_variable_and_level, output_variables, chunk_name,
                    create_webknossos=False, compute_z500_anomaly=False, save_dir="/mnt/lukaska/data/era5/chunks")


#compute_z500_anomaly(0)

# create_chunk_with_random_samples_AR_TC(datetime.datetime(year=1980, month=1, day=1, hour=0), datetime.datetime(year=2023, month=1, day=1, hour=0), 10,
#                                  excluded_dates_path = './data/timestamps/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_dates.npy')

# compute_z500_mean_values(from_year=1980, to_year=2022).to_netcdf("data/z500_mean_values.nc")


# create_chunk_for_time_interval_BE(start_date=datetime.datetime(year=2000, month=1, day=1, hour=0),
#                                   end_date=datetime.datetime(year=2013, month=9, day=8, hour=0), 
#                                   hours_between_samples=24)
# create_chunk_for_time_interval_BE(start_date=datetime.datetime(year=2000, month=1, day=1, hour=0),
#                                   end_date=datetime.datetime(year=2001, month=1, day=8, hour=0), 
#                                   hours_between_samples=24)

# dates = pandas.Series(np.load(dates_path))
# download_ar_tc(dates, "downloaded_dataset_03")

download_all_of_era5()
