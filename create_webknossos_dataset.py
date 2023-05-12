import os
import shutil
from tqdm import tqdm
import datetime
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

webknossos_token = "hAswxSKPyjrxSKyrYIqGFw" # TODO: don't save this in code

# Downloads samples from the ERA5 dataset for a timestamp
# Retrieves data on specific pressure levels, then retrieves data on single levels, then combines them
# Returns the combined data
def download_samples(dates : list[datetime.datetime], pressure_level_variables : list[str], pressure_level : int, single_level_variables : list[str]) -> None:
    samples = []

    c = cdsapi.Client()
    Path("data/temp").mkdir(parents=True, exist_ok=True)
    
    for i, date in enumerate(dates): # TODO: we might want to parallelize this
        # pressure_level_path = f"data/temp/pressure_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        single_level_path = f"data/temp/single_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        print(f"sample {i+1}/{len(dates)} |  date: {date} hour: {date.strftime('%H:%M')}")
        """
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': pressure_level_variables, 
                'pressure_level': pressure_level,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'time': date.strftime("%H:%M"),
            },
            pressure_level_path,
        )
        """
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
        print()

        # pressure_level_data = xr.open_dataset(pressure_level_path)
        single_level_data = xr.open_dataset(single_level_path)
        # os.remove(pressure_level_path) # remove temp files
        # os.remove(single_level_path)

        # samples += [xr.merge([pressure_level_data, single_level_data])]
        samples += [single_level_data]

    return xr.concat(samples, dim="time")

def create_webknossos_dataset(samples : xr.Dataset, output_variables : list[str], output_path : str) -> None:

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds = wk.Dataset(name="AR_TC_random_4", # TODO: define the name somewhere else
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

    pressure_level_variables = [
        # 'u_component_of_wind', # TODO: don't define this here
        # 'v_component_of_wind',
    ]
    pressure_level = 850 # TODO: don't define this here
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
    samples = download_samples(dates, pressure_level_variables, pressure_level, single_level_variables)
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

def create_chunk_with_random_samples(start_date : datetime.datetime, end_date : datetime, number_of_samples : int) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be greater than 0")

    dates = pandas.date_range(start_date, end_date, freq="H").to_series().sample(number_of_samples).dt.to_period("H").tolist()
    chunk_name=f"chunk_random_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
        f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_samples_{number_of_samples}"

    create_chunk(dates, chunk_name)

# TODO: remove this test function
"""
create_chunk_for_time_interval(
    start_date = datetime.datetime(year=2004, month=1, day=1, hour=0),
    end_date = datetime.datetime(year=2004, month=1, day=1, hour=0), # inclusively
    hours_between_samples = 12,
)
"""

# TODO: remove this test function
create_chunk_with_random_samples(datetime.datetime(year=1980, month=1, day=1, hour=0), datetime.datetime(year=2023, month=1, day=1, hour=0), 3)

