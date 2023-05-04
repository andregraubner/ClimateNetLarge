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
from pathlib import Path
import numpy as np
import xarray as xr

# Downloads samples from the ERA5 dataset for a timestamp
# Retrieves data on specific pressure levels, then retrieves data on single levels, then combines them
# Returns the combined data
def download_samples(dates : list[datetime.datetime], pressure_level_variables : list[str], pressure_level : int, single_level_variables : list[str]) -> None:
    samples = []

    c = cdsapi.Client()
    Path("data/temp").mkdir(parents=True, exist_ok=True)
    
    for date in dates: # TODO: we might want to parallelize this
        pressure_level_path = f"data/temp/pressure_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        single_level_path = f"data/temp/single_level_{date.year}_{date.month}_{date.day}_{date.strftime('%H:%M')}:00.nc"
        print(f"date: {date}")
        print(f"hour: {date.strftime('%H:%M')}")
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

        pressure_level_data = xr.open_dataset(pressure_level_path)
        single_level_data = xr.open_dataset(single_level_path)
        os.remove(pressure_level_path) # remove temp files
        os.remove(single_level_path)

        samples += [xr.merge([pressure_level_data, single_level_data])]

    return xr.concat(samples, dim="time")

def create_webknossos_dataset(samples, output_path) -> None:

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds = wk.Dataset(dataset_path=output_path, voxel_size=(1, 1, 1))  # TODO: what voxel size?
    ds.default_view_configuration = DatasetViewConfiguration(zoom=0.35) # TODO: zoom
    

    for variable_name in samples.keys():
        # switch on variable name
        ch = ds.add_layer(
            variable_name,
            COLOR_CATEGORY,
            dtype_per_layer=samples.get('u').dtype,
        )
        match variable_name:
            case 'u' | 'v' | 'tcwv' | 'msl': # TODO: If we don't distinguish between variables, we can drop the match statement
                ch.add_mag(1, compress=True).write(samples.get('u'))
                ch.default_view_configuration = LayerViewConfiguration(
                    color=(17, 212, 17), intensity_range=(0, 16000)
                )
            case _:
                raise NotImplementedError(f"Variable type {variable_name} specified but not implemented")


# creates a chunk of ERA5 data in webKnossos format 
# the chunk includes all samples between start_date and end_date (exclusively)
# the samples are spaced by delta_between_samples
# samples are only possible at full hours
def create_chunk(start_date : datetime.datetime, end_date : datetime, hours_between_samples : int) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    if start_date.minute != 0 or start_date.second != 0 or start_date.microsecond != 0:
        raise ValueError("start_date must be at the beginning of an hour")
    if hours_between_samples <= 0:
        raise ValueError("hours_between_samples must be greater than 0")

    dates = pandas.date_range(start_date, end_date, freq=datetime.timedelta(hours=hours_between_samples))
    pressure_level_variables = [
        'u_component_of_wind', # TODO: don't define this here
        'v_component_of_wind',
    ]
    pressure_level = 850 # TODO: don't define this here
    single_level_variables = [
        'mean_sea_level_pressure', # TODO: don't define this here
        'total_column_water_vapour', # TODO: is this the right variable?
    ]
    samples = download_samples(dates, pressure_level_variables, pressure_level, single_level_variables)
    create_webknossos_dataset(
        samples,
        output_path=f"data/chunks/chunk_{start_date.year}_{start_date.month}_{start_date.day}_{start_date.strftime('%H:%M')}-" + \
            f"{end_date.year}_{end_date.month}_{end_date.day}_{end_date.strftime('%H:%M')}_delta_{hours_between_samples}h.wkw"
    )

# TODO: remove this test function
create_chunk(
    start_date = datetime.datetime(year=2004, month=1, day=1, hour=0),
    end_date = datetime.datetime(year=2004, month=1, day=3, hour=0),
    hours_between_samples = 12,
)
