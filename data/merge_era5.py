import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY
from webknossos.dataset.properties import (
    DatasetViewConfiguration,
    LayerViewConfiguration,
)
from pathlib import Path
import tifffile
import numpy as np
import xarray as xr


def main():

    pressure_levels = xr.open_dataset("pressure_levels.nc")
    single_levels = xr.open_dataset("single_levels.nc")

    merged = xr.merge([pressure_levels, single_levels])

    merged = merged.to_array().to_numpy()
    merged = np.transpose(merged, [0, 2, 3, 1])

    name = f"test_climatenet"

    # voxel_size is defined in nm
    ds = wk.Dataset(name, voxel_size=(260, 260, 290))

    ds.default_view_configuration = DatasetViewConfiguration(zoom=0.35)

    ch_1 = ds.add_layer(
        "ch1",
        COLOR_CATEGORY,
        dtype_per_layer=merged.dtype,
    )
    ch_1.add_mag(1, compress=True).write(merged[0, :])
    ch_1.default_view_configuration = LayerViewConfiguration(
        color=(17, 212, 17), intensity_range=(0, 16000)
    )

    ch_2 = ds.add_layer(
        "ch2",
        COLOR_CATEGORY,
        dtype_per_layer=merged.dtype,
    )
    ch_2.add_mag(1, compress=True).write(merged[0, :])
    ch_2.default_view_configuration = LayerViewConfiguration(
        color=(17, 212, 17), intensity_range=(0, 16000)
    )

    ch_3 = ds.add_layer(
        "ch3",
        COLOR_CATEGORY,
        dtype_per_layer=merged.dtype,
    )
    ch_3.add_mag(1, compress=True).write(merged[0, :])
    ch_3.default_view_configuration = LayerViewConfiguration(
        color=(17, 212, 17), intensity_range=(0, 16000)
    )

    ch_4 = ds.add_layer(
        "ch4",
        COLOR_CATEGORY,
        dtype_per_layer=merged.dtype,
    )
    ch_4.add_mag(1, compress=True).write(merged[0, :])
    ch_4.default_view_configuration = LayerViewConfiguration(
        color=(17, 212, 17), intensity_range=(0, 16000)
    )

    ds.downsample()
    ds.compress()
