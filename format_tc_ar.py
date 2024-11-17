from os import path
from tqdm import tqdm
from os import listdir, path
import xarray as xr
import pandas as pd
import torch
import numpy as np

import glob
from dask import delayed
import os

os.makedirs("dataset", exist_ok=True)
os.makedirs(os.join("dataset", "AR"), exist_ok=True)
os.makedirs(os.join("dataset", "TC"), exist_ok=True)

def has_repeat(lst):
  for i in range(len(lst)-1):
    if lst[i] == lst[i+1]:
      return True
  return False

datasets_class1 = [xr.open_dataset(f, chunks={'ts': 10}) for f in glob.glob("tmp/AR/*.nc")]
datasets_class2 = [xr.open_dataset(f, chunks={'ts': 10}) for f in glob.glob("tmp/TC/*.nc")]

#datasets_class1 = [d.drop_vars('annotator') for d in datasets_class1]
datasets_class1 = [d.squeeze("annotator", drop=False) for d in datasets_class1]
datasets_class2 = [d.squeeze("annotator", drop=False) for d in datasets_class2]

# Concatenate the datasets for each class
ds_class1 = xr.concat(datasets_class1, dim='ts').sortby("ts")
ds_class2 = xr.concat(datasets_class2, dim='ts').sortby("ts")

assert has_repeat(ds_class1['ts'].values)
assert has_repeat(ds_class2['ts'].values)

instances = ('ts', np.tile([0, 1], len(ds_class1.ts) // 2))
ds_class1['annotator'] = instances
ds_class1 = ds_class1.set_index(time=['ts', 'annotator'])
ds_class1 = ds_class1.unstack('time')

instances = ('ts', np.tile([0, 1], len(ds_class2.ts) // 2))
ds_class2['annotator'] = instances
ds_class2 = ds_class2.set_index(time=['ts', 'annotator'])  
ds_class2 = ds_class2.unstack('time')     

ds = xr.concat([ds_class1, ds_class2], dim="class", join="inner")
ds = ds.rename({"ts": "time"})

ds = ds.transpose("time", "class", "annotator", "latitude", "longitude")

ds["label"] = ds["label"].astype(bool)
ds = ds.assign_coords({
    "class": ["Atmospheric River (AR)", "Tropical Cyclone (TC)"],
})

for ts in tqdm(ds["time"]):

    labels = ds.sel(time=ts)["label"]
    ar = labels[0]
    tc = labels[1]    

    fname = f"{str(ds.time.values[0])}.nc"

    comp = dict(zlib=True, complevel=5)
    encoding = {"label": comp}
    
    ar.to_netcdf(path.join("dataset", "AR", fname), encoding=encoding)
    tc.to_netcdf(path.join("dataset", "TC", fname), encoding=encoding)