from os import path
from tqdm import tqdm
from os import listdir, path
import xarray as xr
import pandas as pd
import glob
import os

os.makedirs("dataset", exist_ok=True)
os.makedirs(os.join("dataset", "blocking"), exist_ok=True)

label_fnames = glob.glob("tmp/blocking/*.nc")
labels = [xr.open_dataset(fname, chunks={'ts': 10}) for fname in label_fnames]
print(f"Found {len(label_fnames)} individual files.")

labels = [l.squeeze("annotator", drop=False) for l in labels]

# Concatenate and sort by timestep
labels = xr.concat(labels, dim='ts').sortby("ts")
annotator_list = labels["annotator"].values.tolist()

# "Fold" annotator dimension (i.e. create shape [n, 10, 2, 721, 1440])
instances = ('ts', np.tile([0, 1], len(labels.ts) // 2))
labels['annotator'] = instances
labels = labels.set_index(time=['ts', 'annotator'])
labels = labels.unstack('time')

labels.expand_dims(["class"])
labels = labels.assign_coords({
    "class": ["Blocking Event (BE)"],
})

labels["label"] = labels["label"].astype(bool)
labels = labels.rename({"ts": "time"})
labels = labels.transpose("time", "class", "annotator", "latitude", "longitude")

for i in tqdm(range(0, len(labels.time), 10)):

    l = labels.isel(time=slice(i, i+10))
    l = l.compute()
    
    annotator_idx_to_id = [annotator_list[i], annotator_list[i+1]]
    l = l.assign_attrs(annotator_idx_to_id=annotator_idx_to_id)

    comp = dict(zlib=True, complevel=5)
    encoding = {"label": comp}
    fname = f"{str(ds.time.values[0])}.nc"
    l.to_netcdf(path.join("dataset", "blocking", fname), encoding=encoding)