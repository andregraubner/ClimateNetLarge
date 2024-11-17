from dataclasses import dataclass, field
from typing import Tuple
from dateutil import parser
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from tqdm import tqdm

import webknossos as wk

AUTH_TOKEN = ""
PROJECT_NAME = "2023-05-TC-AR"
DATASET_PATH = "./tmp/"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.join(DATASET_PATH, "AR"), exist_ok=True)
os.makedirs(os.join(DATASET_PATH, "TC"), exist_ok=True)

@dataclass
class AnnotationInfo:
    annotation_id: str
    annotator_name: str
    annotation_start_timestamp: datetime
    annotation_duration_in_seconds: timedelta
    task_type: str
    task_id: str
    dataset_name: str
    range: Tuple[int, int]
    segmentation_data: np.ndarray = field(repr=False)

def main():

    with open("dates_0.txt") as file:
        timestamps_1 = [parser.parse(line.rstrip()) for line in file]

    with open("dates_1.txt") as file:
        timestamps_2 = [parser.parse(line.rstrip()) for line in file]

    reference = xr.open_dataset("reference.nc")

    project = wk.Project.get_by_name(PROJECT_NAME)
    tasks = list(project.get_tasks(fetch_all=True))
    print(f"Found {len(tasks)} tasks")
    for task in tqdm(tasks):
        for ai in task.get_annotation_infos():

            # Get data
            annotation = ai.download_annotation()
            seg_data = (
                annotation.get_remote_annotation_dataset()
                .get_segmentation_layers()[0]
                .get_finest_mag()
                .read(absolute_bounding_box=annotation.task_bounding_box)[0]
            )

            # Calculate timestamps using dataset name and range
            range=(
                annotation.task_bounding_box.topleft.x,
                annotation.task_bounding_box.bottomright.x,
            )

            dataset_name = annotation.dataset_name

            if dataset_name == "AR_TC_5000_samples_02":
                ts = timestamps_2[range[0]:range[1]]
            else:
                ts = timestamps_1[range[0]:range[1]]

            # Create xarray dataset
            data = xr.DataArray(
                data=seg_data[None], 
                dims=["annotator", "ts", "latitude", "longitude"],
                coords={
                    "annotator": [annotation.owner_name],
                    "ts": ts,
                    "longitude": reference.coords["longitude"],
                    "latitude": reference.coords["latitude"]
                })
            
            # Save to file
            data = xr.Dataset(
                data_vars={"label": data},
                attrs={
                    # might want to add additional metadata here
                    "dataset_name": annotation.dataset_name,
                })

            data.to_netcdf(
                path=f"{DATASET_PATH}/{task.task_type.name[-2:]}/{annotation.annotation_id}.nc", 
                encoding={"label": {"zlib": True, "complevel": 9}}
            )

if __name__ == "__main__":
    with wk.webknossos_context(token=AUTH_TOKEN):
        main()
