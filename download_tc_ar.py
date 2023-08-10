from dataclasses import dataclass
from typing import Tuple
from dateutil import parser

import numpy as np
import xarray as xr
from tqdm import tqdm

import webknossos as wk

AUTH_TOKEN = ""
PROJECT_NAME = "2023-05-TC-AR"


@dataclass
class AnnotationInfo:
    annotation_id: str
    annotator_name: str
    annotation_time_in_seconds: int
    task_type: str
    task_id: str
    dataset_name: str
    range: Tuple[int, int]
    segmentation_data: np.ndarray

def main():

    with open("timestamps/chunk_random_1980_1_1_00_00-2023_1_1_00_00_samples_5000_dates.txt") as file:
        timestamps_1 = [parser.parse(line.rstrip()) for line in file]

    with open("timestamps/chunk_random_1980_1_1_00_00-2023_1_1_00_00_samples_5000_02_dates.txt") as file:
        timestamps_2 = [parser.parse(line.rstrip()) for line in file]

    reference = xr.open_dataset("/Users/andre/projects/climatenet_v2/data/z500_mean_values.nc")

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
                data=seg_data, 
                dims=["ts", "latitude", "longitude"],
                coords={
                    "ts": ts,
                    "longitude": reference.coords["longitude"],
                    "latitude": reference.coords["latitude"]
                },
                attrs={
                    # might want to add additional metadata here
                    "dataset_name": annotation.dataset_name,
                    "description": annotation.description,
                    "annotator_name": annotation.owner_name
                })
            
            # Save to file
            data = data.to_dataset(name="label")
            data.to_netcdf(
                path=f"dataset/{annotation.annotation_id}.nc", 
                encoding={"label": {"zlib": True, "complevel": 9}}
            )


if __name__ == "__main__":
    with wk.webknossos_context(token=AUTH_TOKEN):
        main()