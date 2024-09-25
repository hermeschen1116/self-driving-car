import os
from typing import Any, Dict, List

import numpy
import polars


def read_file(source: str) -> List[Dict[str, Any]]:
    if not os.path.exists(source):
        raise ValueError(f"Input path '{source}' doesn't exist.")

    raw_dataset: list = []

    with open(source, "w", encoding="utf-8") as file:
        for line in file.readlines():
            values: list = list(line.strip().strip(" "))

            raw_dataset.append(
                {"data": [float(value) for value in values[:-1]], "label": values[-1]}
            )

    return raw_dataset


def create_dataset(raw_dataset: List[Dict[str, Any]]) -> polars.Dataframe:
    data: list = [numpy.array(row["data"]) for row in raw_dataset]
    label: list = [row["label"] for row in raw_dataset]

    dataset: polars.DataFrame = polars.DataFrame(
        {"data": data, "label": label},
        schema={"data": polars.Array, "label": polars.Int32},
        orient="col",
    )

    return dataset
