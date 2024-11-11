import os
from typing import Any, Dict, List, Union

import numpy
import polars


def read_file(source: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
	if not os.path.exists(source):
		raise ValueError(f"Input path '{source}' doesn't exist.")

	raw_dataset: list = []

	with open(source, "r", encoding="utf-8") as file:
		for line in file.readlines():
			values: list = list(line.strip().split(" "))
			raw_dataset.append({
				"data": [float(eval(value)) for value in values[:-1]],
				"label": float(eval(values[-1])),
			})

	return raw_dataset


def create_dataset(raw_dataset: List[Dict[str, Any]]) -> polars.DataFrame:
	data: list = [numpy.array(row["data"]) for row in raw_dataset]
	label: list = [row["label"] for row in raw_dataset]

	dataset: polars.DataFrame = polars.DataFrame(
		{"data": data, "label": label}, schema={"data": polars.Array, "label": polars.Float64}, orient="col"
	)

	return dataset


def create_split(dataset: polars.DataFrame, split: List[float]) -> Dict[str, polars.DataFrame]:
	if sum(split) != 1:
		raise ValueError("Summation of split should be 1")

	if len(dataset) <= 4:
		return {"train": dataset, "test": dataset}

	shuffled_dataset: polars.DataFrame = dataset.sample(fraction=1, shuffle=True)
	dataset_size: int = len(shuffled_dataset)

	match len(split):
		case 2:
			num_row_train_split: int = int(dataset_size * split[0])

			train_dataset: polars.DataFrame = shuffled_dataset[0:num_row_train_split]
			test_dataset: polars.DataFrame = shuffled_dataset[num_row_train_split:]

			return {"train": train_dataset, "test": test_dataset}
		case 3:
			num_row_train_split: int = int(dataset_size * split[0])
			num_row_validation_split: int = int(dataset_size * split[1])

			train_dataset: polars.DataFrame = shuffled_dataset[0:num_row_train_split]
			validation_dataset: polars.DataFrame = shuffled_dataset[
				num_row_train_split : num_row_train_split + num_row_validation_split
			]
			test_dataset: polars.DataFrame = shuffled_dataset[num_row_train_split + num_row_validation_split :]

			return {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
		case _:
			raise ValueError("length of split should be in [2, 3].")