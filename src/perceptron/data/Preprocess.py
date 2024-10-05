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
			raw_dataset.append({"data": [float(eval(value)) for value in values[:-1]], "label": int(eval(values[-1]))})

	return raw_dataset


def label_regularization(dataset: polars.DataFrame) -> polars.DataFrame:
	old_labels: list = dataset.get_column("label").unique().to_list()
	new_labels_map: dict = {label: i for i, label in enumerate(old_labels)}

	dataset = dataset.with_columns(polars.col("label").replace_strict(new_labels_map).alias("label"))

	return dataset


def add_one_hot_label(dataset: polars.DataFrame) -> polars.DataFrame:
	labels: list = dataset.get_column("label").unique().to_list()
	one_hot_labels: list = numpy.eye(len(labels)).tolist()
	one_hot_labels_map: dict = dict(zip(labels, one_hot_labels))

	dataset = dataset.with_columns(polars.col("label").replace_strict(one_hot_labels_map).alias("one_hot_label"))

	return dataset


def create_dataset(raw_dataset: List[Dict[str, Any]]) -> polars.DataFrame:
	data: list = [numpy.array(row["data"]) for row in raw_dataset]
	label: list = [row["label"] for row in raw_dataset]

	dataset: polars.DataFrame = polars.DataFrame(
		{"data": data, "label": label},
		schema={"data": polars.Array, "label": polars.Int64},
		orient="col",
	)

	dataset = label_regularization(dataset)
	dataset = add_one_hot_label(dataset)

	return dataset


def create_split(dataset: polars.DataFrame, split: List[float]) -> Union[polars.DataFrame, Dict[str, polars.DataFrame]]:
	if sum(split) != 1:
		raise ValueError("Summation of split should be 1")

	shuffled_dataset: polars.DataFrame = dataset.sample(fraction=1, shuffle=True)
	dataset_size: int = len(shuffled_dataset)

	match len(split):
		case 1:
			return shuffled_dataset
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
			raise ValueError("length of split should be in [1, 3].")
