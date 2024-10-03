from typing import List, Tuple, Union

import numpy
from matplotlib.figure import Figure


def get_points(dataset) -> Tuple[List[List[float]], List[int]]:
	dimension: int = numpy.array(dataset[0]["data"]).shape[1]

	data: list = [[] for _ in range(dimension)]
	for value in dataset.get_column("data").to_list():
		for i, e in enumerate(value):
			data[i].append(e)

	label: list = dataset.get_column("label").to_list()

	return data, label


def draw_points(figure: Figure, x: Union[List[float], List[List[float]]], y: List[float]):
	dimension: int = len(x)

	match dimension:
		case 1 | 2:
			ax = figure.add_subplot()
			ax.plot(*x)
		case 3:
			ax = figure.add_subplot()
		case _:
			return figure
