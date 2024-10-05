import random
from typing import List

import numpy
import polars
from matplotlib.figure import Figure


def get_points_groups(dataset: polars.DataFrame, predictions: List[int]) -> List[List[List[float]]]:
	points: polars.DataFrame = dataset.select("data").with_columns(polars.lit(predictions).alias("predictions"))

	group_points: list = points.group_by("predictions").agg(polars.col("data")).get_column("data").to_list()

	dimension: int = numpy.array(dataset[0]["data"]).shape[1]

	groups: list = []
	for group in group_points:
		elements: list = [[] for _ in range(dimension)]
		for value in group:
			for i, e in enumerate(value):
				elements[i].append(e)
		groups.append(elements)

	return groups


def random_color_hex() -> str:
	return f"#{("%06x" % random.randint(0, 0xFFFFFF)).upper()}"


def generate_point_group_color(num_group: int) -> List[str]:
	colors: list = []
	while True:
		if len(colors) == num_group:
			break

		color: str = random_color_hex()
		if color not in colors:
			colors.append(color)

	return colors


def draw_points(figure: Figure, group_points: List[List[List[float]]]):
	dimension: int = len(group_points[0])
	colors: list = generate_point_group_color(len(group_points))


	match dimension:
		case 2:
			ax = figure.add_subplot()
			for i, group in enumerate(group_points):
				ax.scatter(group[0], group[1], c=colors[i])
		case 3:
			ax = figure.add_subplot(projection="3d")
			for i, group in enumerate(group_points):
				ax.scatter(group[0], group[1], group[2], c=colors[i])
		case _:
			return figure
