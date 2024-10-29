import random
from typing import List

import numpy
import polars
from matplotlib.axes import Axes


def get_points_groups(dataset: polars.DataFrame, predictions: List[int]) -> List[List[List[float]]]:
	points: polars.DataFrame = dataset.select("data").with_columns(
		polars.Series(name="predictions", values=predictions)
	)

	group_points: list = (
		points.group_by("predictions", maintain_order=True).agg(polars.col("data")).get_column("data").to_list()
	)
	dimension: int = numpy.array(dataset[0]["data"]).shape[1]

	groups: list = []
	for group in group_points:
		elements: list = [[] for _ in range(dimension)]
		for value in group:
			for i, e in enumerate(value):
				elements[i].append(e)
		groups.append(elements)

	return groups


def generate_point_group_color(num_group: int) -> List[str]:
	color_candidates: list = ["#FCBA03", "#FC1C03", "#FC7703", "#24FC03", "#03C2FC", "#7703FC", "#FC03C2"]
	random.shuffle(color_candidates)

	return color_candidates[:num_group]


def draw_points(ax: Axes, group_points: List[List[List[float]]], colors: List[str]) -> bool:
	dimension: int = len(group_points[0])

	match dimension:
		case 2:
			for i, group in enumerate(group_points):
				ax.scatter(group[0], group[1], c=colors[i], marker="*", label=f"class{i}")
			ax.legend()
		case 3:
			for i, group in enumerate(group_points):
				ax.scatter(group[0], group[1], group[2], c=colors[i], marker="*", label=f"class{i}")
			ax.legend()
		case _:
			return False

	return True
