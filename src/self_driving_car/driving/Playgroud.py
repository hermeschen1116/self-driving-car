from typing import List, Union

import numpy
from matplotlib.lines import Line2D

from self_driving_car.driving.Geometry import HorizontalLineSegment, Point, VerticalLineSegment, get_line_segment


class Playgroud:
	def __init__(self, points: List[Point], goal: List[Point]) -> None:
		if not numpy.array_equal(points[0].coordinate, points[-1].coordinate):
			raise ValueError("Playgroud: points do not form a closed field.")

		self.edges: List[Union[HorizontalLineSegment, VerticalLineSegment]] = []

		print(goal[0].coordinate, goal[1].coordinate)
		line_segment = get_line_segment(goal[0], goal[1])
		if line_segment is None:
			raise ValueError("Playgroud: fail to get goal line")
		self.goal: Union[HorizontalLineSegment, VerticalLineSegment] = line_segment

		num_points: int = len(points)
		for i in range(num_points):
			line_segment = get_line_segment(points[i], points[(i + 1) % num_points])
			if line_segment is not None:
				self.edges.append(line_segment)

	def draw(self) -> List[Line2D]:
		lines: List[Line2D] = []
		for edge in self.edges:
			lines.append(edge.draw())
		lines.append(self.goal.draw(color="blue"))

		return lines
