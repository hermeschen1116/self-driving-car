from typing import List, Tuple

import numpy
from matplotlib.lines import Line2D

from self_driving_car.driving.Geometry import LineSegment, Point


class Playgroud:
	def __init__(self, points: List[Point], goal: List[Point]) -> None:
		if not numpy.array_equal(points[0].coordinate, points[-1].coordinate):
			raise ValueError("Playgroud: points do not form a closed field.")

		self.edges: List[LineSegment] = []

		print(goal[0].coordinate, goal[1].coordinate)
		self.goal: LineSegment = LineSegment(goal[0], goal[1])

		points = points[:-1]
		num_points: int = len(points)
		for i in range(num_points):
			self.edges.append(LineSegment(points[i], points[(i + 1) % num_points]))

	def playground_range(self) -> Tuple[float, float, float, float]:
		x: List[float] = []
		for edge in self.edges:
			x += [edge.endpoint1.x, edge.endpoint2.x]
		y: List[float] = []
		for edge in self.edges:
			y += [edge.endpoint1.y, edge.endpoint2.y]

		return min(x), max(x), min(y), max(y)

	def draw(self) -> List[Line2D]:
		lines: List[Line2D] = []
		for edge in self.edges:
			lines.append(edge.draw())
		lines.append(self.goal.draw(color="blue"))

		return lines
