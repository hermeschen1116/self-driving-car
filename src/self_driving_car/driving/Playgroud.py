from typing import List, Union

from self_driving_car.driving.Geometry import HorizontalLineSegment, Point, VerticalLineSegment, get_line_segment


class Playgroud:
	def __init__(self, points: List[Point], goal: List[Point]) -> None:
		if points[0].coordinate != points[-1].coordinate:
			raise ValueError("Playgroud: points do not form a closed field.")

		self.lines: List[Union[HorizontalLineSegment, VerticalLineSegment]] = []

		line_segment = get_line_segment(goal[0], goal[1])
		if line_segment is None:
			raise ValueError("Playgroud: fail to get goal line")
		self.goal: Union[HorizontalLineSegment, VerticalLineSegment] = line_segment

		num_points: int = len(points)
		for i in range(num_points):
			line_segment = get_line_segment(points[i], points[(i + 1) % num_points])
			if line_segment is not None:
				self.lines.append(line_segment)
