import math
from typing import List, Optional, Union

import numpy


class Angle:
	def __init__(self, degree: float) -> None:
		self.__radian: float = math.radians(degree)

	@property
	def radian(self) -> float:
		return self.__radian

	@radian.setter
	def radian(self, radian: float) -> None:
		self.__radian = radian

	@property
	def degree(self) -> float:
		return math.degrees(self.__radian)

	@degree.setter
	def degree(self, degree: float) -> None:
		self.__radian = math.radians(degree)


class LimitedAngle:
	def __init__(self, degree: float, degree_range: List[float]) -> None:
		if len(degree_range) != 2:
			raise ValueError("LimitedAngle: angle_range should only contain 2 elements.")
		self.degree_range: List[float] = sorted(degree_range)
		self.__radian: float = math.radians(self.__check_range(degree, self.degree_range[0], self.degree_range[1]))

	@staticmethod
	def __check_range(x: float, min: float, max: float) -> float:
		if x < min:
			return min
		if x > max:
			return max

		return x

	@property
	def radian(self) -> float:
		return self.__radian

	@radian.setter
	def radian(self, radian: float) -> None:
		self.__radian = math.radians(
			self.__check_range(math.degrees(radian), self.degree_range[0], self.degree_range[1])
		)

	@property
	def degree(self) -> float:
		return math.degrees(self.__radian)

	@degree.setter
	def degree(self, degree: float) -> None:
		self.__radian = math.radians(self.__check_range(degree, self.degree_range[0], self.degree_range[1]))


class Point:
	def __init__(self, coordinate: numpy.ndarray) -> None:
		self.coordinate: numpy.ndarray = coordinate

	@property
	def x(self) -> float:
		return self.coordinate[0]

	@property
	def y(self) -> float:
		return self.coordinate[1]

	def distance_to(self, point: "Point") -> float:
		return numpy.sqrt(numpy.sum(numpy.pow(point.coordinate - self.coordinate, 2))).item()


class Radial:
	def __init__(self, base_point: Point, radian: float) -> None:
		self.__base_point: Point = base_point
		self.slope: numpy.ndarray = numpy.array([math.cos(radian), math.sin(radian)])

	@property
	def base_point(self) -> Point:
		return self.__base_point

	@base_point.setter
	def base_point(self, point: Point) -> None:
		self.__base_point = point

	@property
	def angle(self) -> float:
		return math.degrees(self.__radian)

	@angle.setter
	def angle(self, radian: float):
		self.__radian = numpy.array([math.cos(radian), math.sin(radian)])

	def contains(self, point: Point) -> bool:
		ts: numpy.ndarray = (point.coordinate - self.__base_point.coordinate) / self.slope

		return numpy.all(ts == ts[0]).item() and numpy.all(ts >= 0).item()

	def get_point(self, t: float) -> Point:
		if t < 0:
			raise ValueError("Radial.get_point: t should greater than 0.")

		point: Point = self.__base_point
		point.coordinate = point.coordinate + t * self.slope

		return point


class VerticalLineSegment:
	def __init__(self, endpoints: List[Point]) -> None:
		if endpoints[0].x != endpoints[1].x:
			raise ValueError("VerticalLineSegment: the end points not on a vertical line segment.")
		if endpoints[0].y == endpoints[1].y:
			raise ValueError("VerticalLineSegment: the end points are the same")

		self.x: float = endpoints[0].x
		self.y_min: float = min([endpoints[0].y, endpoints[1].y])
		self.y_max: float = max([endpoints[0].y, endpoints[1].y])

	def contains(self, point: Point) -> bool:
		if point.x != self.x:
			return False

		if point.y < self.y_min or point.y > self.y_max:
			return False

		return True

	def intersect_with_radial(self, radial: Radial) -> Optional[Point]:
		if radial.slope[0] == 0:
			return None

		t: float = (self.x - radial.__base_point.x) / radial.slope[0]
		intersect_point: Point = radial.get_point(t)
		if not self.contains(intersect_point):
			return None

		return intersect_point

	def distance_to_point(self, point: Point) -> float:
		if point.y < self.y_min:
			return point.distance_to(Point(numpy.array([self.x, self.y_min])))

		if point.y > self.y_max:
			return point.distance_to(Point(numpy.array([self.x, self.y_max])))

		return abs(point.x - self.x)


class HorizontalLineSegment:
	def __init__(self, endpoints: List[Point]) -> None:
		if endpoints[0].y != endpoints[1].y:
			raise ValueError("HorizontalLineSegment: the end points not on a horizontal line segment.")
		if endpoints[0].x == endpoints[1].x:
			raise ValueError("HorizontalLineSegment: the end points are the same")

		self.x_min: float = min([endpoints[0].x, endpoints[1].x])
		self.x_max: float = max([endpoints[0].x, endpoints[1].x])
		self.y: float = endpoints[0].y

	def contains(self, point: Point) -> bool:
		if point.y != self.y:
			return False

		if point.x < self.x_min or point.x > self.x_max:
			return False

		return True

	def intersect_with_radial(self, radial: Radial) -> Optional[Point]:
		if radial.slope[1] == 0:
			return None

		t: float = (self.y - radial.__base_point.y) / radial.slope[1]
		intersect_point: Point = radial.get_point(t)
		if not self.contains(intersect_point):
			return None

		return intersect_point

	def distance_to_point(self, point: Point) -> float:
		if point.x < self.x_min:
			return point.distance_to(Point(numpy.array([self.x_min, self.y])))

		if point.x > self.x_max:
			return point.distance_to(Point(numpy.array([self.x_max, self.y])))

		return abs(point.y - self.y)


def get_line_segment(point1: Point, point2: Point) -> Optional[Union[VerticalLineSegment, HorizontalLineSegment]]:
	if (point1.x == point2.x) and (point1.y != point2.y):
		return VerticalLineSegment([point1, point2])
	if (point1.y == point2.y) and (point1.x != point2.x):
		return HorizontalLineSegment([point1, point2])

	return None
