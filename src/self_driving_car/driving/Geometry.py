import math
from typing import List, Optional

import numpy
from matplotlib.lines import Line2D


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

	def __eq__(self, value: object, /) -> bool:
		if not isinstance(value, "Point"):
			raise NotImplementedError

		return numpy.array_equal(value.coordinate, self.coordinate)

	def __repr__(self) -> str:
		return f"({self.coordinate[0]}, {self.coordinate[1]})"

	def __add__(self, value: object, /) -> numpy.ndarray:
		if not isinstance(value, Point):
			raise NotImplementedError

		if isinstance(value, numpy.ndarray):
			return self.coordinate + value

		return self.coordinate + value.coordinate

	def __sub__(self, value: object, /) -> numpy.ndarray:
		if not isinstance(value, Point):
			raise NotImplementedError

		if isinstance(value, numpy.ndarray):
			return self.coordinate - value

		return self.coordinate - value.coordinate

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
		self.direction_vector: numpy.ndarray = numpy.array([math.cos(radian), math.sin(radian)])

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
		ts: numpy.ndarray = (point - self.__base_point) / self.direction_vector

		return numpy.all(ts == ts[0]).item() and numpy.all(ts >= 0).item()

	def get_point(self, t: float) -> Point:
		if t < 0:
			raise ValueError("Radial.get_point: t should greater than 0.")

		return Point(self.__base_point + t * self.direction_vector)

	def draw(self, color: str = "red", line_width: int = 2, line_length: float = 4) -> Line2D:
		end_point: numpy.ndarray = self.__base_point + self.direction_vector * line_length
		return Line2D(
			[self.__base_point.x, end_point[0]], [self.__base_point.y, end_point[1]], color=color, linewidth=line_width
		)


class LineSegment:
	def __init__(self, endpoint1: Point, endpoint2: Point) -> None:
		if numpy.array_equal(endpoint1.coordinate, endpoint2.coordinate):
			raise ValueError("VerticalLineSegment: the end points are the same")
		self.__endpoint1: Point = endpoint1
		self.__endpoint2: Point = endpoint2

		raw_vector: numpy.ndarray = endpoint2 - endpoint1
		self.max_t: float = numpy.linalg.norm(raw_vector).item()
		self.direction_vector: numpy.ndarray = raw_vector / self.max_t

	def contains(self, point: Point) -> bool:
		ts: numpy.ndarray = (point - self.__endpoint1) / self.direction_vector

		return numpy.all(ts == ts[0]).item() and numpy.all(ts >= 0).item() and 0 <= ts[0] <= self.max_t

	def intersect_with_radial(self, radial: Radial) -> Optional[Point]:
		xs: numpy.ndarray = numpy.column_stack((self.direction_vector, -radial.direction_vector))
		y: numpy.ndarray = radial.base_point - self.__endpoint1

		try:
			_, t_segment = numpy.linalg.solve(xs, y)
		except numpy.linalg.LinAlgError:
			return None

		if t_segment < 0 or t_segment > self.max_t:
			return None

		return self.get_point(t_segment)

	def get_point(self, t: float) -> Point:
		if t < 0 or t > self.max_t:
			raise ValueError(f"LineSegment.get_point: t should be in range [0, {self.max_t}]")

		return Point(self.__endpoint1 + t * self.direction_vector)

	def distance_to_point(self, point: Point) -> float:
		if self.contains(point):
			return 0

		vector2point: numpy.ndarray = point - self.__endpoint1
		projection: float = vector2point.dot(self.direction_vector)
		if projection < 0:
			return self.__endpoint1.distance_to(point)
		if projection > self.max_t:
			return self.__endpoint2.distance_to(point)

		slope: float = self.direction_vector[1] / self.direction_vector[0]
		return abs(vector2point[1] - vector2point[0] * slope) / numpy.sqrt((1 + slope))

	def draw(self, color: str = "black", line_width: int = 2) -> Line2D:
		return Line2D(
			[self.__endpoint1.x, self.__endpoint2.x],
			[self.__endpoint1.y, self.__endpoint2.y],
			color=color,
			linewidth=line_width,
		)
