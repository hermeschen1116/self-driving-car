import math
from typing import Dict, List, Tuple

import numpy

from self_driving_car.driving.Geometry import LimitedAngle, Point, Radial
from self_driving_car.driving.Playgroud import Playgroud


class Car:
	def __init__(
		self, radius: float = 3, initial_position: numpy.ndarray = numpy.zeros((1, 2)), initial_direction: float = 90
	) -> None:
		self.__radius: float = radius
		self.__car_angle: LimitedAngle = LimitedAngle(90, [-90, 270])
		self.__position: Point = Point(initial_position)
		self.__sensor: Dict[str, Radial] = {
			"left": Radial(self.__position, math.radians(self.__car_angle.degree + 45)),
			"front": Radial(self.__position, self.__car_angle.radian),
			"right": Radial(self.__position, math.radians(self.__car_angle.degree - 45)),
		}

	def __call__(self, handler_angle: LimitedAngle, playground: Playgroud) -> None:
		self.__turn(handler_angle)
		self.__move(handler_angle)

	@property
	def car_length(self) -> float:
		return self.__radius * 2

	@property
	def car_angle(self) -> float:
		return self.__car_angle.degree

	def __turn(self, handler_angle: LimitedAngle) -> None:
		new_car_degree: float = self.__car_angle.degree - math.degrees(
			math.asin(math.sin(handler_angle.radian) * 2 / self.car_length)
		)

		self.__car_angle.degree = new_car_degree

	def __move(self, handler_angle: LimitedAngle) -> None:
		new_car_coordinate: numpy.ndarray = self.__position.coordinate

		car_radian, handler_radian = self.__car_angle.radian, handler_angle.radian
		radian_sum: float = car_radian + handler_radian
		new_car_coordinate[0] = (
			new_car_coordinate[0] + math.cos(radian_sum) + math.sin(handler_radian) * math.sin(car_radian)
		)
		new_car_coordinate[1] = (
			new_car_coordinate[1] + math.sin(radian_sum) - math.sin(handler_radian) * math.cos(car_radian)
		)

		self.__position.coordinate = new_car_coordinate

	def __update_sensors(self) -> None:
		self.__sensor["left"].base_point = self.__position
		self.__sensor["left"].angle = math.radians(self.__car_angle.degree + 45)
		self.__sensor["front"].base_point = self.__position
		self.__sensor["front"].angle = self.__car_angle.radian
		self.__sensor["right"].base_point = self.__position
		self.__sensor["right"].angle = math.radians(self.__car_angle.degree - 45)

	def sensor(self, playground: Playgroud) -> Tuple[float, float, float]:
		left_distance: List[float] = []
		front_distance: List[float] = []
		right_distance: List[float] = []

		for line in playground.lines:
			left_intersect = line.intersect_with_radial(self.__sensor["left"])
			if left_intersect is not None:
				left_distance.append(self.__position.distance_to(left_intersect))
			front_intersect = line.intersect_with_radial(self.__sensor["front"])
			if front_intersect is not None:
				left_distance.append(self.__position.distance_to(front_intersect))
			right_intersect = line.intersect_with_radial(self.__sensor["right"])
			if right_intersect is not None:
				right_distance.append(self.__position.distance_to(right_intersect))

		return min(left_distance), min(front_distance), min(right_distance)
