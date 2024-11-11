import math
from typing import List, Tuple

import numpy


def get_rotate_matrix(radian: float) -> numpy.ndarray:
	return numpy.array([[numpy.cos(radian), -numpy.sin(radian)], [numpy.sin(radian), numpy.cos(radian)]])


class Car:
	def __init__(
		self, radius: float = 3, initial_position: List[float] = [0, 0], initial_direction: float = 90
	) -> None:
		self.__radius: float = radius
		self.__handler_rotation_range: Tuple[float, float] = (math.radians(-40), math.radians(40))
		self.__car_rotation_range: Tuple[float, float] = (math.radians(-90), math.radians(270))
		self.__position: List[float] = initial_position
		self.__handler_rotation: float = 0
		self.__car_rotation: float = math.radians(initial_direction)

	@property
	def car_length(self) -> float:
		return self.__radius * 2

	@property
	def car_rotation(self) -> float:
		return math.degrees(self.__car_rotation)

	def __rotate_handler(self, handler_radian: float) -> None:
		if handler_radian > self.__handler_rotation_range[1]:
			handler_radian = self.__handler_rotation_range[1]

		if handler_radian < self.__handler_rotation_range[0]:
			handler_radian = self.__handler_rotation_range[0]

		self.__handler_rotation = handler_radian

	def __turn(self) -> None:
		new_car_degree: float = (
			self.__car_rotation - math.asin(math.sin(self.__handler_rotation) * 2 / self.car_length) * math.pi
		)

		if new_car_degree > self.__car_rotation_range[1]:
			new_car_degree = self.__car_rotation_range[1]

		if new_car_degree < self.__car_rotation_range[0]:
			new_car_degree = self.__car_rotation_range[0]

		self.__car_rotation = new_car_degree

	def __move(self) -> None:
		self.__position[0] += math.cos(self.__handler_rotation + self.__car_rotation) + math.sin(
			self.__handler_rotation
		) * math.sin(self.__car_rotation)
		self.__position[1] += math.sin(self.__handler_rotation + self.__car_rotation) - math.sin(
			self.__handler_rotation
		) * math.cos(self.__car_rotation)

	def drive(self, handler_degree: float) -> None:
		self.__rotate_handler(handler_degree)
		self.__turn()
		self.__move()
