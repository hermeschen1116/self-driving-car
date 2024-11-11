from typing import List, Tuple

from self_driving_car.car.Car import Car


class Playgroud:
	def __init__(self, points: List[Tuple[int, int]], cars: List[Car]) -> None:
		if points[0] != points[-1]:
			raise ValueError("Playgroud: points do not form a closed field.")

		self.points: List[Tuple[int, int]] = points
		self.cars: List[Car] = cars

	def
