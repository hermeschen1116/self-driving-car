from abc import ABC, abstractmethod

import numpy


class Module(ABC):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__()

	@classmethod
	@abstractmethod
	def forward(cls, x: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
		return x

	@classmethod
	@abstractmethod
	def backward(cls, y_i: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
		return y_i
