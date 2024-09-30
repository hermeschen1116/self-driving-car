from abc import ABC, abstractmethod

import numpy


class Module(ABC):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__()

		self.__gradient: numpy.ndarray = numpy.empty(0)

	def __call__(self, *args, **kwargs) -> numpy.ndarray:
		self.__gradient = self.backward(*args, **kwargs)

		return self.forward(*args, **kwargs)

	@property
	def gradient(self) -> numpy.ndarray:
		if self.__gradient.size == 0:
			raise ValueError("Gradient hasn't been computed.")

		return self.__gradient

	@classmethod
	@abstractmethod
	def forward(cls, x: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
		return x

	@classmethod
	@abstractmethod
	def backward(cls, y: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
		return y
