import numpy

from self_driving_car.network import Module


class ReLU(Module):
	def __init__(self) -> None:
		super(ReLU, self).__init__()

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		return x * (x >= 0)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return numpy.astype(x >= 0, x.dtype)


class Sigmoid(Module):
	def __init__(self) -> None:
		super(Sigmoid, self).__init__()

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		return 1 / (1 + numpy.exp(x * -1))

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return (1 / (1 + numpy.exp(x * -1))) * (1 - 1 / (1 + numpy.exp(x * -1)))
