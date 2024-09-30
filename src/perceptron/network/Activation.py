import numpy

from perceptron.network import Module


class ReLU(Module):
	def __init__(self) -> None:
		super(ReLU, self).__init__()

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient: numpy.ndarray = numpy.empty(0)

		filter: numpy.ndarray = numpy.astype(x >= 0, x.dtype)

		return x * filter

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return numpy.astype(x >= 0, x.dtype)
