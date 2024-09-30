import numpy

from perceptron.network import Module


class ReLU(Module):
	def __init__(self) -> None:
		super(ReLU, self).__init__()
		pass

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		filter: numpy.ndarray = numpy.astype(x >= 0, x.dtype)

		return x * filter

	def backward(self, y_i: numpy.ndarray) -> numpy.ndarray:
		return numpy.astype(y_i >= 0, y_i.dtype)
