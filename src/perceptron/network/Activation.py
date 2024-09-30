import numpy

from perceptron.network import Module


class ReLU(Module):
	def __init__(self) -> None:
		super(ReLU, self).__init__()
		pass

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		if x >= 0:
			return x
		else:
			return numpy.zeros((1), dtype=x.dtype)

	def backward(self, y_i: numpy.ndarray) -> numpy.ndarray:
		if y_i >= 0:
			return numpy.ones((1), dtype=y_i.dtype)
		else:
			return numpy.zeros((1), dtype=y_i.dtype)
