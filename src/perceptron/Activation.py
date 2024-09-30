import numpy


class ReLU:
	def __init__(self) -> None:
		pass

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		if x >= 0:
			return x
		else:
			return numpy.zeros((1), dtype=x.dtype)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		if x >= 0:
			return numpy.ones((1), dtype=x.dtype)
		else:
			return numpy.zeros((1), dtype=x.dtype)
