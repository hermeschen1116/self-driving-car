import numpy

from perceptron.network import Module


class ReLU(Module):
	def __init__(self) -> None:
		super(ReLU, self).__init__()

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		self.__gradient: numpy.ndarray = numpy.empty(0)

		return x_i * (x_i >= 0)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		return numpy.astype(x_i >= 0, x_i.dtype)


class Sigmoid(Module):
	def __init__(self) -> None:
		super(Sigmoid, self).__init__()

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		self.__gradient: numpy.ndarray = numpy.empty(0)

		return 1 / (1 + numpy.exp(x_i * -1))

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		return (1 / (1 + numpy.exp(x_i * -1))) * (1 - 1 / (1 + numpy.exp(x_i * -1)))
