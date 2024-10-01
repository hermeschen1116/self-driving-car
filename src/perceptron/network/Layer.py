from typing import Optional

import numpy

from perceptron.network import Module


class Linear(Module):
	def __init__(
		self,
		input_features: int,
		output_features: int,
		dtype: Optional[type] = numpy.float32,
	) -> None:
		super(Linear, self).__init__()
		self.weights: numpy.ndarray = numpy.random.rand((input_features + 1), output_features)

		self.dtype: type = dtype

	def __concatenate_bias(self, x: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = x.shape[0]
		bias: numpy.ndarray = numpy.ones((batch_size, 1, 1), dtype=self.dtype) * -1

		return numpy.concatenate((x, bias), axis=-1, dtype=self.dtype)

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		self.__gradient: numpy.ndarray = numpy.empty(0)

		return self.__concatenate_bias(x_i).dot(self.weights)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x
		if len(x_i.shape) < 3:
			print(f"Warning: x should be a batched 2d array, you input a {x.shape[0]} size batch of 1d array.")
			x_i = numpy.expand_dims(x_i, 1)

		batch_size: int = x_i.shape[0]

		return self.__concatenate_bias(x_i).sum(axis=0) / batch_size
