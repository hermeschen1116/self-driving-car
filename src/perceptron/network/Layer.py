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

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = x.shape[0]
		bias: numpy.ndarray = numpy.ones((batch_size, 1), dtype=self.dtype)

		x_with_bias: numpy.ndarray = numpy.concatenate((x, bias), axis=1, dtype=self.dtype)

		return x_with_bias.dot(self.weights)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return x
