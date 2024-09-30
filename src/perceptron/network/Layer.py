from typing import Optional

import numpy

from perceptron.network import Module


class Linear(Module):
	def __init__(
		self,
		input_features: int,
		output_features: int,
		bias: bool,
		dtype: Optional[type] = numpy.float32,
	) -> None:
		super(Linear, self).__init__()
		self.weights: numpy.ndarray = numpy.random.rand((input_features + 1), output_features)

		self.bias: bool = bias

		self.dtype: type = dtype

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = x.shape[0]
		bias_array: numpy.ndarray = numpy.ones((batch_size, 1), dtype=self.dtype)

		x_with_bias: numpy.ndarray = numpy.concatenate((x, bias_array), dtype=self.dtype)

		return x_with_bias.dot(self.weights)

	def backward(self, y_i: numpy.ndarray) -> numpy.ndarray:
		return y_i
