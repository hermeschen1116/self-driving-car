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
		self.weights: numpy.ndarray = numpy.random.rand(output_features, (input_features + 1))

		self.dtype: type = dtype

	def __concatenate_bias(self, x: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = x.shape[0]
		bias: numpy.ndarray = numpy.ones((batch_size, 1), dtype=self.dtype) * -1

		return numpy.concatenate((x, bias), axis=-1, dtype=self.dtype)

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient: numpy.ndarray = numpy.empty(0)

		return self.__concatenate_bias(x).dot(self.weights.T)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return self.__concatenate_bias(x)

	def optimize(self, local_gradient: numpy.ndarray) -> numpy.ndarray:
		old_weights: numpy.ndarray = self.weights

		weight_update: numpy.ndarray = self.gradient.dot(local_gradient)
		if weight_update.shape != self.weights.shape:
			raise ValueError(f"weight_update should be in shape {self.weights.shape}")
		self.weights = self.weights - weight_update.T

		return old_weights
