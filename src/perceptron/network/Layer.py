
import numpy

from perceptron.network import Module


class Linear(Module):
	def __init__(
		self,
		input_features: int,
		output_features: int,
		dtype: type = numpy.float32,
	) -> None:
		super(Linear, self).__init__()
		self.weights: numpy.ndarray = numpy.random.rand(input_features, output_features)
		self.bias: numpy.ndarray = numpy.random.rand(1, output_features)

		self.dtype: type = dtype

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		return x.dot(self.weights) + self.bias

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return x

	def optimize(self, local_gradient: numpy.ndarray) -> numpy.ndarray:
		old_weights: numpy.ndarray = self.weights

		weight_update: numpy.ndarray = self.gradient.T.dot(local_gradient)
		bias_update: numpy.ndarray = local_gradient.sum(0)
		if weight_update.shape != self.weights.shape:
			raise ValueError(f"weight_update should be in shape {self.weights.shape}")
		self.weights = self.weights - weight_update
		self.bias = self.bias - bias_update

		return old_weights
