from typing import Optional

import numpy

from perceptron.network import Module
from perceptron.network.Activation import ReLU
from perceptron.network.Layer import Linear


class Perceptron(Module):
	def __init__(
		self,
		input_features: int,
		output_features: int,
		learnin_rate: float,
		num_hidden_layers: int = 1,
		dtype: Optional[type] = numpy.float32,
	) -> None:
		super(Perceptron, self).__init__()

		hidden_layer_features: int = int((input_features + output_features) / 2) + 1

		self.layers: list = [
			Linear(input_features, hidden_layer_features, dtype),
			ReLU(),
			Linear(hidden_layer_features, hidden_layer_features, dtype),
			ReLU,
			Linear(hidden_layer_features, output_features, dtype),
			ReLU(),
		]

		self.learning_rate: float = learnin_rate
		self.dtype: type = dtype

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x.astype(self.dtype)

		y_i: numpy.ndarray = x_i
		for layer in self.layers:
			y_i = layer(x_i)
			if layer is not self.layers[-1]:
				x_i = y_i

		return y_i

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return x

	def optimize(self, loss_gradient: numpy.ndarray) -> None:
		local_gradient: numpy.ndarray = loss_gradient * self.layers[-1].gradient

		self.layers[-2] -= self.learning_rate * local_gradient * self.layers[-2].gradient

		local_gradient *= self.layers[-2].weights * self.layers[-3].gradient

		self.layers[-4] -= self.learning_rate * local_gradient * self.layers[-4].gradient

		local_gradient *= self.layers[-4].weights * self.layers[-5].gradient

		self.layers[-6] -= self.learning_rate * local_gradient * self.layers[-6].gradient
