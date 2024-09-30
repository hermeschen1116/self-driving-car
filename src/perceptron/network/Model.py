import random
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
		num_hidden_layers: int = 1,
		dtype: Optional[type] = numpy.float32,
	) -> None:
		super(Perceptron, self).__init__()

		hidden_layer_features: int = random.randint(input_features, output_features)

		self.layers: list = [
			{"input_layer": Linear(input_features, hidden_layer_features, dtype)},
			{"input_activation": ReLU()},
			{"hidden_layer": Linear(hidden_layer_features, hidden_layer_features, dtype)},
			{"hidden_activation": ReLU},
			{"output_layer": Linear(hidden_layer_features, output_features, dtype)},
			{"output_activation": ReLU()},
		]

		self.dtype: type = dtype

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x.astype(self.dtype)

		y_i: numpy.ndarray = x_i
		for layer in self.layers:
			y_i = layer.values()[0](x_i)
			if layer is not self.layers[-1]:
				x_i = y_i

		return y_i
