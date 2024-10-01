from typing import Optional

import numpy

from perceptron.network import Module
from perceptron.network.Activation import ReLU, Sigmoid
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

		self.input_layer = Linear(input_features, hidden_layer_features, dtype)
		print(self.input_layer.weights.shape)
		self.activation0 = ReLU()
		self.hidden_layer = Linear(hidden_layer_features, hidden_layer_features, dtype)
		print(self.hidden_layer.weights.shape)
		self.activation1 = ReLU()
		self.output_layer = Linear(hidden_layer_features, output_features, dtype)
		print(self.output_layer.weights.shape)
		self.activation2 = Sigmoid()

		self.learning_rate: float = learnin_rate
		self.dtype: type = dtype

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x.astype(self.dtype)

		x_i = self.input_layer(x_i)
		x_i = self.activation0(x_i)
		x_i = self.hidden_layer(x_i)
		x_i = self.activation1(x_i)
		x_i = self.output_layer(x_i)
		y: numpy.ndarray = self.activation2(x_i)

		return y

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return x

	def optimize(self, loss_gradient: numpy.ndarray) -> None:
		local_gradient: numpy.ndarray = loss_gradient * self.activation2.gradient
		weight_update: numpy.ndarray = self.learning_rate * (local_gradient * self.output_layer.gradient)

		local_gradient = (local_gradient * self.output_layer.weights) * self.activation1.gradient
		self.output_layer.weights = self.output_layer.weights - weight_update
		weight_update = self.learning_rate * (self.hidden_layer.gradient * local_gradient)

		local_gradient = (local_gradient * self.hidden_layer.weights) * self.activation0.gradient
		self.hidden_layer.weights = self.hidden_layer.weights - weight_update
		weight_update = self.learning_rate * local_gradient.dot(self.input_layer.gradient)

		self.input_layer.weights = self.input_layer.weights - weight_update
