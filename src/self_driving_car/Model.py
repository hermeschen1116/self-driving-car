from typing import List

import numpy

from self_driving_car.network import Module
from self_driving_car.network.Activation import ReLU, Sigmoid
from self_driving_car.network.Layer import Linear


class Perceptron(Module):
	def __init__(
		self, input_features: int, output_features: int, learning_rate: float, dtype: type = numpy.float32
	) -> None:
		super(Perceptron, self).__init__()

		hidden_layer_features: int = int((input_features + output_features) / 2)

		self.input_layer = Linear(input_features, hidden_layer_features, dtype)
		self.activation0 = ReLU()
		self.hidden_layer = Linear(hidden_layer_features, hidden_layer_features, dtype)
		self.activation1 = ReLU()
		self.output_layer = Linear(hidden_layer_features, output_features, dtype)
		self.activation2 = Sigmoid()

		self.learning_rate: float = learning_rate
		self.dtype: type = dtype

	@property
	def weights(self) -> List[numpy.ndarray]:
		return [self.input_layer.weights.T, self.hidden_layer.weights.T, self.output_layer.weights.T]

	@staticmethod
	def __show_layer(layer_name: str, layer: numpy.ndarray) -> str:
		nodes: list = [
			f"node({layer_name}, {i}): [{', '.join([str(round(w, 5)) for w in node])}]"
			for i, node in enumerate(layer.T.tolist())
		]

		return "\n".join(nodes)

	def show_weights(self) -> str:
		return f"""
{self.__show_layer("0", self.input_layer.weights)}
{self.__show_layer("1", self.hidden_layer.weights)}
{self.__show_layer("2", self.output_layer.weights)}
		"""

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
		layer_gradient: numpy.ndarray = self.output_layer.optimize(self.learning_rate * local_gradient)

		local_gradient = local_gradient.dot(layer_gradient[:-1].T) * self.activation1.gradient
		layer_gradient = self.hidden_layer.optimize(self.learning_rate * local_gradient)

		local_gradient = local_gradient.dot(layer_gradient[:-1].T) * self.activation0.gradient
		self.input_layer.optimize(self.learning_rate * local_gradient)
