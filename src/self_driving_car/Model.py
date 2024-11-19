import numpy

from self_driving_car.network import Module
from self_driving_car.network.Activation import ReLU, Sigmoid
from self_driving_car.network.Layer import Linear


class CarController(Module):
	def __init__(
		self, input_features: int, output_features: int, learning_rate: float, dtype: type = numpy.float32
	) -> None:
		super(CarController, self).__init__()

		hidden_layer_features: int = int((input_features + output_features) / 2)
		self.__num_hidden_layer: int = 3

		self.input_feature: int = input_features
		self.input_layer = Linear(input_features, hidden_layer_features, dtype)
		self.input_activation = ReLU()
		self.hidden_layer: list = [
			Linear(hidden_layer_features, hidden_layer_features, dtype) for _ in range(self.__num_hidden_layer)
		]
		self.hidden_activation: list = [ReLU() for _ in range(self.__num_hidden_layer)]
		self.output_layer = Linear(hidden_layer_features, output_features, dtype)
		self.output_activation = Sigmoid()

		self.learning_rate: float = learning_rate
		self.dtype: type = dtype

	@staticmethod
	def __show_layer(layer_name: str, layer: numpy.ndarray) -> str:
		nodes: list = [
			f"node({layer_name}, {i}): [{', '.join([str(round(w, 5)) for w in node])}]"
			for i, node in enumerate(layer.T.tolist())
		]

		return "\n".join(nodes)

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		x_i: numpy.ndarray = x.astype(self.dtype)

		x_i = self.input_layer(x_i)
		x_i = self.input_activation(x_i)
		for i in range(self.__num_hidden_layer):
			x_i = self.hidden_layer[i](x_i)
			x_i = self.hidden_activation[i](x_i)
		x_i = self.output_layer(x_i)
		y: numpy.ndarray = self.output_activation(x_i)

		return y

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		return x

	def optimize(self, loss_gradient: numpy.ndarray) -> None:
		local_gradient: numpy.ndarray = loss_gradient * self.output_activation.gradient
		layer_gradient: numpy.ndarray = self.output_layer.optimize(self.learning_rate * local_gradient)

		for i in range(self.__num_hidden_layer - 1, -1, -1):
			local_gradient = local_gradient.dot(layer_gradient[:-1].T) * self.hidden_activation[i].gradient
			layer_gradient = self.hidden_layer[i].optimize(self.learning_rate * local_gradient)

		local_gradient = local_gradient.dot(layer_gradient[:-1].T) * self.input_activation.gradient
		self.input_layer.optimize(self.learning_rate * local_gradient)
