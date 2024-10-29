import numpy

from self_driving_car.network import Module


class Linear(Module):
	def __init__(self, input_features: int, output_features: int, dtype: type = numpy.float32) -> None:
		super(Linear, self).__init__()

		self.weights: numpy.ndarray = self.__initialize_weights(input_features, output_features)

		self.dtype: type = dtype

	@staticmethod
	def __initialize_weights(input_features: int, output_features: int) -> numpy.ndarray:
		weight_limit: float = numpy.sqrt(1 / input_features).item()

		return numpy.random.uniform(-1 * weight_limit, weight_limit, (input_features + 1, output_features))

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		batch_size: int = x.shape[0]
		x_0: numpy.ndarray = numpy.ones((batch_size, 1), dtype=x.dtype) * -1

		return numpy.concatenate((x, x_0), axis=-1).dot(self.weights)

	def backward(self, x: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = x.shape[0]
		x_0: numpy.ndarray = numpy.ones((batch_size, 1), dtype=x.dtype) * -1

		return numpy.concatenate((x, x_0), axis=-1)

	def optimize(self, local_gradient: numpy.ndarray) -> numpy.ndarray:
		old_weights: numpy.ndarray = self.weights

		weight_update: numpy.ndarray = self.gradient.T.dot(local_gradient)
		if weight_update.shape != self.weights.shape:
			raise ValueError(f"weight_update should be in shape {self.weights.shape}")
		self.weights = self.weights - weight_update

		return old_weights
