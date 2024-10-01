import numpy

from perceptron.network import Module


class MeanSquareError(Module):
	def __init__(self) -> None:
		super(MeanSquareError, self).__init__()

	def forward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		y_i: numpy.ndarray = y_predicted
		if len(y_i.shape) < 3:
			print(
				f"Warning: y_prediction should be a batched 2d array, you input a {y_i.shape[0]} size batch of 1d array."
			)
			y_i = numpy.expand_dims(y_i, 1)

		y: numpy.ndarray = y_true
		if len(y.shape) < 3:
			print(f"Warning: y_true should be a batched 2d array, you input a {y_i.shape[0]} size batch of 1d array.")
			y = numpy.expand_dims(y, 1)

		self.__gradient = numpy.empty(0)

		batch_size: int = y_i.shape[0]

		return numpy.sum((numpy.power((y_i - y), 2) * 0.5), axis=0) / batch_size

	def backward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		y_i: numpy.ndarray = y_predicted
		if len(y_i.shape) < 3:
			print(
				f"Warning: y_prediction should be a batched 2d array, you input a {y_i.shape[0]} size batch of 1d array."
			)
			y_i = numpy.expand_dims(y_i, 1)

		y: numpy.ndarray = y_true
		if len(y.shape) < 3:
			print(f"Warning: y_true should be a batched 2d array, you input a {y_i.shape[0]} size batch of 1d array.")
			y = numpy.expand_dims(y, 1)

		batch_size: int = y_i.shape[0]

		return numpy.sum(y_i - y, axis=0) / batch_size
