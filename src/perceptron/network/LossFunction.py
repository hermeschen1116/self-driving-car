import numpy

from perceptron.network import Module


class MeanSquareError(Module):
	def __init__(self) -> None:
		super(MeanSquareError, self).__init__()

	def forward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		batch_size: int = y_predicted.shape[0]

		return numpy.sum((numpy.power((y_predicted - y_true), 2) * 0.5), axis=0) / batch_size

	def backward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		return y_predicted - y_true
