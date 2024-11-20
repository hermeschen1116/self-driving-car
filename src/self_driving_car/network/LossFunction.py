import numpy

from self_driving_car.network import Module


class MeanSquareError(Module):
	def __init__(self) -> None:
		super(MeanSquareError, self).__init__()

	def forward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		self.__gradient = numpy.empty(0)

		return numpy.array(numpy.mean(numpy.square(y_predicted - y_true) * 0.5, axis=0))

	def backward(self, y_predicted: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
		return y_predicted - y_true
