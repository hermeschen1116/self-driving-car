import numpy

from perceptron.network import Module


class MeanSquareError(Module):
	def __init__(self) -> None:
		super(MeanSquareError, self).__init__()
		pass

	def forward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
		batch_size: int = y_i.shape[0]

		return numpy.power((y_i - y), 2).sum() * 0.5 / batch_size

	def backward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
		return y_i - y
