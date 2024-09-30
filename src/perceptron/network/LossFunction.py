import numpy

from perceptron.network import Module


class MSELoss(Module):
	def __init__(self) -> None:
		super(MSELoss, self).__init__()
		pass

	def forward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
		return numpy.power((y_i - y), 2).sum() * 0.5

	def backward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
		return y_i - y
