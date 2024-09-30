import numpy


class MSELoss:
	def __init__(self) -> None:
		pass

	def forward(self, y_prediction: numpy.ndarray, y_truth: numpy.ndarray) -> numpy.ndarray:
		difference: numpy.ndarray = y_prediction - y_truth

		return difference.dot(difference) * 0.5

	def backward(self, y_prediction: numpy.ndarray, y_truth: numpy.ndarray) -> numpy.ndarray:
		return y_prediction - y_truth
