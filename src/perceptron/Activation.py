import numpy


def ReLU(x: numpy.ndarray) -> numpy.ndarray:
	if x >= 0:
		return x
	else:
		return numpy.zeros((1))
