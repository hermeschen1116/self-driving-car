import numpy


class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        return numpy.power((y_i - y), 2).sum() * 0.5

    def backward(self, y_i: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        return y_i - y
