import numpy


class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        if x >= 0:
            return x
        else:
            return numpy.zeros((1))
