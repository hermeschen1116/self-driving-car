import numpy


class ReLU:
    def __init__(self) -> None:
        pass

    def forwared(self, x: numpy.ndarray) -> numpy.ndarray:
        if x >= 0:
            return x
        else:
            return numpy.zeros((1), dtype=x.dtype)
