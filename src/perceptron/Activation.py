import numpy


class UnitStep:
    def __init__(self) -> None:
        pass

    def forwared(self, x: numpy.ndarray) -> numpy.ndarray:
        if x > 0:
            return numpy.ones((1))
        else:
            return numpy.zeros((1))
