from typing import Optional

import numpy


class Perceptron:
    def __init__(
        self,
        input_features: int,
        bias: bool,
        dtype: Optional[type] = numpy.float32,
    ) -> None:
        self.weights: numpy.ndarray = numpy.random.rand((input_features + 1))

        if bias:
            self.bias: numpy.ndarray = numpy.ones((1), dtype=dtype)
        else:
            self.bias: numpy.ndarray = numpy.zeros((1), dtype=dtype)

        self.dtype: type = dtype

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        input: numpy.ndarray = numpy.concatenate(x, self.bias)

        return input.dot(self.weights)
