from typing import Literal, Optional

import numpy


class Perceptron:
    def __init__(
        self,
        input_features: int,
        bias: bool,
        weight_init_method: Optional[Literal["one", "zero", "random"]] = "random",
        dtype: Optional[type] = numpy.float32,
    ) -> None:
        match weight_init_method:
            case "random":
                self.weights: numpy.ndarray = numpy.random.rand((input_features + 1))
            case "one":
                self.weights: numpy.ndarray = numpy.ones((input_features + 1), dtype=dtype)
            case "zero":
                self.weights: numpy.ndarray = numpy.zeros((input_features + 1), dtype=dtype)
            case _:
                raise ValueError(
                    f"Weight init method should be 'randome', 'one', or 'zero', you input {weight_init_method}."
                )

        if bias:
            self.bias: numpy.ndarray = numpy.ones((1), dtype=dtype)
        else:
            self.bias: numpy.ndarray = numpy.zeros((1), dtype=dtype)

        self.dtype: type = dtype

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        input: numpy.ndarray = numpy.ndarray(x, dtype=self.dtype)

        return input * self.weights + self.bias
