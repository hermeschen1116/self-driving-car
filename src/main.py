import numpy

from perceptron.Model import Perceptron
from perceptron.network.LossFunction import MeanSquareError

data = numpy.array([[[-0.900000, 1.900000]], [[-0.903083, 1.978459]], [[-0.912312, 2.056434]]])
label = numpy.array([[[1]], [[1]], [[1]]])

model = Perceptron(2, 1, 0.05)
loss_fn = MeanSquareError()

output = model(data)
loss = loss_fn(output, label)
model.optimize(loss_fn.gradient)
