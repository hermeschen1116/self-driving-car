import tkinter
from typing import Dict, List, Tuple

import numpy
import polars
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, classification_report

from self_driving_car.Model import CarController
from self_driving_car.data.Visualize import draw_points, get_points_groups


def get_in_out_features(dataset: polars.DataFrame) -> Tuple[int, int]:
	data_size: int = numpy.array(dataset[0]["data"]).shape[1]
	label_size: int = 1

	return data_size, label_size


def train(
	dataset: polars.DataFrame, model: CarController, loss_function, variables: Dict[str, tkinter.Variable]
) -> Tuple[int, float]:
	total_loss: float = 0
	num_epochs: int = variables["num_epochs"].get()

	for i in range(num_epochs):
		all_loss: list = []
		for row in dataset.iter_slices(4):
			data, label = row["data"].to_numpy(), row["label"].to_numpy()
			label = label.reshape((label.shape[0], 1))

			output: numpy.ndarray = model(data)

			loss: float = loss_function(output, label)
			all_loss.append(loss)

			model.optimize(loss_function.gradient)

		total_loss = sum(all_loss) / len(all_loss)
		print(f"epchs{i}, loss: {total_loss}")

	return num_epochs, total_loss
