import tkinter
from typing import Dict, Tuple

import numpy
import polars
from matplotlib.axes import Axes
from sklearn.metrics import accuracy_score

from perceptron.data.Visualize import draw_points, get_points_groups


def get_in_out_features(dataset: polars.DataFrame) -> Tuple[int, int]:
	data_size: int = numpy.array(dataset[0]["data"]).shape[1]
	label_size: int = len(dataset.get_column("label").unique().to_list())

	return data_size, label_size


def train(
	dataset: polars.DataFrame, model, loss_function, ax: Axes, canvas, variables: Dict[str, tkinter.Variable]
) -> Tuple[int, float]:
	label_true: list = dataset.get_column("label").to_list()
	accuracy: float = 0
	current_epochs: int = 0

	while True:
		label_predicted: list = []
		all_loss: list = []
		for row in dataset.iter_slices(4):
			data, one_hot_label = row["data"].to_numpy(), row["one_hot_label"].to_numpy()

			output = model(data)
			label_predicted += output.argmax(-1).tolist()

			loss: float = loss_function(output, one_hot_label)
			all_loss.append(loss)

			model.optimize(loss_function.gradient)

		print(f"epchs{current_epochs}, loss: {sum(all_loss)/ len(all_loss)}")
		accuracy = accuracy_score(label_true, label_predicted)

		if ((current_epochs + 1) == variables["num_epochs"].get()) or (
			(variables["optimize_target"].get() == "accuracy") and (accuracy >= variables["target_accuracy"].get())
		):
			break

		current_epochs += 1

	return current_epochs, accuracy


def evaluate(dataset: polars.DataFrame, model, ax: Axes, canvas, variables: Dict[str, tkinter.Variable]) -> float:
	label_true: list = dataset.get_column("label").to_list()
	label_predicted: list = []

	for row in dataset.iter_slices(4):
		data = row["data"].to_numpy()

		output = model(data)
		label_predicted += output.argmax(-1).tolist()

	group_points: list = get_points_groups(dataset, label_predicted)
	draw_points(ax, group_points)
	canvas.draw()

	return accuracy_score(label_true, label_predicted)
