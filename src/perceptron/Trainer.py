from typing import Tuple

from polars.dependencies import numpy
from sklearn.metrics import accuracy_score


def get_in_out_features(dataset) -> Tuple[int, int]:
	data_size: int = numpy.array(dataset[0]["data"]).shape[1]
	label_size: int = int(numpy.ceil(numpy.log2(len(dataset.get_column("label").unique().to_list()))))

	return data_size, label_size


def train(dataset, model, loss_function, variables):
	label_true: list = dataset.get_column("label").to_list()
	for i in range(variables["num_epochs"].get()):
		label_predicted: list = []
		for row in dataset.iter_slices(4):
			data, one_hot_label = row["data"].to_numpy(), row["one_hot_label"].to_numpy()

			output = model(data)
			label_predicted += output.argmax(-1).tolist()

			loss_function(output, one_hot_label)
			model.optimize(loss_function.gradient)

		accuracy: float = accuracy_score(label_true, label_predicted)
		variables["train_accuracy"].set(accuracy)

		if variables["optimize_target"].get() == "accuracy" and accuracy >= variables["train_accuracy"]:
			break


def evaluate(dataset, model, variables):
	label_true: list = dataset.get_column("label").to_list()
	label_predicted: list = []
	for row in dataset.iter_slices(4):
		data = row["data"].to_numpy()

		output = model(data)
		label_predicted += output.argmax(-1).tolist()

	accuracy: float = accuracy_score(label_true, label_predicted)
	variables["test_accuracy"].set(accuracy)
