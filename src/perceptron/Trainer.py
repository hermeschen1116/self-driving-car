from typing import Tuple

from polars.dependencies import numpy
from sklearn.metrics import accuracy_score


def get_in_out_features(dataset) -> Tuple[int, int]:
	data_size: int = numpy.array(dataset[0]["data"]).shape[1]
	label_size: int = int(numpy.ceil(numpy.log2(len(dataset.get_column("label").unique().to_list()))))

	return data_size, label_size


def train(dataset, model, loss_function, variables) -> (int, float):
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

		current_epochs += 1
		print(f"epchs{current_epochs}, loss: {sum(all_loss)/ len(all_loss)}")
		accuracy = accuracy_score(label_true, label_predicted)

		if ((current_epochs + 1) == variables["num_epochs"].get()) or (
			(variables["optimize_target"].get() == "accuracy") and (accuracy >= variables["target_accuracy"].get())
		):
			break

	return current_epochs, accuracy


def evaluate(dataset, model, variables) -> float:
	label_true: list = dataset.get_column("label").to_list()
	label_predicted: list = []
	for row in dataset.iter_slices(4):
		data = row["data"].to_numpy()

		output = model(data)
		label_predicted += output.argmax(-1).tolist()

	return accuracy_score(label_true, label_predicted)
