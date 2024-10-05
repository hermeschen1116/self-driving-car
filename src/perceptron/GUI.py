import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from typing import Dict, Tuple

import polars
from matplotlib import pyplot
from matplotlib.figure import Figure

from perceptron.data.Preprocess import create_dataset, create_split, read_file
from perceptron.Model import Perceptron
from perceptron.network.LossFunction import MeanSquareError
from perceptron.Trainer import evaluate, get_in_out_features, train
from perceptron.ui.Button import create_button
from perceptron.ui.Canvas import create_figure_canvas
from perceptron.ui.Menu import create_named_menu
from perceptron.ui.TextBox import create_named_textbox
from perceptron.ui.Window import create_window


def create_app() -> Tuple[tkinter.Tk, Dict[str, tkinter.Variable], Figure]:
	window: tkinter.Tk = create_window("Perceptron", "")

	variables: dict = {
		"learning_rate": tkinter.DoubleVar(name="learning_rate", value=0.05),
		"num_epochs": tkinter.IntVar(name="num_epochs", value=1),
		"target_accuracy": tkinter.DoubleVar(name="target_accuracy", value=0.7),
		"optimize_target": tkinter.StringVar(name="optimize_target", value="epoch"),
	}

	control_group = tkinter.LabelFrame(padx=10, pady=10, border=0)
	control_group.pack(side="right")

	textbox_learning_rate = create_named_textbox(control_group, "Learning Rate", variables["learning_rate"])
	textbox_learning_rate.pack()

	textbox_num_epochs = create_named_textbox(control_group, "Number of Epochs (within)", variables["num_epochs"])
	textbox_num_epochs.pack()

	textbox_target_accuracy = create_named_textbox(control_group, "Target Accuracy", variables["target_accuracy"])
	textbox_target_accuracy.pack()

	menu_optimize_target = create_named_menu(
		control_group, "Optimize Target", ["epoch", "accuracy"], variables["optimize_target"]
	)
	menu_optimize_target.pack(fill="x")

	def on_button_activate():
		raw_dataset: list = read_file(askopenfilename())
		dataset: polars.DataFrame = create_dataset(raw_dataset)
		dataset_splits: dict = create_split(dataset, [2 / 3, 1 / 3])

		train_dataset: polars.DataFrame = dataset_splits["train"]
		test_dataset: polars.DataFrame = dataset_splits["test"]

		in_feature, out_feature = get_in_out_features(dataset)

		model = Perceptron(in_feature, out_feature, variables["learning_rate"].get())
		loss_function = MeanSquareError()

		train_accuracy: float = train(train_dataset, model, loss_function, variables)
		test_accuracy: float = evaluate(test_dataset, model, variables)

		result_message: str = f"""
							   Train Accuracy: {train_accuracy}
							   Test Accuracy: {test_accuracy}
							   Weight:
							   {model.show_weights()}
							   """

		messagebox.showinfo(message=result_message)

	button_train = create_button(control_group, name="Train & Evaluation", function=on_button_activate)
	button_train.pack(side="bottom", fill="x")

	visual_group = tkinter.LabelFrame(padx=30, pady=30, border=0)
	visual_group.pack(side="left", fill="both")

	figure: Figure = pyplot.figure()
	canvas_data = create_figure_canvas(visual_group, figure)
	canvas_data.draw()
	canvas_data.get_tk_widget().pack(side="top", fill="x")

	return window, variables, figure
