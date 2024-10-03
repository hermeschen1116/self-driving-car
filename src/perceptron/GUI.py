import tkinter
from typing import Dict, Tuple

from matplotlib import pyplot
from matplotlib.figure import Figure

from perceptron.ui.Button import create_button
from perceptron.ui.Canvas import create_figure_canvas
from perceptron.ui.Menu import create_named_menu
from perceptron.ui.TextBox import create_named_textbox
from perceptron.ui.Window import create_window


def create_app() -> Tuple[tkinter.Tk, Dict[str, tkinter.Variable], Figure]:
	window: tkinter.Tk = create_window("Perceptron", "")

	variables: dict = {
		"data_path": tkinter.StringVar(name="data_path", value=""),
		"learning_rate": tkinter.DoubleVar(name="learning_rate", value=0.05),
		"num_epochs": tkinter.IntVar(name="num_epochs", value=1),
		"target_accuracy": tkinter.DoubleVar(name="target_accuracy", value=0.7),
		"optimize_target": tkinter.StringVar(name="optimize_target", value="epoch"),
	}

	control_group = tkinter.LabelFrame(padx=10, pady=10, border=0)
	control_group.pack(side="right")

	textbox_data = create_named_textbox(control_group, "Data", variables["data_path"])
	textbox_data.pack()

	textbox_learning_rate = create_named_textbox(control_group, "Learning Rate", variables["learning_rate"])
	textbox_learning_rate.pack()

	textbox_num_epochs = create_named_textbox(control_group, "Number of Epochs", variables["num_epochs"])
	textbox_num_epochs.pack()

	textbox_target_accuracy = create_named_textbox(control_group, "Target Accuracy", variables["target_accuracy"])
	textbox_target_accuracy.pack()

	menu_optimize_target = create_named_menu(
		control_group, "Optimize Target", ["epoch", "accuracy"], variables["optimize_target"]
	)
	menu_optimize_target.pack(fill="x")

	show = lambda: print(
		f"Learning Rate: {variables['learning_rate'].get()}, Number of Epochs: {variables['num_epochs'].get()}"
	)
	button_train = create_button(control_group, name="Train & Evaluation", function=show)
	button_train.pack(side="bottom", fill="x")

	visual_group = tkinter.LabelFrame(padx=30, pady=30, border=0)
	visual_group.pack(side="left", fill="both")

	figure: Figure = pyplot.figure()
	canvas_data = create_figure_canvas(visual_group, figure)
	canvas_data.draw()
	canvas_data.get_tk_widget().pack(side="left", fill="both")

	return window, variables, figure
