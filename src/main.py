import random
import tkinter
from tkinter import LabelFrame, messagebox
from tkinter.filedialog import askopenfilename

import numpy
import polars
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from self_driving_car.Model import CarController
from self_driving_car.Trainer import evaluate, get_in_out_features, train
from self_driving_car.data.Preprocess import create_dataset, create_split, read_file
from self_driving_car.data.Visualize import draw_points, generate_point_group_color, get_points_groups
from self_driving_car.network.LossFunction import MeanSquareError
from self_driving_car.ui.Button import create_button
from self_driving_car.ui.Canvas import create_figure_canvas
from self_driving_car.ui.Menu import create_named_menu
from self_driving_car.ui.TextBox import create_named_textbox
from self_driving_car.ui.Window import create_window

random.seed(37710)
numpy.random.seed(37710)

window: tkinter.Tk = create_window("Perceptron", "")

variables: dict = {
	"learning_rate": tkinter.DoubleVar(name="learning_rate", value=0.1),
	"num_epochs": tkinter.IntVar(name="num_epochs", value=20),
	"target_accuracy": tkinter.DoubleVar(name="target_accuracy", value=0.7),
	"optimize_target": tkinter.StringVar(name="optimize_target", value="epoch"),
}

control_group = tkinter.LabelFrame(padx=10, pady=10, border=0)
control_group.pack(side="right")

textbox_learning_rate: LabelFrame = create_named_textbox(control_group, "Learning Rate", variables["learning_rate"])
textbox_learning_rate.pack()

textbox_num_epochs: LabelFrame = create_named_textbox(
	control_group, "Maximum Number of Epochs", variables["num_epochs"]
)
textbox_num_epochs.pack()

textbox_target_accuracy: LabelFrame = create_named_textbox(
	control_group, "Target Train Accuracy", variables["target_accuracy"]
)
textbox_target_accuracy.pack()

menu_optimize_target: LabelFrame = create_named_menu(
	control_group, "Optimize Target", ["epoch", "accuracy"], variables["optimize_target"]
)
menu_optimize_target.pack(fill="x")

visual_group: LabelFrame = tkinter.LabelFrame(padx=30, pady=30, border=0)
visual_group.pack(side="left", fill="both")

figure0: Figure = pyplot.figure(figsize=(3, 3))
ax0: Axes = figure0.add_subplot(projection="3d")
figure1: Figure = pyplot.figure(figsize=(3, 3))
ax1: Axes = figure1.add_subplot(projection="3d")

canvas_data0: FigureCanvasTkAgg = create_figure_canvas(visual_group, figure0)
canvas_data0.get_tk_widget().pack(side="left", fill="x")
canvas_data1: FigureCanvasTkAgg = create_figure_canvas(visual_group, figure1)
canvas_data1.get_tk_widget().pack(side="right", fill="x")

# def


def on_button_activate() -> None:
	canvas_data0.draw()
	canvas_data1.draw()
	file_path: str = askopenfilename()
	print(f"file: {file_path}")

	raw_dataset: list = read_file(file_path)
	dataset: polars.DataFrame = create_dataset(raw_dataset)
	dataset_splits: dict = create_split(dataset, [2 / 3, 1 / 3])

	train_dataset: polars.DataFrame = dataset_splits["train"]
	test_dataset: polars.DataFrame = dataset_splits["test"]

	point_groups: list = get_points_groups(test_dataset, test_dataset.get_column("label").to_list())
	group_colors: list = generate_point_group_color(len(point_groups))
	drawable: bool = draw_points(ax0, point_groups, group_colors)
	canvas_data0.draw()

	in_feature, out_feature = get_in_out_features(dataset)

	model = CarController(in_feature, out_feature, variables["learning_rate"].get())
	loss_function = MeanSquareError()

	current_epoch, train_accuracy = train(train_dataset, model, loss_function, ax1, canvas_data1, variables)
	test_accuracy: float = evaluate(test_dataset, model, ax1, canvas_data1, group_colors, variables)

	result_message: str = f"""
Train Epochs: {current_epoch + 1}
Train Accuracy: {round(train_accuracy * 100, 2)}%
Test Accuracy: {round(test_accuracy * 100, 2)}%
Weight:\n{model.show_weights()}
	"""
	if not drawable:
		result_message = f"Data dimension above 3 so it's not drawable\n{result_message}"
	print(result_message)
	messagebox.showinfo(message=result_message)

	ax0.clear()
	ax1.clear()


button_train: LabelFrame = create_button(control_group, name="Train & Evaluation", function=on_button_activate)
button_train.pack(side="bottom", fill="x")

window.mainloop()
