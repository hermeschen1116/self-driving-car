import random
import tkinter
from tkinter import LabelFrame
from tkinter.filedialog import askopenfilename

import numpy
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from self_driving_car.data.Preprocess import read_playground_file
from self_driving_car.driving.Car import Car
from self_driving_car.driving.Playgroud import Playgroud
from self_driving_car.ui.Button import create_button
from self_driving_car.ui.Canvas import create_figure_canvas
from self_driving_car.ui.TextBox import create_named_textbox
from self_driving_car.ui.Window import create_window

random.seed(37710)
numpy.random.seed(37710)

window: tkinter.Tk = create_window("Self Driving Car", "")

variables: dict = {
	"learning_rate": tkinter.DoubleVar(name="learning_rate", value=0.1),
	"num_epochs": tkinter.IntVar(name="num_epochs", value=20),
}
car, playground = None, None
playground_edges: list = []

control_group = tkinter.LabelFrame(padx=10, pady=10, border=0)
control_group.pack(side="right")

textbox_learning_rate: LabelFrame = create_named_textbox(control_group, "Learning Rate", variables["learning_rate"])
textbox_learning_rate.pack()

textbox_num_epochs: LabelFrame = create_named_textbox(control_group, "Number of Epochs", variables["num_epochs"])
textbox_num_epochs.pack()

visual_group: LabelFrame = tkinter.LabelFrame(padx=30, pady=30, border=0)
visual_group.pack(side="left", fill="both")

fig, ax = pyplot.subplots()
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

canvas_playground: FigureCanvasTkAgg = create_figure_canvas(visual_group, fig)
canvas_playground.get_tk_widget().pack(side="left", fill="x")


# def on_button_activate() -> None:
# 	canvas_data0.draw()
# 	file_path: str = askopenfilename()print(f"file: {file_path}")print(f"file: {file_path}")
# 	print(f"file: {file_path}")

# 	raw_dataset: list = read_file(file_path)print(f"file: {file_path}")
# 	dataset: polars.DataFrame = create_dataset(raw_dataset)
# 	dataset_splits: dict = create_split(dataset, [2 / 3, 1 / 3])

# 	train_dataset: polars.DataFrame = dataset_splits["train"]
# 	test_dataset: polars.DataFrame = dataset_splits["test"]

# 	point_groups: list = get_points_groups(test_dataset, test_dataset.get_column("label").to_list())
# 	group_colors: list = generate_point_group_color(len(point_groups))
# 	drawable: bool = draw_points(ax0, point_groups, group_colors)
# 	canvas_data0.draw()

# 	in_feature, out_feature = get_in_out_features(dataset)

# 	model = CarController(in_feature, out_feature, variables["learning_rate"].get())
# 	loss_function = MeanSquareError()

# 	current_epoch, train_accuracy = train(train_dataset, model, loss_function, ax1, canvas_data1, variables)
# 	test_accuracy: float = evaluate(test_dataset, model, ax1, canvas_data1, group_colors, variables)

# 	result_message: str = f"""
# Train Epochs: {current_epoch + 1}
# Train Accuracy: {round(train_accuracy * 100, 2)}%
# Test Accuracy: {round(test_accuracy * 100, 2)}%
# Weight:\n{model.show_weights()}
# 	"""
# 	if not drawable:
# 		result_message = f"Data dimension above 3 so it's not drawable\n{result_message}"
# 	print(result_message)
# 	messagebox.showinfo(message=result_message)


# 	ax0.clear()
# 	ax1.clear()
#
def on_button_data_activate():
	file_path: str = askopenfilename()
	print(f"playground data file: {file_path}")

	raw_data = read_playground_file(file_path)
	car = Car(initial_position=raw_data[0], initial_direction=raw_data[1])
	playgroud = Playgroud(raw_data[3], raw_data[2])

	global playground_edges
	if len(playground_edges) != 0:
		for edge in playground_edges:
			edge.remove()

	playground_edges = playgroud.draw()
	for edge in playground_edges:
		ax.add_line(edge)

	canvas_playground.draw()


button_data: LabelFrame = create_button(control_group, name="Import Playground Data", function=on_button_data_activate)
button_data.pack(fill="x")

button_train: LabelFrame = create_button(control_group, name="Train & Evaluation", function=on_button_data_activate)
button_train.pack(side="bottom", fill="x")

window.mainloop()
