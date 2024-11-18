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
car, playground, playground_edges = None, None, []

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

canvas_playground: FigureCanvasTkAgg = create_figure_canvas(visual_group, fig)
canvas_playground.get_tk_widget().pack(side="left", fill="x")


def on_button_data_activate():
	file_path: str = askopenfilename()
	print(f"playground data file: {file_path}")

	raw_data = read_playground_file(file_path)
	car = Car(initial_position=raw_data[0], initial_direction=raw_data[1])
	playgroud = Playgroud(raw_data[3], raw_data[2])

	global playground_edges
	if len(playground_edges) == 0:
		for edge in playground_edges:
			edge.remove()

	x_min, x_max, y_min, y_max = playgroud.playground_range()
	ax.set_xlim(x_min - 10, x_max + 10)
	ax.set_ylim(y_min - 10, y_max + 10)
	ax.set_aspect("equal")

	playground_edges = playgroud.draw()
	for edge in playground_edges:
		ax.add_line(edge)

	car, sensor = car.draws()
	ax.add_patch(car)
	ax.add_line(sensor)

	canvas_playground.draw()


# def on_button_train_activate() -> None:

button_data: LabelFrame = create_button(control_group, name="Import Playground Data", function=on_button_data_activate)
button_data.pack(fill="x")

button_train: LabelFrame = create_button(control_group, name="Train & Evaluation", function=on_button_data_activate)
button_train.pack(side="bottom", fill="x")

window.mainloop()
