import random
import tkinter
from tkinter import LabelFrame, messagebox
from tkinter.filedialog import askopenfilename
from typing import List, Optional

import numpy
import polars
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from self_driving_car.Model import CarController
from self_driving_car.Trainer import get_in_out_features, train
from self_driving_car.data.Preprocess import create_dataset, read_file, read_playground_file
from self_driving_car.driving.Car import Car
from self_driving_car.driving.Geometry import LimitedAngle
from self_driving_car.driving.Playground import Playground
from self_driving_car.network.LossFunction import MeanSquareError
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
trajectory_line = None
controller = None
handler_angle = LimitedAngle(0, [0, 0])
controller_record: List[List[float]] = []

control_group = tkinter.LabelFrame(padx=10, pady=10, border=0)
control_group.pack(side="right")

textbox_learning_rate: LabelFrame = create_named_textbox(control_group, "Learning Rate", variables["learning_rate"])
textbox_learning_rate.pack()

textbox_num_epochs: LabelFrame = create_named_textbox(control_group, "Number of Epochs", variables["num_epochs"])
textbox_num_epochs.pack()

visual_group: LabelFrame = tkinter.LabelFrame(padx=30, pady=30, border=0)
visual_group.pack(side="left", fill="both")

fig, ax = pyplot.subplots(figsize=(4, 4), dpi=100)
ax.axis("off")

canvas_playground: FigureCanvasTkAgg = create_figure_canvas(visual_group, fig)
canvas_playground.get_tk_widget().pack(side="left", fill="x")


def on_button_data_activate():
	for patch in ax.patches:
		patch.remove()
	for line in ax.lines:
		line.remove()

	file_path: str = askopenfilename()
	if not file_path:
		messagebox.showerror("No file selected.")
		return
	print(f"playground data file: {file_path}")

	raw_data = read_playground_file(file_path)

	global car, playground
	car = Car(initial_position=raw_data[0], initial_direction=raw_data[1])
	playground = Playground(raw_data[3], raw_data[2])

	playground_edges = playground.draw()
	for edge in playground_edges:
		ax.add_line(edge)

	global car_circle, sensor_line
	car_circle, sensor_line = car.draws()
	ax.add_patch(car_circle)
	ax.add_line(sensor_line)

	ax.set_aspect("equal")
	ax.autoscale(True, axis="both", tight=True)
	ax.margins(0.1, tight=True)

	canvas_playground.draw()


def control_car() -> Optional[bool]:
	global car, controller

	if car is None:
		raise ValueError("Self Driving Car: car object not initialized.")
	if controller is None:
		raise ValueError("Self Driving Car: controller object not initialized.")

	sensor_data = car.check_distance
	if sensor_data is None:
		return False
	input_data: numpy.ndarray = numpy.array(sensor_data)
	if controller.input_feature == 5:
		input_data = numpy.concatenate((car.car_position, input_data))

	angle: float = controller.forward(input_data).item()

	global handler_angle, playground
	if handler_angle is None:
		raise ValueError("Self Driving Car: handler_angle object not initialized.")
	if playground is None:
		raise ValueError("Self Driving Car: playground object not initialized.")
	handler_angle.radian = angle
	car(handler_angle, playground)

	global controller_record
	controller_record.append(input_data.tolist() + [car.car_angle])

	if car.check_goal(playground):
		return True

	return None


def update_animation(frame):
	global car_circle, sensor_line
	car_circle.remove()
	sensor_line.remove()

	result = control_car()

	if result is not None:
		global animation
		if animation is None:
			raise ValueError("Self Driving Car: animation object not initialized.")
		animation.event_source.stop()
		if result:
			log_path: str = "./controller_record.txt"
			with open(log_path, "w") as file:
				file.writelines([f"{''.join([str(value) for value in record])}\n" for record in controller_record])
			messagebox.showinfo(f"Experiment successfully finished!\nLog file write to {log_path}.")
		if result is False:
			messagebox.showerror("Experiment failed. The car broken.")

	global car
	if car is None:
		raise ValueError("Self Driving Car: car object not initialized.")
	car_circle, sensor_line = car.draws()
	ax.add_patch(car_circle)
	ax.add_line(sensor_line)

	global trajectory_line
	if trajectory_line is None:
		raise ValueError("Self Driving Car: trajectory_line object not initialized.")
	trajectory_x, trajectory_y = trajectory_line.get_data()
	trajectory_line.set_data(
		trajectory_x.tolist() + [car.car_position[0]], trajectory_y.tolist() + [car.car_position[1]]
	)

	canvas_playground.draw()


def on_button_train_activate() -> None:
	file_path: str = askopenfilename()
	if not file_path:
		messagebox.showerror("No file selected.")
		return
	print(f"train data file: {file_path}")

	raw_dataset: list = read_file(file_path)
	dataset: polars.DataFrame = create_dataset(raw_dataset)

	in_feature, out_feature = get_in_out_features(dataset)

	controller = CarController(in_feature, out_feature, variables["learning_rate"].get())
	loss_function = MeanSquareError()

	current_epoch, train_accuracy = train(dataset, controller, loss_function, variables)

	result_message: str = f"""
Train Epochs: {current_epoch + 1}
Train Accuracy: {round(train_accuracy * 100, 2)}%
	"""
	print(result_message)

	global handler_angle
	handler_angle = LimitedAngle(0, [-40, 40])
	global trajectory_line
	trajectory_line = ax.plot([], [], color="blue", alpha=0.6)

	global animation
	animation = FuncAnimation(fig, update_animation, frames=100, interval=100, repeat=False)


button_data: LabelFrame = create_button(control_group, name="Import Playground Data", function=on_button_data_activate)
button_data.pack(fill="x")

button_train: LabelFrame = create_button(control_group, name="Train & Evaluation", function=on_button_train_activate)
button_train.pack(side="bottom", fill="x")

window.mainloop()
