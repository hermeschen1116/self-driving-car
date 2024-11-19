import random
import tkinter
from tkinter import LabelFrame, messagebox
from tkinter.filedialog import askopenfilename
from typing import List, Optional

import numpy
import polars
from matplotlib import pyplot
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
	"learning_rate": tkinter.DoubleVar(name="learning_rate", value=0.6),
	"num_epochs": tkinter.IntVar(name="num_epochs", value=500),
}
car, playground = None, None
car_circle, sensor_line = None, None
trajectory_x, trajectory_y, trajectory_line = [], [], None
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
		messagebox.showerror(title="Self Driving Car", message="No file selected.")
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

	global playground
	if playground is None:
		raise ValueError("Self Driving Car: playground object not initialized.")
	sensor_data = car.check_distance(playground)
	if sensor_data is None:
		return False
	input_data: numpy.ndarray = numpy.array(sensor_data)
	if controller.input_feature == 5:
		input_data = numpy.concatenate((car.car_position, input_data))

	input_data = input_data.reshape((1, input_data.shape[0]))
	angle: float = controller.forward(input_data)[0]

	global handler_angle
	if handler_angle is None:
		raise ValueError("Self Driving Car: handler_angle object not initialized.")
	handler_angle.radian = angle
	car(handler_angle, playground)

	global controller_record
	controller_record.append(input_data.tolist()[0] + [car.car_angle])

	if car.check_goal(playground):
		return True

	return None


def animation():
	global car_circle, sensor_line
	if car_circle is None:
		raise ValueError("Self Driving Car: car_circle object not initialized.")
	if sensor_line is None:
		raise ValueError("Self Driving Car: sensor_line object not initialized.")
	car_circle.remove()
	sensor_line.remove()

	result = control_car()

	if result is not None:
		log_path: str = "./dist/controller_record.txt"
		log: List[str] = [
			f"{' '.join([f'{round(value, 5):.5f}' for value in record])}\n" for record in controller_record
		]
		print("".join(log))
		if result:
			with open(log_path, "w") as file:
				file.writelines(log)
			print(f"Experiment successfully finished!\nLog file write to {log_path}.")
			messagebox.showinfo(
				title="Self Driving Car", message=f"Experiment successfully finished!\nLog file write to {log_path}."
			)
		if result is False:
			print("Experiment failed. The car is broken.")
			messagebox.showerror(title="Self Driving Car", message="Experiment failed. The car is broken.")
		return

	global car
	if car is None:
		raise ValueError("Self Driving Car: car object not initialized.")
	car_circle, sensor_line = car.draws()
	ax.add_patch(car_circle)
	ax.add_line(sensor_line)

	global trajectory_x, trajectory_y, trajectory_line
	if trajectory_line is None:
		raise ValueError("Self Driving Car: trajectory_line object not initialized.")
	trajectory_x.append(car.car_position[0])
	trajectory_y.append(car.car_position[1])
	trajectory_line.set_data(trajectory_x, trajectory_y)

	canvas_playground.draw()

	window.after(100, animation)


def on_button_train_activate() -> None:
	file_path: str = askopenfilename()
	if not file_path:
		messagebox.showerror(title="Self Driving Car", message="No file selected.")
		return
	print(f"train data file: {file_path}")

	raw_dataset: list = read_file(file_path)
	dataset: polars.DataFrame = create_dataset(raw_dataset)

	in_feature, out_feature = get_in_out_features(dataset)

	global controller
	controller = CarController(in_feature, out_feature, variables["learning_rate"].get())
	loss_function = MeanSquareError()

	current_epoch, train_loss = train(dataset, controller, loss_function, variables)

	result_message: str = f"""
Train Epochs: {current_epoch}
Train loss: {round(train_loss * 100, 2)}%
	"""
	print(result_message)

	global handler_angle
	handler_angle = LimitedAngle(0, [-40, 40])
	global trajectory_line
	trajectory_line = ax.plot([], [], color="blue", alpha=0.6)[0]

	animation()


button_data: LabelFrame = create_button(control_group, name="Import Playground Data", function=on_button_data_activate)
button_data.pack(fill="x")

button_train: LabelFrame = create_button(control_group, name="Train & Evaluation", function=on_button_train_activate)
button_train.pack(side="bottom", fill="x")

window.mainloop()
