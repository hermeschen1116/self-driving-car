from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def create_figure_canvas(parent, figure: Figure):
	canvas = FigureCanvasTkAgg(figure, master=parent)

	return canvas
