import tkinter
from typing import Callable, Optional, Union


def create_button(
	group: Optional[Union[tkinter.Frame, tkinter.LabelFrame]], name: str, function: Callable
) -> tkinter.LabelFrame:
	container = tkinter.LabelFrame(group, padx=10, pady=10, border=0)
	button = tkinter.Button(container, text=name, justify="center", command=function)
	button.pack(fill="x")

	return container
