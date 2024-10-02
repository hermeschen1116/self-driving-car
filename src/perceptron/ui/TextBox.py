import tkinter
from typing import Optional, Union


def create_named_textbox(
	group: Optional[Union[tkinter.Frame, tkinter.LabelFrame]], name: str, variable
) -> tkinter.LabelFrame:
	frame = tkinter.LabelFrame(group, text=name, padx=10, pady=10)
	textbox = tkinter.Entry(
		frame,
		background="white",
		foreground="black",
		font="16",
		justify="center",
		textvariable=variable,
	)
	textbox.pack(side="bottom", fill="x")

	return frame
