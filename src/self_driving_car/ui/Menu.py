import tkinter
from typing import List, Optional, Union


def create_named_menu(
	group: Optional[Union[tkinter.Frame, tkinter.LabelFrame]],
	name: str,
	options: List[str],
	variable: tkinter.StringVar,
) -> tkinter.LabelFrame:
	frame = tkinter.LabelFrame(group, text=name, padx=10, pady=10)
	textbox = tkinter.OptionMenu(frame, variable, *options)
	textbox.pack(side="bottom", fill="x")

	return frame
