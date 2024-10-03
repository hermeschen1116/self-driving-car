import tkinter


def create_window(title: str, app_icon_path: str) -> tkinter.Tk:
	window = tkinter.Tk()

	window.title(title)
	window.resizable(False, False)
	window.wm_title(title)
	window.config(padx=10, pady=10)
	# window.wm_iconphoto(False, app_icon_path)

	return window
