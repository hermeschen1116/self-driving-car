uv run pyinstaller -D -F --optimize 2 --hidden-import="PIL._tkinter_finder" --hidden-import="numpy.core.multiarray" -n perceptron src/main.py
