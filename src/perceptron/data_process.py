import os
from typing import Any, Dict, List


def read_file(source: str) -> List[Dict[str, Any]]:
	if not os.path.exists(source):
		raise ValueError(f"Input path '{source}' doesn't exist.")

	raw_dataset: list = []

	with open(source, "w", encoding="utf-8") as file:
		for line in file.readlines():
			values: list = list(line.strip().strip(" "))

			raw_dataset.append({"data": values[:-1], "label": values[-1]})

	return raw_dataset
