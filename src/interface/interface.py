"""Entry point for the PySide6 sign-language interface."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from interface_core import (
	CLASS_NAMES,
	INPUT_SIZE,
	MODEL_PATH,
	import_custom_tracker,
	load_model,
)
from interface_window import MainWindow


def run_interface() -> None:
	"""Run the Qt webcam application."""
	model_path = Path(__file__).with_name(MODEL_PATH)
	tracker = import_custom_tracker()
	if tracker is None:
		print("Warning: UI started without tracker. Hand overlays/predictions require hand_tracking.py.")

	model = load_model(model_path=model_path, input_size=INPUT_SIZE, class_names=CLASS_NAMES)
	if model is None:
		print("Warning: UI started without model. Predictions will stay at '?' and 0%.")

	app = QApplication(sys.argv)
	app.setStyle("Fusion")
	window = MainWindow(tracker=tracker, model=model)
	window.show()
	sys.exit(app.exec())


def main() -> None:
	"""Entry point for the real-time interface."""
	run_interface()


if __name__ == "__main__":
	main()
