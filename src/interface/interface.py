from __future__ import annotations

import sys
import argparse

from PySide6.QtWidgets import QApplication

from interface_core import (
    MODE_CONFIGS,
    import_custom_tracker,
    load_model,
)
from interface_window import MainWindow


def run_interface(mode: str) -> None:
    """Run the Qt webcam application."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"mode doit être 'Letter' ou 'Word', reçu : '{mode}'")

    cfg = MODE_CONFIGS[mode]
    tracker = import_custom_tracker()

    model = load_model(
        model_path=cfg["model_path"],
        input_size=cfg["input_size"],
        class_names=cfg["class_names"],
        mode=mode,
    )
    if model is None:
        print("Warning: UI started without model. Predictions will stay at '?' and 0%.")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(tracker=tracker, model=model, mode=mode)
    window.show()
    sys.exit(app.exec())


def main() -> None:
    """Entry point for the real-time interface."""
    parser = argparse.ArgumentParser(description="Sign language interface")
    parser.add_argument(
        "--mode",
        choices=["Letter", "Word"],
        default="Letter",
        help="Mode de prédiction : 'Letter' (défaut) ou 'Word'",
    )
    args = parser.parse_args()
    run_interface(mode=args.mode)


if __name__ == "__main__":
    main()