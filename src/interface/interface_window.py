from __future__ import annotations

import time
from typing import Any

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from interface_core import (
    CONFIDENCE_THRESHOLD,
    MAX_FPS,
    MODE_CONFIGS,
    SEQUENCE_LENGTH,
    WEBCAM_INDEX,
    WEBCAM_MIRROR,
    WINDOW_NAME,
    InferenceResult,
    InterfaceState,
    draw_bbox,
    draw_landmarks,
    get_hand_data,
    load_model,
    prepare_landmarks,
    prepare_sequence,
    predict_sign,
)


class MainWindow(QMainWindow):
    """Main Qt window for real-time sign-language inference."""
    def __init__(self, tracker: Any, model: Any, mode: str = "Letter") -> None:
        super().__init__()
        self._tracker = tracker
        self._model = model
        self._mode = mode
        self._state = InterfaceState()
        self._sequence_buffer: list = []
        self._recording = False
        self._last_inference = InferenceResult(label="?", confidence=0.0, hand_detected=False)
        self._capture = cv2.VideoCapture(WEBCAM_INDEX)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {WEBCAM_INDEX}.")
        self._previous_time = time.time()
        self._build_ui()
        self._refresh_mode_ui()  # état initial cohérent
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_frame)
        interval_ms = max(1, int(1000 / MAX_FPS))
        self._timer.start(interval_ms)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        # Flux vidéo
        self._video_label = QLabel()
        self._video_label.setFixedSize(640, 480)
        self._video_label.setStyleSheet("background: black; border-radius: 4px;")
        self._video_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self._video_label)

        panel = QVBoxLayout()
        panel.setSpacing(10)

        pred_group = QGroupBox("Prédiction")
        pred_layout = QVBoxLayout(pred_group)

        self._pred_label = QLabel("?")
        big_font = QFont()
        big_font.setPointSize(52)
        big_font.setBold(True)
        self._pred_label.setFont(big_font)
        self._pred_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self._pred_label)

        self._conf_label = QLabel("Conf: 0%")
        self._conf_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self._conf_label)

        # Buffer
        self._buffer_label = QLabel(f"Buffer: 0 / {SEQUENCE_LENGTH} frames")
        self._buffer_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self._buffer_label)

        panel.addWidget(pred_group)

        stats_group = QGroupBox("Stats")
        stats_layout = QVBoxLayout(stats_group)
        self._fps_label = QLabel("FPS: --")
        stats_layout.addWidget(self._fps_label)
        self._mode_label = QLabel(f"Mode: {self._mode}")
        stats_layout.addWidget(self._mode_label)
        panel.addWidget(stats_group)

        ctrl_group = QGroupBox("Contrôles")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self._btn_switch_mode = QPushButton("")
        self._btn_switch_mode.clicked.connect(self._switch_mode)
        ctrl_layout.addWidget(self._btn_switch_mode)

        self._cb_landmarks = QCheckBox("Landmarks  [L]")
        self._cb_landmarks.setChecked(self._state.show_landmarks)
        self._cb_landmarks.toggled.connect(
            lambda value: setattr(self._state, "show_landmarks", value)
        )
        ctrl_layout.addWidget(self._cb_landmarks)

        self._cb_bbox = QCheckBox("Bounding box  [B]")
        self._cb_bbox.setChecked(self._state.show_bounding_box)
        self._cb_bbox.toggled.connect(
            lambda value: setattr(self._state, "show_bounding_box", value)
        )
        ctrl_layout.addWidget(self._cb_bbox)

        # Bouton clear
        self._btn_clear = QPushButton("Effacer buffer  [C]")
        self._btn_clear.clicked.connect(self._clear_buffer)
        ctrl_layout.addWidget(self._btn_clear)

        panel.addWidget(ctrl_group)
        panel.addStretch()
        root.addLayout(panel)

        self._btn_record = QPushButton("⏺ Enregistrer  [R]")
        self._btn_record.setCheckable(True)
        self._btn_record.setChecked(False)
        self._btn_record.clicked.connect(self._toggle_recording)
        self._btn_record.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                border-radius: 6px;
                border: 2px solid #555;
                background-color: transparent;
                color: #aaa;
            }
            QPushButton:checked {
                background-color: #cc3333;
                border-color: #cc3333;
                color: white;
            }
            QPushButton:hover:!checked {
                border-color: #cc3333;
                color: #ddd;
            }
        """)
        ctrl_layout.addWidget(self._btn_record)

    def _refresh_mode_ui(self) -> None:
        """Met à jour tous les widgets dépendants du mode courant."""
        is_word = self._mode == "Word"
        self.setWindowTitle(f"{WINDOW_NAME} — {self._mode}")
        self._mode_label.setText(f"Mode: {self._mode}")
        self._btn_switch_mode.setText(
            "Passer en mode Word" if not is_word else "Passer en mode Letter"
        )
        self._buffer_label.setVisible(is_word)
        self._btn_clear.setVisible(is_word)
        self._pred_label.setText("?")
        self._conf_label.setText("Conf: 0%")
        self._btn_record.setVisible(is_word)
        if not is_word:
            self._recording = False

    def _clear_buffer(self) -> None:
        self._sequence_buffer.clear()
        self._buffer_label.setText(f"Buffer: 0 / {SEQUENCE_LENGTH} frames")

    def _toggle_recording(self) -> None:
        self._recording = not self._recording
        self._btn_record.setChecked(self._recording)
        self._btn_record.setText(
            "⏹ Stop  [R]" if self._recording else "⏺ Enregistrer  [R]"
        )
        if not self._recording:
            self._clear_buffer()

    def _switch_mode(self) -> None:
        """Bascule entre Letter et Word"""
        new_mode = "Word" if self._mode == "Letter" else "Letter"
        cfg = MODE_CONFIGS[new_mode]

        new_model = load_model(
            model_path=cfg["model_path"],
            input_size=cfg["input_size"],
            class_names=cfg["class_names"],
            mode=new_mode,
        )
        if new_model is None:
            print(f"Warning: modèle {new_mode} introuvable. Mode non changé.")
            return

        self._mode = new_mode
        self._model = new_model
        self._clear_buffer()
        self._refresh_mode_ui()
        print(f"[Mode] Basculé vers {new_mode}")

    def _on_frame(self) -> None:
        success, frame = self._capture.read()
        if not success:
            return

        if WEBCAM_MIRROR:
            frame = cv2.flip(frame, 1)

        hand_data = get_hand_data(self._tracker, frame)
        frame_vector = prepare_landmarks(hand_data.get("landmarks"))  # 63 ou None
        cfg = MODE_CONFIGS[self._mode]

        if self._mode == "Letter":
            inference = predict_sign(
                model=self._model,
                landmark_vector=frame_vector,
                class_names=cfg["class_names"],
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
        else:
            if self._recording and frame_vector is not None:
                self._sequence_buffer.append(frame_vector)
            buf_len = len(self._sequence_buffer)
            self._buffer_label.setText(
                f"{'🔴 ' if self._recording else ''}"
                f"Buffer: {buf_len} / {SEQUENCE_LENGTH} frames"

            )

            if buf_len >= SEQUENCE_LENGTH:
                sequence_vector = prepare_sequence(self._sequence_buffer)
                new_inference = predict_sign(
                    model=self._model,
                    landmark_vector=sequence_vector,
                    class_names=cfg["class_names"],
                    confidence_threshold=CONFIDENCE_THRESHOLD,

                )

                self._last_inference = new_inference
                self._recording = False
                self._btn_record.setChecked(False)
                self._btn_record.setText("⏺ Enregistrer  [R]")
                self._clear_buffer()
            inference = self._last_inference

        current_time = time.time()
        fps = 1.0 / max(current_time - self._previous_time, 1e-6)
        self._previous_time = current_time

        annotated = frame.copy()
        if self._state.show_bounding_box and hand_data.get("bbox") is not None:
            draw_bbox(annotated, hand_data["bbox"])
        if self._state.show_landmarks:
            draw_landmarks(annotated, self._tracker, hand_data.get("raw_landmarks"))

        self._pred_label.setText(inference.label)
        self._conf_label.setText(f"Conf: {inference.confidence:.0%}")
        self._fps_label.setText(f"FPS: {fps:.1f}")

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self._video_label.setPixmap(
            pixmap.scaled(
                self._video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_L:
            value = not self._state.show_landmarks
            self._state.show_landmarks = value
            self._cb_landmarks.setChecked(value)
        elif key == Qt.Key_B:
            value = not self._state.show_bounding_box
            self._state.show_bounding_box = value
            self._cb_bbox.setChecked(value)
        elif key == Qt.Key_C and self._mode == "Word":
            self._clear_buffer()
        elif key == Qt.Key_M:
            self._switch_mode()
        elif key == Qt.Key_R and self._mode == "Word":
            self._toggle_recording()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:
        self._timer.stop()
        self._capture.release()
        self._tracker.close()
        super().closeEvent(event)