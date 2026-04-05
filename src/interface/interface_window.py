"""PySide6 main window for the sign-language interface."""

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
    WEBCAM_MIRROR,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    INPUT_SIZE,
    InterfaceState,
    MAX_FPS,
    WEBCAM_INDEX,
    WINDOW_NAME,
    draw_bbox,
    draw_landmarks,
    get_hand_data,
    prepare_landmarks,
    predict_sign,
)


class MainWindow(QMainWindow):
    """Main Qt window for real-time sign-language inference."""

    def __init__(self, tracker: Any, model: Any) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_NAME)
        self._tracker = tracker
        self._model = model
        self._state = InterfaceState()
        self._capture = cv2.VideoCapture(WEBCAM_INDEX)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {WEBCAM_INDEX}.")
        self._previous_time = time.time()
        self._build_ui()
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
        panel.addWidget(pred_group)

        # seq_group = QGroupBox("Séquence")
        # seq_layout = QVBoxLayout(seq_group)
        # self._seq_label = QLabel("-")
        # self._seq_label.setWordWrap(True)
        # seq_font = QFont()
        # seq_font.setPointSize(18)
        # self._seq_label.setFont(seq_font)
        # self._seq_label.setAlignment(Qt.AlignCenter)
        # seq_layout.addWidget(self._seq_label)
        # panel.addWidget(seq_group)

        stats_group = QGroupBox("Stats")
        stats_layout = QVBoxLayout(stats_group)
        self._fps_label = QLabel("FPS: --")
        stats_layout.addWidget(self._fps_label)
        # self._mode_label = QLabel(f"Mode: {self._state.mode_name}")
        # stats_layout.addWidget(self._mode_label)
        panel.addWidget(stats_group)

        ctrl_group = QGroupBox("Contrôles")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self._cb_landmarks = QCheckBox("Landmarks  [L]")
        self._cb_landmarks.setChecked(self._state.show_landmarks)
        self._cb_landmarks.toggled.connect(
            lambda value: setattr(self._state, "show_landmarks", value)
        )
        self._cb_bbox = QCheckBox("Bounding box  [B]")
        self._cb_bbox.setChecked(self._state.show_bounding_box)
        self._cb_bbox.toggled.connect(
            lambda value: setattr(self._state, "show_bounding_box", value)
        )
        # self._cb_seq_mode = QCheckBox("Mode séquence  [M]")
        # self._cb_seq_mode.setChecked(self._state.sequence_mode)
        # self._cb_seq_mode.toggled.connect(self._toggle_sequence_mode)
        # self._btn_clear = QPushButton("Effacer séquence  [C]")
        # self._btn_clear.clicked.connect(self._state.clear_sequence)
        ctrl_layout.addWidget(self._cb_landmarks)
        ctrl_layout.addWidget(self._cb_bbox)
        # ctrl_layout.addWidget(self._cb_seq_mode)
        # ctrl_layout.addWidget(self._btn_clear)
        panel.addWidget(ctrl_group)

        panel.addStretch()
        root.addLayout(panel)

    # def _toggle_sequence_mode(self, value: bool) -> None:
    # 	self._state.sequence_mode = value
    # 	self._state.reset_candidate()
    # 	self._mode_label.setText(f"Mode: {self._state.mode_name}")

    def _on_frame(self) -> None:
        success, frame = self._capture.read()
        if not success:
            return

        if WEBCAM_MIRROR:
            frame = cv2.flip(frame, 1)

        hand_data = get_hand_data(self._tracker, frame)
        landmark_vector = prepare_landmarks(hand_data.get("landmarks"), INPUT_SIZE)

        inference = predict_sign(
            model=self._model,
            landmark_vector=landmark_vector,
            class_names=CLASS_NAMES,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        # update_sequence_state(self._state, inference)

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
        # self._seq_label.setText(self._state.sequence_text)

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
        # elif key == Qt.Key_M:
        # 	self._cb_seq_mode.setChecked(not self._state.sequence_mode)
        # elif key == Qt.Key_C:
        # 	self._state.clear_sequence()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:
        self._timer.stop()
        self._capture.release()
        self._tracker.close()
        super().closeEvent(event)