"""MediaPipe hand tracker adapter for the real-time interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


DEFAULT_MODEL_NAME = "hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    """Extract hand landmarks from webcam frames using MediaPipe Tasks."""

    def __init__(
        self,
        model_asset_path: Optional[str] = None,
        num_hands: int = 1,
    ) -> None:
        self._drawing_utils = None
        self.hand_connections = HAND_CONNECTIONS
        self._timestamp_ms = 0

        resolved_model = self._resolve_model_path(model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _resolve_model_path(self, model_asset_path: Optional[str]) -> Path:
        if model_asset_path is not None:
            candidate = Path(model_asset_path)
            if candidate.exists():
                return candidate

        local_candidate = Path(__file__).with_name(DEFAULT_MODEL_NAME)
        if local_candidate.exists():
            return local_candidate

        detection_candidate = Path(__file__).resolve().parent.parent / "detection" / DEFAULT_MODEL_NAME
        if detection_candidate.exists():
            return detection_candidate

        raise FileNotFoundError(
            "Could not find hand_landmarker.task. Expected it in "
            "src/interface/ or src/detection/, or pass model_asset_path explicitly."
        )

    def _next_timestamp(self) -> int:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        self._timestamp_ms += 1
        return self._timestamp_ms

    def _compute_bbox(
        self,
        landmarks: List[Any],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        xs = np.array([point.x for point in landmarks], dtype=np.float32)
        ys = np.array([point.y for point in landmarks], dtype=np.float32)

        x_min = int(np.clip(xs.min() * frame_width, 0, frame_width - 1))
        y_min = int(np.clip(ys.min() * frame_height, 0, frame_height - 1))
        x_max = int(np.clip(xs.max() * frame_width, 0, frame_width - 1))
        y_max = int(np.clip(ys.max() * frame_height, 0, frame_height - 1))
        return (x_min, y_min, x_max, y_max)

    def get_hand_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Return normalized landmarks, bbox and drawable landmarks for one frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp())

        if not result.hand_landmarks:
            return {"landmarks": None, "bbox": None, "raw_landmarks": None}

        first_hand = result.hand_landmarks[0]
        flat_landmarks = np.array(
            [coord for point in first_hand for coord in (point.x, point.y, point.z)],
            dtype=np.float32,
        )
        bbox = self._compute_bbox(
            first_hand,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        raw_landmarks = first_hand

        return {
            "landmarks": flat_landmarks,
            "bbox": bbox,
            "raw_landmarks": raw_landmarks,
        }

    def draw_landmarks(self, frame: np.ndarray, raw_landmarks: Any) -> None:
        """Draw landmarks and hand connections directly with OpenCV."""
        if raw_landmarks is None:
            return

        height, width = frame.shape[:2]
        points: List[Tuple[int, int]] = []
        for point in raw_landmarks:
            x = int(np.clip(point.x * width, 0, width - 1))
            y = int(np.clip(point.y * height, 0, height - 1))
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for connection in self.hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (0, 180, 255), 2)

    def close(self) -> None:
        self._landmarker.close()
