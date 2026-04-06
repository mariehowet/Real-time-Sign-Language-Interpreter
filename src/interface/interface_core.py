"""Core logic for real-time sign-language inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.config import Config


MODEL_PATH = str(Config.MODEL_LANDMARKER)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_NAMES.insert(CLASS_NAMES.index("T"), "Space")
INPUT_SIZE = 63
CONFIDENCE_THRESHOLD = 0.6
WEBCAM_INDEX = 0
MAX_FPS = 30
WINDOW_NAME = "Sign Language Interpreter"
SHOW_LANDMARKS = True
SHOW_BOUNDING_BOX = True
WEBCAM_MIRROR = True
BBOX_MARGIN = 16
SEQUENCE_MODE = False
SEQUENCE_STABILITY_FRAMES = 8
SEQUENCE_COOLDOWN_FRAMES = 6


@dataclass
class InferenceResult:
    """Container for per-frame model output."""

    label: str = "?"
    confidence: float = 0.0
    hand_detected: bool = False


def _normalize_external_prediction(
    raw_prediction: Any,
    class_names: Sequence[str],
    confidence_threshold: float,
) -> InferenceResult:
    """Convert model outputs to the shared inference result."""
    if isinstance(raw_prediction, InferenceResult):
        prediction = raw_prediction
    elif isinstance(raw_prediction, dict):
        prediction = InferenceResult(
            label=str(raw_prediction.get("label", "?")),
            confidence=float(raw_prediction.get("confidence", 0.0)),
            hand_detected=bool(raw_prediction.get("hand_detected", True)),
        )
    else:
        raise TypeError("model.predict must return InferenceResult or dict")

    if not prediction.hand_detected or prediction.confidence < confidence_threshold:
        return InferenceResult(label="?", confidence=float(prediction.confidence), hand_detected=prediction.hand_detected)

    label = prediction.label if prediction.label in class_names else "?"
    return InferenceResult(
        label=label,
        confidence=float(prediction.confidence),
        hand_detected=prediction.hand_detected,
    )


@dataclass
class InterfaceState:
    """Mutable UI state for real-time interaction."""

    show_landmarks: bool = SHOW_LANDMARKS
    show_bounding_box: bool = SHOW_BOUNDING_BOX
    sequence_mode: bool = SEQUENCE_MODE
    current_sequence: List[str] = field(default_factory=list)
    candidate_label: Optional[str] = None
    candidate_frames: int = 0
    cooldown_frames_left: int = 0

    def reset_candidate(self) -> None:
        self.candidate_label = None
        self.candidate_frames = 0

    def clear_sequence(self) -> None:
        self.current_sequence.clear()
        self.cooldown_frames_left = 0
        self.reset_candidate()

    @property
    def mode_name(self) -> str:
        return "SEQUENCE" if self.sequence_mode else "LETTER"

    @property
    def sequence_text(self) -> str:
        return "".join(self.current_sequence) or "-"


def import_custom_tracker() -> Any:
    """Import and instantiate the project hand tracker."""
    from src.detection.hand_tracking import HandTracker  # type: ignore

    return HandTracker()


def get_hand_data(tracker: Any, frame: np.ndarray) -> Dict[str, Any]:
    """Request hand data from the tracker."""
    return tracker.get_hand_data(frame)


def build_model(input_size: int, num_classes: int) -> Optional[Any]:
    """Instantiate the project model when available."""
    try:
        from src.detection.sign_model import SignModel  # type: ignore
    except ImportError:
        print("Warning: model.py unavailable. Running without predictions.")
        return None
    return SignModel(input_size=input_size, num_classes=num_classes)


def load_model(
    model_path: Path | str, input_size: int, class_names: Sequence[str]
) -> Optional[Any]:
    """Load the trained model when available."""
    model_path = Path(model_path)
    model = build_model(input_size=input_size, num_classes=len(class_names))
    if model is None:
        return None
    if not model_path.exists():
        print(f"Warning: model file not found at {model_path}. Running without predictions.")
        return None

    try:
        model.load(model_path)
    except Exception as error:
        print(f"Warning: model loading failed: {error}")
        return None
    return model


def prepare_landmarks(landmarks: Any, input_size: int) -> Optional[np.ndarray]:
    """Validate and reshape tracker landmarks for the model."""
    if landmarks is None:
        return None
    array = np.asarray(landmarks, dtype=np.float32).reshape(-1)
    if array.size != input_size:
        print(
            f"Warning: expected {input_size} features but received {array.size}. "
            "Skipping frame."
        )
        return None
    return array


def predict_sign(
    model: Optional[Any],
    landmark_vector: Optional[np.ndarray],
    class_names: Sequence[str],
    confidence_threshold: float,
) -> InferenceResult:
    """Get a prediction from the model layer and normalize it for UI display."""
    if landmark_vector is None:
        return InferenceResult(label="?", confidence=0.0, hand_detected=False)
    if model is None:
        return InferenceResult(label="?", confidence=0.0, hand_detected=True)

    try:
        raw_prediction = model.predict(landmark_vector)
    except Exception as error:
        print(f"Warning: model.predict failed: {error}")
        return InferenceResult(label="?", confidence=0.0, hand_detected=True)
    return _normalize_external_prediction(
        raw_prediction=raw_prediction,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
    )


def update_sequence_state(state: InterfaceState, inference: InferenceResult) -> None:
    """Update the sequence buffer from the current frame prediction."""
    if not state.sequence_mode:
        return

    if state.cooldown_frames_left > 0:
        state.cooldown_frames_left -= 1

    if not inference.hand_detected or inference.label == "?":
        state.reset_candidate()
        return

    if state.cooldown_frames_left > 0:
        return

    if state.candidate_label == inference.label:
        state.candidate_frames += 1
    else:
        state.candidate_label = inference.label
        state.candidate_frames = 1

    if state.candidate_frames >= SEQUENCE_STABILITY_FRAMES:
        state.current_sequence.append(inference.label)
        state.cooldown_frames_left = SEQUENCE_COOLDOWN_FRAMES
        state.reset_candidate()


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: int = BBOX_MARGIN,
) -> None:
    """Draw the hand bounding box on a frame."""
    height, width = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    padding = max(0, int(margin))

    x_min_out = max(0, x_min - padding)
    y_min_out = max(0, y_min - padding)
    x_max_out = min(width - 1, x_max + padding)
    y_max_out = min(height - 1, y_max + padding)

    cv2.rectangle(frame, (x_min_out, y_min_out), (x_max_out, y_max_out), (0, 255, 0), 2)


def draw_landmarks(frame: np.ndarray, tracker: Any, raw_landmarks: Any) -> None:
    """Render hand landmarks when available."""
    tracker.draw_landmarks(frame, raw_landmarks)