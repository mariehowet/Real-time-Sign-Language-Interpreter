from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Config:
    MODELS_DIR = BASE_DIR / "assets" / "models"

    HAND_LANDMARKER = MODELS_DIR / "hand_landmarker.task"
    MODEL_CLASSIFIER = MODELS_DIR / "model.pth"
    MODEL_LANDMARKER = MODELS_DIR / "model_landmarks.pth"
