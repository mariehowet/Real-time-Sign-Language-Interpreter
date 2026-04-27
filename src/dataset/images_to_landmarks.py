import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


# Config
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "hand_landmarker.task")
DATASET_PATH    = os.path.join(BASE_DIR, "data/splits/train")
OUTPUT_CSV      = os.path.join(BASE_DIR, "landmarks_dataset.csv")

NUM_HANDS       = 1
NUM_LANDMARKS   = 21
CSV_HEADER      = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")] + ["label"]


def create_landmarker(model_path):
    """Initialise et retourne un HandLandmarker en mode IMAGE."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
    )
    return vision.HandLandmarker.create_from_options(options)


def init_csv(path):
    """Crée le fichier CSV avec son en-tête si il n'existe pas encore."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def load_image(img_path):
    """
    Charge une image depuis le disque et la convertit au format MediaPipe.
    Retourne None si l'image est illisible.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extract_landmarks(mp_image, landmarker):
    """
    Détecte les landmarks de la main dans une image.
    Retourne une liste de 63 coordonnées (x, y, z) x 21 points, ou None si aucune main détectée.
    """
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return [coord for point in lm for coord in (point.x, point.y, point.z)]


def append_row(path, row):
    """Ajoute une ligne au fichier CSV."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def process_dataset(dataset_path, output_csv, landmarker):
    """Parcourt toutes les classes du dataset et écrit les landmarks dans le CSV."""
    for label in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            continue

        print(f"Traitement classe : {label}")
        saved = skipped = 0

        for img_name in os.listdir(class_path):
            mp_image = load_image(os.path.join(class_path, img_name))
            if mp_image is None:
                skipped += 1
                continue

            landmarks = extract_landmarks(mp_image, landmarker)
            if landmarks is None:
                skipped += 1
                continue

            append_row(output_csv, landmarks + [label])
            saved += 1

        print(f"{saved} sauvegardées, {skipped} ignorées")

if __name__ == "__main__":
    print(f"Modèle    : {MODEL_PATH} ({'trouvé' if os.path.exists(MODEL_PATH) else 'INTROUVABLE'})")
    print(f"Dataset   : {DATASET_PATH}")
    print(f"Sortie CSV: {OUTPUT_CSV}\n")

    init_csv(OUTPUT_CSV)
    landmarker = create_landmarker(MODEL_PATH)
    process_dataset(DATASET_PATH, OUTPUT_CSV, landmarker)

    print("\nDataset de landmarks créé !")