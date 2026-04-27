import os
import sys

# ─── PATH SETUP ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Imports projet
from split_dataset import split_dataset
from images_to_landmarks import create_landmarker, init_csv, process_dataset

# ─── CONFIG ────────────────────────────────────────────────
TRAIN_PATH = os.path.join(BASE_DIR, "data", "splits", "train")
VAL_PATH   = os.path.join(BASE_DIR, "data", "splits", "val")
TEST_PATH  = os.path.join(BASE_DIR, "data", "splits", "test")

OUTPUT_CSV = os.path.join(BASE_DIR, "landmarks_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "hand_landmarker.task")

VAL_RATIO  = 0.10
TEST_RATIO = 0.20


# ─── PIPELINE ──────────────────────────────────────────────
def check_model():
    print("\n[CHECK] Modèle MediaPipe")
    if not os.path.exists(MODEL_PATH):
        print("ERREUR : hand_landmarker.task introuvable !")
        print("Télécharge-le et place-le à la racine du projet")
        exit()
    print("Modèle trouvé")


def split_if_needed():
    print("\n[STEP 1] Rééquilibrage dataset (train → val/test)")

    # Vérifie si val/test sont vides
    if len(os.listdir(VAL_PATH)) == 0 or len(os.listdir(TEST_PATH)) == 0:
        print("Split en cours...")
        split_dataset(TRAIN_PATH, VAL_PATH, VAL_RATIO, "val")
        split_dataset(TRAIN_PATH, TEST_PATH, TEST_RATIO, "test")
    else:
        print("Déjà split, on skip")


def extract_landmarks():
    print("\n[STEP 2] Extraction des landmarks")

    init_csv(OUTPUT_CSV)

    landmarker = create_landmarker(MODEL_PATH)

    process_dataset(TRAIN_PATH, OUTPUT_CSV, landmarker)

    print("Landmarks extraits dans :", OUTPUT_CSV)


def main():
    print("PIPELINE DATASET")

    check_model()
    split_if_needed()
    extract_landmarks()

    print("\nPipeline terminé avec succès !")


# ─── EXECUTION ─────────────────────────────────────────────
if __name__ == "__main__":
    main()