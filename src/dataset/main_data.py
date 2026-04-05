import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from split_dataset import split_dataset
from images_to_landmarks import create_landmarker, init_csv, process_dataset
from preprocessing import train_transforms, val_test_transforms


# Config
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train")
VAL_PATH   = os.path.join(BASE_DIR, "data", "val")
TEST_PATH  = os.path.join(BASE_DIR, "data", "test")
OUTPUT_CSV = os.path.join(BASE_DIR, "landmarks_dataset.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "hand_landmarker.task")


VAL_RATIO    = 0.10
TEST_RATIO   = 0.20


if __name__ == "__main__":
    # Split dataset
    print("Step 1 : Split dataset")
    split_dataset(TRAIN_PATH, VAL_PATH,  VAL_RATIO,  "val")
    split_dataset(TRAIN_PATH, TEST_PATH, TEST_RATIO, "test")

    # Extraction of landmarks
    print("Step 2 : Extraction of landmarks")
    init_csv(OUTPUT_CSV)
    landmarker = create_landmarker(MODEL_PATH)
    process_dataset(TRAIN_PATH, OUTPUT_CSV, landmarker)

    # Preprocessing
    print("Step 3 : Preprocessing")
    print(f"Train transforms    : {train_transforms}")
    print(f"Val_test transforms : {val_test_transforms}")
    print("Transforms load successfully")

    print("\n End")