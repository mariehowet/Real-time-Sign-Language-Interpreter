import os
import csv
import numpy as np

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(BASE_DIR, "sequence_dataset.csv")

SPLITS = ["train", "val", "test"]


# ===============================
# HEADER CSV
# ===============================
def create_header():
    header = []

    for frame in range(30):
        for point in range(21):
            header.append(f"f{frame}_x{point}")
            header.append(f"f{frame}_y{point}")
            header.append(f"f{frame}_z{point}")

    header.append("label")
    return header


# ===============================
# CONVERT DATASET
# ===============================
rows = []

for split in SPLITS:
    split_path = os.path.join(DATA_DIR, split)

    if not os.path.exists(split_path):
        continue

    print(f"\nLecture : {split}")

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".npy"):

                path = os.path.join(label_path, file)

                data = np.load(path)

                # Si shape = (30,21,3)
                if len(data.shape) == 3:
                    data = data.reshape(30, 63)

                # flatten = 1890 valeurs
                row = data.flatten().tolist()

                row.append(label)

                rows.append(row)

print(f"\nTotal samples : {len(rows)}")


# ===============================
# SAVE CSV
# ===============================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(create_header())
    writer.writerows(rows)

print(f"\nCSV créé : {OUTPUT_CSV}")