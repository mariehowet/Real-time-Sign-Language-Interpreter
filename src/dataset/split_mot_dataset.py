import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH   = os.path.join(BASE_DIR, "data", "raw")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train")
VAL_PATH   = os.path.join(BASE_DIR, "data", "val")
TEST_PATH  = os.path.join(BASE_DIR, "data", "test")

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2


def split_class(label):
    src = os.path.join(RAW_PATH, label)

    files = os.listdir(src)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": files[:train_end],
        "val":   files[train_end:val_end],
        "test":  files[val_end:]
    }

    for split_name, file_list in splits.items():
        dst_dir = os.path.join(BASE_DIR, "data", split_name, label)
        os.makedirs(dst_dir, exist_ok=True)

        for f in file_list:
            shutil.copy(
                os.path.join(src, f),
                os.path.join(dst_dir, f)
            )

    print(f"{label} → train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")


def main():
    for label in os.listdir(RAW_PATH):
        split_class(label)

    print("\n✅ Split terminé")


if __name__ == "__main__":
    main()