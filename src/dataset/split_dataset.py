import os
import random
import shutil


# Config
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data/splits/train")
VAL_PATH   = os.path.join(BASE_DIR, "data/splits/val")
TEST_PATH  = os.path.join(BASE_DIR, "data/splits/test")

VAL_RATIO  = 0.10  # 10 % du train → val
TEST_RATIO = 0.20  # 20 % du train → test

def move_images(images, src_dir, dst_dir):
    """Déplace une liste d'images d'un dossier source vers un dossier destination."""
    os.makedirs(dst_dir, exist_ok=True)
    for img in images:
        shutil.move(os.path.join(src_dir, img), os.path.join(dst_dir, img))


def split_class(class_name, src_path, dst_path, ratio):
    """
    Déplace une fraction des images d'une classe d' un dossier source vers un dossier destination.
    Retourne le nombre d'images déplacées.
    """
    src_dir = os.path.join(src_path, class_name)
    dst_dir = os.path.join(dst_path, class_name)

    images = os.listdir(src_dir)
    random.shuffle(images)

    count = int(len(images) * ratio)
    move_images(images[:count], src_dir, dst_dir)
    return count


def split_dataset(src_path, dst_path, ratio, label):
    """Applique split_class sur toutes les classes d'un dossier."""
    for class_name in sorted(os.listdir(src_path)):
        count = split_class(class_name, src_path, dst_path, ratio)
        print(f"  {class_name}: {count} images → {label}")
    print(f"Split '{label}' terminé.\n")



if __name__ == "__main__":
    print("Train → Val")
    split_dataset(TRAIN_PATH, VAL_PATH, VAL_RATIO, "val")

    print("Train → Test")
    split_dataset(TRAIN_PATH, TEST_PATH, TEST_RATIO, "test")