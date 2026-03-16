import os
import shutil
import random

train_path = "data/splits/train"
val_path = "data/splits/val"
val_ratio = 0.15  # 15 % du train

classes = os.listdir(train_path)

for c in classes:
    class_train_path = os.path.join(train_path, c)
    images = os.listdir(class_train_path)
    random.shuffle(images)

    val_split = int(len(images) * val_ratio)
    val_imgs = images[:val_split]

    dest_val_dir = os.path.join(val_path, c)
    os.makedirs(dest_val_dir, exist_ok=True)

    for img in val_imgs:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(dest_val_dir, img)
        shutil.move(src, dst)  # déplacer pour ne pas garder dans train