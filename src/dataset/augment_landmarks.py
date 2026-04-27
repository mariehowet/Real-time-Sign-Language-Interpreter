import os
import numpy as np
import random

# ─── Config ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
AUG_PER_SAMPLE = 5  # nombre d'augmentations par fichier


# ─── Augmentations de base ─────────────────────────────

def add_noise(landmarks, noise_level=0.01):
    return landmarks + np.random.normal(0, noise_level, landmarks.shape)


def translate(landmarks, shift_range=0.05):
    shift = np.random.uniform(-shift_range, shift_range, 3)
    return landmarks + np.tile(shift, 21)


def scale(landmarks, scale_range=(0.9, 1.1)):
    scale_factor = random.uniform(*scale_range)
    return landmarks * scale_factor


def rotate_z(landmarks, angle_range=(-15, 15)):
    angle = np.radians(random.uniform(*angle_range))
    cos, sin = np.cos(angle), np.sin(angle)

    rotated = landmarks.copy()
    for i in range(0, len(landmarks), 3):
        x, y = landmarks[i], landmarks[i + 1]
        rotated[i]     = x * cos - y * sin
        rotated[i + 1] = x * sin + y * cos

    return rotated


def augment_frame(frame):
    """Augmentation d'une seule frame (63 valeurs)"""
    aug = frame.copy()

    if random.random() < 0.8:
        aug = add_noise(aug)

    if random.random() < 0.8:
        aug = translate(aug)

    if random.random() < 0.5:
        aug = scale(aug)

    if random.random() < 0.5:
        aug = rotate_z(aug)

    return aug


def augment_sequence(sequence):
    """Augmente une séquence (mot) frame par frame"""
    return np.array([augment_frame(frame) for frame in sequence])


def augment_auto(data):
    """
    Détecte automatiquement :
    - lettre (63,) → IGNORÉ
    - mot (N,63) → AUGMENTÉ
    """
    if len(data.shape) == 1:
        return None  # ❌ lettre → pas d'augmentation

    elif len(data.shape) == 2:
        return augment_sequence(data)  # ✔ mot

    else:
        raise ValueError(f"Format inconnu: {data.shape}")


# ─── Pipeline principal ─────────────────────────────────

def augment_dataset(split="train"):
    split_path = os.path.join(DATA_PATH, split)

    print(f"\nAugmentation du dataset : {split_path}")

    for label in os.listdir(split_path):
        class_path = os.path.join(split_path, label)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.endswith(".npy")]

        print(f"\nClasse : {label} ({len(files)} fichiers)")

        for file in files:
            file_path = os.path.join(class_path, file)
            data = np.load(file_path)

            aug_data = augment_auto(data)

            # ❌ si lettre → on skip
            if aug_data is None:
                continue

            # ✔ sauvegarde augmentations
            for i in range(AUG_PER_SAMPLE):
                new_name = file.replace(".npy", f"_aug{i}.npy")
                new_path = os.path.join(class_path, new_name)

                np.save(new_path, aug_data)

    print("\n✅ Augmentation terminée !")


# ─── Lancement ─────────────────────────────────────────

if __name__ == "__main__":
    augment_dataset("train")  # ⚠️ uniquement train