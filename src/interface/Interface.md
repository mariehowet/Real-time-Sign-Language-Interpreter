# Interface

Ce document decrit le contrat d'integration du module interface temps reel.

## 1) Architecture

Pipeline execute a chaque frame:

1. L'UI capture une image webcam (`cv2.VideoCapture`).
2. L'UI recupere les donnees main via `HandTracker.get_hand_data(frame)`.
3. L'UI valide/prepare le vecteur landmarks (`prepare_landmarks`).
4. L'UI appelle le modele (`model.predict(...)`).
5. L'UI affiche prediction, confiance, FPS, overlays et sequence.

## 2) Lancer l'interface

Depuis `src/interface`:

```powershell
python interface.py
```

Comportement degrade:

- Si `model.py` ou `model.pth` est absent/invalide: l'UI se lance quand meme, prediction `?` a `0%`.
- Si le tracker leve une erreur (ex: modele MediaPipe manquant), l'application ne peut pas demarrer.

## 3) Dependances

- Python 3.10+
- torch
- opencv-python
- PySide6
- mediapipe

Installation minimale:

```powershell
pip install torch opencv-python PySide6 mediapipe
```

Fichier de tracking attendu:

- `hand_landmarker.task` dans `src/interface/` ou `src/detection/`

## 4) Contrat tracker

L'UI charge automatiquement:

```python
from hand_tracking import HandTracker
tracker = HandTracker()
```

API requise du tracker:

1. `get_hand_data(frame: np.ndarray) -> dict`
2. `draw_landmarks(frame: np.ndarray, raw_landmarks: Any) -> None`
3. `close() -> None`

Format de retour attendu pour `get_hand_data`:

```python
{
  "landmarks": np.ndarray(shape=(63,), dtype=np.float32) | None,
  "bbox": tuple[int, int, int, int] | None,
  "raw_landmarks": Any | None,
}
```

Regles:

- `landmarks` doit contenir exactement `INPUT_SIZE` valeurs (63 par defaut).
- Si aucune main n'est detectee: retourner `landmarks=None`, `bbox=None`, `raw_landmarks=None`.
- `bbox` est optionnel mais recommande pour l'overlay.

## 5) Contrat modele

L'UI instancie le modele depuis `model.py`:

```python
SignModel(input_size=63, num_classes=26)
```

Le modele doit exposer:

```python
predict(landmark_vector: np.ndarray) -> InferenceResult | dict
```

Formats acceptes:

1. `InferenceResult(label="A", confidence=0.92, hand_detected=True)`
2. `{"label": "A", "confidence": 0.92, "hand_detected": True}`

Regles de normalisation appliquees par l'UI:

- Si `hand_detected=False`: affichage `?`.
- Si `confidence < CONFIDENCE_THRESHOLD`: affichage `?`.
- Si `label` n'est pas dans `CLASS_NAMES`: affichage `?`.

## 6) Mode sequence

Quand `sequence_mode=True`, l'UI accumule des lettres stables:

- Stabilite requise: `SEQUENCE_STABILITY_FRAMES = 8`
- Cooldown apres ajout: `SEQUENCE_COOLDOWN_FRAMES = 6`
- Texte affiche: concat de `current_sequence` (ou `-` si vide)

## 7) Constantes par defaut

- `MODEL_PATH = "model.pth"`
- `CLASS_NAMES = A..Z`
- `INPUT_SIZE = 63`
- `CONFIDENCE_THRESHOLD = 0.6`
- `WEBCAM_INDEX = 0`
- `MAX_FPS = 30`
- `WEBCAM_MIRROR = True`
- `BBOX_MARGIN = 16`

## 8) Gestion des erreurs

- `landmarks is None` -> prediction `?`, confiance `0%`.
- Mauvaise taille de vecteur landmarks -> frame ignoree, prediction `?`.
- Erreur dans `model.predict(...)` -> prediction `?`, confiance `0%`.
- Modele non charge -> prediction `?`, confiance `0%`.

## 9) Raccourcis clavier

- `L`: afficher/masquer landmarks
- `B`: afficher/masquer bounding box
- `M`: activer/desactiver mode sequence
- `C`: vider la sequence
- `Q`: quitter
