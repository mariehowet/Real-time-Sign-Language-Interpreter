# Interface

Ce document definit le contrat d'integration du module interface.

## 1) Architecture choisie

Le pipeline est:

1. L'UI capture une frame camera.
2. L'UI envoie la frame au module tracking.
3. L'UI recupere les landmarks.
4. L'UI envoie les landmarks au module modele.
5. L'UI affiche prediction, confiance, overlays et sequence.

## 2) Lancer l'interface

Depuis le dossier `src/interface`:

```powershell
python interface.py
```

Si `hand_tracking.py` ou `model.py`/`model.pth` est absent, l'UI se lance quand meme en mode degrade.

## 3) Dependances minimales

- Python 3.10+
- torch
- opencv-python
- PySide6

```powershell
pip install torch opencv-python PySide6
```

## 4) Contrat tracking

L'UI tente automatiquement:

```python
from hand_tracking import HandTracker
tracker = HandTracker()
```

Le tracker doit exposer cette API:

### API obligatoire

`get_hand_data(frame) -> dict`

Retour attendu:

```python
{
  "landmarks": np.ndarray(shape=(63,), dtype=np.float32) | None,
  "bbox": tuple[int, int, int, int] | None,
  "raw_landmarks": Any | None,
}
```

### Regles importantes

- `landmarks` doit avoir exactement `INPUT_SIZE` valeurs (par defaut 63).
- Si aucune main n'est detectee, renvoyer `None` pour `landmarks`.
- `bbox` est optionnel mais recommande pour l'overlay.
- Pour dessiner les landmarks natifs, exposer aussi:
  - `self._drawing_utils`
  - `self.hand_connections`
  - `raw_landmarks` dans le retour.

## 5) Contrat modele

L'UI charge `SignModel` depuis `model.py`.
Le modele doit exposer `predict(...)`.

```python
predict(landmark_vector: np.ndarray) -> InferenceResult | dict
```

Formats acceptes:

1. `InferenceResult(label="A", confidence=0.92, hand_detected=True)`
2. `{"label": "A", "confidence": 0.92, "hand_detected": True}`

L'UI applique ensuite un seuil de confiance (`CONFIDENCE_THRESHOLD`).

### Construction du modele

L'UI appelle:

```python
SignModel(input_size=63, num_classes=26)
```

## 6) Fichiers et constantes

- Poids par defaut: `model.pth` (dans le dossier interface)
- Classes par defaut: `A..Z`
- Taille entree par defaut: `INPUT_SIZE = 63`
- Seuil par defaut: `CONFIDENCE_THRESHOLD = 0.6`
- FPS max UI: `MAX_FPS = 30` (modifiable dans `interface_core.py`)

## 7) Gestion des erreurs

- Si `landmarks` vaut `None`: l'UI affiche `?` avec `0%`.
- Si la confiance est sous `CONFIDENCE_THRESHOLD`: l'UI affiche `?`.
- Si le tracker est absent: pas de landmarks, pas de bbox, prediction a `?`.
- Si le modele est absent ou echoue: prediction a `?` avec `0%`.

## 8) Raccourcis clavier UI

- `L`: afficher/masquer landmarks
- `B`: afficher/masquer bounding box
- `M`: activer/desactiver mode sequence
- `C`: vider la sequence
- `Q`: quitter
