import torch
from src.models.test_landmarks import LandmarkClassifier
from src.models.train_only_words import SequenceClassifier

idx_to_sign = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'Space', 20: 'T', 21: 'U', 22: 'V',
    23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
}

idx_to_words = {
    0: 'HELLO',
    1: 'PLEASE',
    2: 'SEE YOU LATER',
    3: 'THANK YOU',
}

MODES = ("Letter", "Word")

class SignModel:
    def __init__(self, input_size: int, num_classes: int, mode: str="Letter"):
        if mode not in MODES:
            raise ValueError(f"mode doit être l'un de {MODES}")

        self.mode = mode
        self.input_size = input_size
        self.num_classes = num_classes
        self._label_map = idx_to_sign if mode == "Letter" else idx_to_words

        self.model = (
            LandmarkClassifier(input_size, num_classes)
            if mode == "Letter"
            else SequenceClassifier(input_size, num_classes)
        )

    def load(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return self

    def predict(self, landmark_vector):
        if landmark_vector is None:
            return {
                "label": "?",
                "confidence": 0.0,
                "hand_detected": False
            }

        x = torch.tensor(landmark_vector, dtype=torch.float32)
        if x.shape[0] != self.input_size:
            raise ValueError(
                f"Taille du vecteur incorrecte : attendu {self.input_size}, "
                f"Taille reçu {x.shape[0]}"
            )

        x = x.unsqueeze(0)

        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        return {
            "label": self._label_map.get(pred.item(), "?"),
            "confidence": conf.item(),
            "hand_detected": True
        }