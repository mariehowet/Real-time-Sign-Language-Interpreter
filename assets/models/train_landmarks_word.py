import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# Modèles
# ─────────────────────────────────────────────
class LandmarkClassifier(nn.Module):
    """MLP pour la reconnaissance de lettres (63 features)."""
    def __init__(self, input_size=63, num_classes=28):
        super(LandmarkClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class SequenceClassifier(nn.Module):
    """MLP pour la reconnaissance de mots/gestes (1890 features = 30 frames × 21 landmarks × 3 coords)."""
    def __init__(self, input_size=1890, num_classes=4):
        super(SequenceClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────
# Fonction d'entraînement avec early stopping
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader, train_size, criterion, optimizer, device,
                epochs=100, patience=5, label=""):
    """
    Entraîne le modèle avec early stopping sur la val_loss.
    - patience : nombre d'epochs sans amélioration avant arrêt
    - Restaure automatiquement les meilleurs poids à la fin
    """
    best_val_loss   = float("inf")
    patience_counter = 0
    best_weights    = None

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss, train_correct = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        # ── Validation ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs   = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)

        print(f"  [{label}] Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {train_loss/len(train_loader):.3f} "
              f"| Train Acc: {train_correct/train_size*100:.1f}% "
              f"| Val Loss: {val_loss:.3f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  → Early stopping à l'epoch {epoch+1} "
                      f"(patience={patience}, meilleure val_loss={best_val_loss:.3f})\n")
                break

    # Restaure les meilleurs poids
    if best_weights is not None:
        model.load_state_dict(best_weights)


# ─────────────────────────────────────────────
# Fonction d'évaluation + matrice de confusion
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, test_size, device, classes, title):
    model.eval()
    all_preds, all_labels = [], []
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds   = outputs.argmax(1)
            correct    += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print(f"  → Test Accuracy : {correct/test_size*100:.1f}%\n")

    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='RdPu')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    ax.set_xlabel('Prédictions', fontsize=13)
    ax.set_ylabel('Valeurs réelles', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUtilisation de : {device}\n")
    print("=" * 55)


    # ──────────────────────────────────────────
    # 1. LETTRES — landmarks_dataset.csv
    # ──────────────────────────────────────────
    print("[ 1/2 ] LETTRES — chargement des données")
    df_letters = pd.read_csv("data/landmarks_dataset.csv")

    X_letters = df_letters.drop("label", axis=1).values.astype(np.float32)
    y_letters = df_letters["label"].values

    encoder_letters  = LabelEncoder()
    y_letters_enc    = encoder_letters.fit_transform(y_letters)
    num_classes_letters = len(encoder_letters.classes_)
    print(f"  → {len(df_letters)} échantillons, {num_classes_letters} classes : {list(encoder_letters.classes_)}\n")

    dataset_letters    = LandmarkDataset(X_letters, y_letters_enc)
    train_size_letters = int(0.75 * len(dataset_letters))
    val_size_letters   = int(0.10 * len(dataset_letters))
    test_size_letters  = len(dataset_letters) - train_size_letters - val_size_letters
    train_letters, val_letters, test_letters = random_split(
        dataset_letters,
        [train_size_letters, val_size_letters, test_size_letters],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader_letters = DataLoader(train_letters, batch_size=32, shuffle=True,  drop_last=True)
    val_loader_letters   = DataLoader(val_letters,   batch_size=32, shuffle=False)
    test_loader_letters  = DataLoader(test_letters,  batch_size=32, shuffle=False)
    print(f"  → Train : {train_size_letters} | Val : {val_size_letters} | Test : {test_size_letters}\n")

    print("  Entraînement lettres...")
    model_letters = LandmarkClassifier(input_size=X_letters.shape[1], num_classes=num_classes_letters).to(device)
    criterion     = nn.CrossEntropyLoss()
    optimizer_letters = torch.optim.Adam(model_letters.parameters(), lr=0.001)
    train_model(model_letters, train_loader_letters, val_loader_letters, train_size_letters,
                criterion, optimizer_letters, device, epochs=10, patience=5, label="Lettres")

    print("\n  Évaluation lettres...")
    evaluate_model(model_letters, test_loader_letters, test_size_letters, device,
                   encoder_letters.classes_, "Matrice de confusion — Lettres")


    # ──────────────────────────────────────────
    # 2. MOTS — sequence_dataset.csv
    # ──────────────────────────────────────────
    print("=" * 55)
    print("[ 2/2 ] MOTS — chargement des données")
    df_words = pd.read_csv("data/sequence_dataset.csv")

    X_words = df_words.drop("label", axis=1).values.astype(np.float32)
    y_words = df_words["label"].values

    encoder_words  = LabelEncoder()
    y_words_enc    = encoder_words.fit_transform(y_words)
    num_classes_words = len(encoder_words.classes_)
    print(f"  → {len(df_words)} échantillons, {num_classes_words} classes : {list(encoder_words.classes_)}\n")
    print(f"  → Taille du vecteur d'entrée : {X_words.shape[1]} features (attendu ~1890)\n")

    dataset_words    = LandmarkDataset(X_words, y_words_enc)
    train_size_words = int(0.75 * len(dataset_words))
    val_size_words   = int(0.10 * len(dataset_words))
    test_size_words  = len(dataset_words) - train_size_words - val_size_words
    train_words, val_words, test_words = random_split(
        dataset_words,
        [train_size_words, val_size_words, test_size_words],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader_words = DataLoader(train_words, batch_size=32, shuffle=True,  drop_last=True)
    val_loader_words   = DataLoader(val_words,   batch_size=32, shuffle=False)
    test_loader_words  = DataLoader(test_words,  batch_size=32, shuffle=False)
    print(f"  → Train : {train_size_words} | Val : {val_size_words} | Test : {test_size_words}\n")

    print("  Entraînement mots...")
    model_words = SequenceClassifier(input_size=X_words.shape[1], num_classes=num_classes_words).to(device)
    optimizer_words = torch.optim.Adam(model_words.parameters(), lr=0.001)
    train_model(model_words, train_loader_words, val_loader_words, train_size_words,
                criterion, optimizer_words, device, epochs=30, patience=10, label="Mots")

    print("\n  Évaluation mots...")
    evaluate_model(model_words, test_loader_words, test_size_words, device,
                   encoder_words.classes_, "Matrice de confusion — Mots")


    # ──────────────────────────────────────────
    # Sauvegarde
    # ──────────────────────────────────────────
    print("=" * 55)
    torch.save(model_letters.state_dict(), "model_landmarks.pth")
    print("  → Modèle lettres sauvegardé dans model_landmarks.pth")
    torch.save(model_words.state_dict(), "landmarks_word_model.pth")
    print("  → Modèle mots   sauvegardé dans landmarks_word_model.pth\n")
