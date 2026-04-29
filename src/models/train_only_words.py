import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#modèle
class SequenceClassifier(nn.Module):
    """MLP pour séquences (30 frames × 21 landmarks × 3 coords = 1890 features)."""
    def __init__(self, input_size=1890, num_classes=4):
        super().__init__()

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


#train
def train_model(model, train_loader, val_loader, train_size, device,
                epochs=30, patience=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):

        #train
        model.train()
        train_loss, train_correct = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        #validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {train_loss/len(train_loader):.3f} "
              f"| Train Acc: {train_correct/train_size*100:.1f}% "
              f"| Val Loss: {val_loss:.3f}")

        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n→ Early stopping (epoch {epoch+1})\n")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)


#evaluation
def evaluate_model(model, test_loader, test_size, device, classes):

    model.eval()
    all_preds, all_labels = [], []
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)

            correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print(f"\nTest Accuracy : {correct/test_size*100:.1f}%\n")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='RdPu')
    plt.title("Matrice de confusion — Mots")
    plt.colorbar()

    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j],
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()


#main
if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUtilisation de : {device}\n")

    #load
    df = pd.read_csv("data/sequence_dataset.csv")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    #encodage
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    classes = encoder.classes_

    print(f"{len(df)} échantillons | {len(classes)} classes : {list(classes)}")
    print(f"Taille entrée : {X.shape[1]} features\n")

    #dataset
    dataset = SequenceDataset(X, y_enc)

    train_size = int(0.75 * len(dataset))
    val_size   = int(0.10 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=32)
    test_loader  = DataLoader(test_set, batch_size=32)

    print(f"Train : {train_size} | Val : {val_size} | Test : {test_size}\n")

    #modèle
    model = SequenceClassifier(input_size=X.shape[1],
                               num_classes=len(classes)).to(device)

    #train
    print("Entraînement...")
    train_model(model, train_loader, val_loader, train_size, device)

    #evaluation
    print("Évaluation...")
    evaluate_model(model, test_loader, test_size, device, classes)

    #save
    torch.save(model.state_dict(), "model_words.pth")
    print("\nModèle sauvegardé → model_words.pth")
