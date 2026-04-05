import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#dataset et dataloader
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#modèle
class LandmarkClassifier(nn.Module):
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


if __name__=="__main__":
    #csv load
    print("Chargement des données")
    df = pd.read_csv("data/landmarks_dataset.csv")

    X = df.drop("label", axis=1).values.astype(np.float32)  # (N, 63)
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    print(f"  → {len(df)} échantillons, {num_classes} classes : {list(encoder.classes_)}\n")


    dataset    = LandmarkDataset(X, y_encoded)
    train_size = int(0.85 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    print(f"  → Train : {train_size} échantillons | Test : {test_size} échantillons\n")

    #entraînement
    print("Entraînement")

    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  → Utilisation de : {device}\n")

    model     = LandmarkClassifier(input_size=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
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

        print(f"  Epoch {epoch+1}/30 "
              f"| Loss: {train_loss/len(train_loader):.3f} "
              f"| Train Acc: {train_correct/train_size*100:.1f}%")

    torch.save(model.state_dict(), "model_landmarks.pth")
    print("\n  → Modèle sauvegardé dans model_landmarks.pth\n")

    #test + matrice confusion +evaluation
    print("Évaluation")

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

    cm      = confusion_matrix(all_labels, all_preds)
    classes = encoder.classes_

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
    ax.set_title('Matrice de confusion — LandmarkClassifier', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
