import torch
import torch.nn as nn
from torch.optim import Adam
from model import SignCNN
import dataset as dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Utilisation de : {device}")

model = SignCNN(num_classes=28).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # --- Entraînement ---
    model.train()
    train_loss, train_correct = 0, 0
    for images, labels in dataset.train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in dataset.val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss/len(dataset.train_loader):.3f} "
          f"| Train Acc: {train_correct/len(dataset.train_dataset)*100:.1f}% "
          f"| Val Acc: {val_correct/len(dataset.train_dataset)*100:.1f}%")

# Sauvegarder le modèle
torch.save(model.state_dict(), "model.pth")
print("Modèle sauvegardé ")
