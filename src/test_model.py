import torch
import torch.nn as nn
from model import SignCNN
import dataset as dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = SignCNN(num_classes=28).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in dataset.test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy : {correct/total*100:.1f}%")
