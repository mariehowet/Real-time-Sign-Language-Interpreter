import os
from torchvision import datasets
from torch.utils.data import DataLoader
from preprocessing import train_transforms, val_test_transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "splits", "train")
test_path = os.path.join(BASE_DIR, "data", "splits", "test")

train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_path, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Si tu veux un val loader provisoire (optionnel)
val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)