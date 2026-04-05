import torch
import torch.nn as nn

class SignCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(SignCNN, self).__init__() #initialise le parent

        self.features = nn.Sequential(
            #détection les formes simples 
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64

            #détection des formes plus complexes 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            #détection des formes compliquées (doigts)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
        )
        #récupération des caractéristiques extraites pour déterminer à quelle lettres elles appartiennent
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
