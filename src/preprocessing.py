from torchvision import transforms

train_transforms = transforms.Compose([
    
    transforms.Resize((128,128)),
    
    transforms.RandomHorizontalFlip(),
    
    transforms.RandomRotation(15),
    
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    
    transforms.ToTensor(),
    
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

val_test_transforms = transforms.Compose([
    
    transforms.Resize((128,128)),
    
    transforms.ToTensor(),
    
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])