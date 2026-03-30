from torchvision import transforms

# Config
IMAGE_SIZE  = (128, 128)
NORM_MEAN   = [0.5, 0.5, 0.5]
NORM_STD    = [0.5, 0.5, 0.5]

base_transform = [
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
]


augmentation = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
]


train_transforms    = transforms.Compose([base_transform[0]] + augmentation + base_transform[1:]) #[0:] and [1:] allow to applie augmentation between steps of base_transform
val_test_transforms = transforms.Compose(base_transform)