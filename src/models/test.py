
import dataset as dataset

# Tester un batch du train_loader
for images, labels in dataset.train_loader:
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels)
    break

# Tester un batch du val_loader
for images, labels in dataset.val_loader:
    print("Validation batch images shape:", images.shape)
    print("Validation batch labels:", labels)
    break

# Tester un batch du test_loader
for images, labels in dataset.test_loader:
    print("Test batch images shape:", images.shape)
    print("Test batch labels:", labels)
    break

print("Tous les DataLoaders testés avec succès !")