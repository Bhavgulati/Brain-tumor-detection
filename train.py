import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import ImageFolder

# Config
TRAIN_DIR = './Training'
TEST_DIR = './Testing'
MODEL_SAVE_PATH = './models/bt_resnet50_model.pt'
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.00001

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Transforms
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = ImageFolder(TEST_DIR, transform=transform)

print(f"Classes found: {train_dataset.classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Testing images: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model (same architecture as app.py)
resnet_model = resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential
    nn.Linear(n_inputs, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 4),
)

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)

# Optimizer & Loss
optimizer = Adam(resnet_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    resnet_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_acc = 100 * correct / total
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Training Accuracy: {train_acc:.2f}% | Avg Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    resnet_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet_model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Validation Accuracy: {val_acc:.2f}%\n")

# Save model
torch.save(resnet_model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
print("Now run: python app.py")